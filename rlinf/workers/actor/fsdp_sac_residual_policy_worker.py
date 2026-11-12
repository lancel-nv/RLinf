# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Residual SAC learner using RLinf's replay, FSDP and checkpoint lifecycle."""

import asyncio
import os
import queue
from dataclasses import asdict

import torch

from rlinf.algorithms.sac_residual import (
    ResidualUpdateBudget,
    residual_actor_loss,
    residual_critic_loss,
)
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.models.embodiment.mlp_policy.sac_residual_policy import SACResidualConfig
from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.actor.async_fsdp_sac_policy_worker import AsyncEmbodiedSACFSDPPolicy
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class SACResidualFSDPPolicy(EmbodiedSACFSDPPolicy):
    """Train FP32 heads with one learner and one decision per replay row."""

    def setup_sac_components(self) -> None:
        if self._world_size != 1:
            raise ValueError(
                "Residual SAC currently supports exactly one actor worker."
            )
        super().setup_sac_components()
        settings = {
            key: self.cfg.actor.model[key]
            for key in SACResidualConfig.__dataclass_fields__
            if key in self.cfg.actor.model
        }
        self.target_entropy = SACResidualConfig(**settings).target_entropy(
            self.cfg.algorithm.entropy_tuning.get("target_entropy")
        )
        self.update_budget = ResidualUpdateBudget(
            **dict(self.cfg.algorithm.residual_schedule)
        )

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel) -> None:
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        split_num = compute_split_num(
            send_num, self._component_placement.get_world_size("actor")
        )
        for _ in range(split_num):
            trajectory = await input_channel.get(async_op=True).async_wait()
            self._ingest_trajectory(trajectory)

    def _ingest_trajectory(self, trajectory: Trajectory) -> None:
        rows = self.transition_rows(trajectory)
        self.replay_buffer.add_trajectories(rows)
        self.update_budget.add(len(rows))

    def transition_rows(self, trajectory: Trajectory) -> list[Trajectory]:
        """Keep actual chunk decisions, including the final observation at timeout."""
        flat = self.replay_buffer._flatten_trajectory(trajectory)
        time_steps, batch_size = trajectory.rewards.shape[:2]
        rows = []
        for env_idx in range(batch_size):
            for step in range(time_steps):
                index = step * batch_size + env_idx
                row = Trajectory(
                    max_episode_length=1, model_weights_id=trajectory.model_weights_id
                )
                for name in (
                    "actions",
                    "rewards",
                    "terminations",
                    "truncations",
                    "dones",
                    "curr_obs",
                    "next_obs",
                ):
                    value = flat[name]
                    if isinstance(value, dict):
                        value = {
                            key: tensor[index : index + 1].unsqueeze(0).clone().cpu()
                            for key, tensor in value.items()
                        }
                    else:
                        value = value[index : index + 1].unsqueeze(0).clone().cpu()
                    setattr(row, name, value)
                rows.append(row)
                if row.dones.any() and not self.cfg.env.train.auto_reset:
                    break
        return rows

    def forward_critic(self, batch):
        return residual_critic_loss(
            self.model,
            self.target_model,
            batch,
            self.entropy_temp.compute_alpha(),
            self.cfg.algorithm.gamma,
            self.cfg.algorithm.backup_entropy,
            self.critic_sample_generator,
            bootstrap_time_limits=self.cfg.algorithm.get("bootstrap_time_limits", True),
        )

    def forward_actor(self, batch, actor_parameters):
        return residual_actor_loss(
            self.model,
            batch["curr_obs"],
            self.entropy_temp.compute_alpha(),
            self.critic_sample_generator,
            actor_parameters=actor_parameters,
        )

    def update_one_epoch(self, train_actor: bool = True):
        batch = put_tensor_device(next(self.buffer_dataloader_iter), device=self.device)
        critic_parameters = [
            p for group in self.qf_optimizer.param_groups for p in group["params"]
        ]
        actor_parameters = [
            p for group in self.optimizer.param_groups for p in group["params"]
        ]

        def step_role(loss, optimizer, scheduler, parameters, max_norm):
            if not torch.isfinite(loss):
                raise FloatingPointError("Expected a finite residual SAC loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return norm.detach().item()

        self.qf_optimizer.zero_grad(set_to_none=True)
        critic_loss, metrics = self.forward_critic(batch)
        metrics = {f"critic/{key}": value for key, value in metrics.items()}
        metrics["critic/grad_norm"] = step_role(
            critic_loss,
            self.qf_optimizer,
            self.qf_lr_scheduler,
            critic_parameters,
            self.cfg.actor.critic_optim.clip_grad,
        )
        metrics["sac/critic_loss"] = critic_loss.detach().item()

        # update_step counts completed critic updates, including restored updates.
        actor_updated = (
            train_actor
            and self.update_step
            >= self.cfg.algorithm.get("residual_schedule", {}).get(
                "critic_warmup_updates", 0
            )
            and (self.update_step + 1) % self.critic_actor_ratio == 0
        )
        if actor_updated:
            self.optimizer.zero_grad(set_to_none=True)
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            try:
                actor_loss, log_prob, actor_metrics = self.forward_actor(
                    batch, actor_parameters
                )
                metrics.update(actor_metrics)
                metrics["actor/grad_norm"] = step_role(
                    actor_loss,
                    self.optimizer,
                    self.lr_scheduler,
                    actor_parameters,
                    self.cfg.actor.optim.clip_grad,
                )
            finally:
                for parameter in critic_parameters:
                    parameter.requires_grad_(True)
            # Use the same sample as the actor update, including the scale Jacobian.
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (
                self.entropy_temp.compute_alpha()
                * (-log_prob.detach() - self.target_entropy)
            ).mean()
            metrics["alpha/grad_norm"] = step_role(
                alpha_loss,
                self.alpha_optimizer,
                self.alpha_lr_scheduler,
                list(self.entropy_temp.parameters()),
                self.cfg.algorithm.entropy_tuning.optim.clip_grad,
            )
            metrics.update(
                {
                    "sac/actor_loss": actor_loss.detach().item(),
                    "sac/alpha_loss": alpha_loss.detach().item(),
                    "actor/entropy": -log_prob.detach().mean().item(),
                }
            )
        self.soft_update_target_model()
        metrics.update(
            {
                "sac/alpha": self.entropy_temp.alpha,
                "sac/target_entropy": self.target_entropy,
                "sac/actor_updated": float(actor_updated),
                "sac/critic_updates": self.update_step + 1,
                "actor/lr": self.optimizer.param_groups[0]["lr"],
                "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            }
        )
        return metrics

    @Worker.timer("run_training")
    def run_training(self):
        updates = self.update_budget.available()
        if updates == 0:
            return {
                "replay/collected_transitions": self.update_budget.collected,
                "sac/critic_updates": self.update_step,
            }
        self.model.train()
        metrics = {}
        for _ in range(updates):
            append_to_dict(metrics, self.update_one_epoch())
            self.update_step += 1
            self.update_budget.updates = self.update_step
        return self._finish_training_metrics(metrics)

    def _finish_training_metrics(self, metrics: dict) -> dict:
        append_to_dict(
            metrics,
            {
                "replay/collected_transitions": self.update_budget.collected,
                "sac/pending_updates": self.update_budget.pending(),
            },
        )
        reduced = self.process_train_metrics(metrics)
        reduced["sac/critic_updates"] = self.update_step
        return reduced

    def save_checkpoint(self, save_base_path, step):
        super().save_checkpoint(save_base_path, step)
        torch.save(
            {
                "budget": asdict(self.update_budget),
                "critic_rng": self.critic_sample_generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "device_rng": self.torch_platform.get_rng_state(),
            },
            os.path.join(save_base_path, f"residual_state_rank_{self._rank}.pt"),
        )

    def load_checkpoint(self, load_base_path):
        super().load_checkpoint(load_base_path)
        state = torch.load(
            os.path.join(load_base_path, f"residual_state_rank_{self._rank}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.update_budget.restore(state["budget"])
        self.update_step = self.update_budget.updates
        self.critic_sample_generator.set_state(state["critic_rng"])
        torch.set_rng_state(state["torch_rng"])
        self.torch_platform.set_rng_state(state["device_rng"])


class AsyncSACResidualFSDPPolicy(AsyncEmbodiedSACFSDPPolicy, SACResidualFSDPPolicy):
    """Receive through native async SAC and train from residual transition credits."""

    def _drain_received_trajectories(self, max_trajectories: int | None = None) -> None:
        if getattr(self, "_recv_queue", None) is None:
            return
        processed = 0
        while max_trajectories is None or processed < max_trajectories:
            try:
                trajectory = self._recv_queue.get_nowait()
            except queue.Empty:
                break
            self._ingest_trajectory(trajectory)
            processed += 1

    @Worker.timer("run_training")
    async def run_training(self) -> dict:
        self._drain_received_trajectories(
            max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 1024)
        )
        limit = self.cfg.get("runner", {}).get("max_critic_updates")
        if limit is not None and self.update_step >= limit:
            return {"sac/critic_updates": self.update_step, "sac/updates_this_call": 0}
        updates = self.update_budget.available()
        if limit is not None:
            updates = min(updates, limit - self.update_step)
        for key in ("save_critic_interval", "val_critic_interval"):
            interval = self.cfg.get("runner", {}).get(key)
            if interval is not None:
                # Return exact milestone weights even when a call crosses the boundary.
                updates = min(updates, interval - self.update_step % interval)
        if updates == 0:
            # An empty result makes the async runner wait without advancing its step.
            return {}
        self.model.train()
        metrics = {}
        for _ in range(updates):
            await asyncio.sleep(0)
            append_to_dict(metrics, self.update_one_epoch())
            self.update_step += 1
            self.update_budget.updates = self.update_step
        result = self._finish_training_metrics(metrics)
        result["sac/updates_this_call"] = updates
        return result
