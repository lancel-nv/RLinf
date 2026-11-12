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

"""Residual SAC objectives and frozen-feature rollout boundaries."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import ForwardType

RESIDUAL_OBS_KEYS = ("state", "visual", "base_action")


def extract_residual_obs(
    forward_inputs: dict[str, Any], *, transition: bool = False
) -> dict[str, torch.Tensor]:
    """Read the cached proposal and features for a replay transition."""
    prefix = "residual_next_" if transition else "residual_"
    obs = {key: forward_inputs[prefix + key] for key in RESIDUAL_OBS_KEYS}
    if prefix + "candidates" in forward_inputs:
        obs["candidates"] = forward_inputs[prefix + "candidates"]
    if prefix + "privileged_state" in forward_inputs:
        obs["privileged_state"] = forward_inputs[prefix + "privileged_state"]
    return obs


@torch.no_grad()
def predict_residual_actions(
    policy_model: Any,
    feature_model: Any,
    env_obs: dict,
    final_obs: dict | None = None,
    mode: str = "train",
    *,
    visual_encoder: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compose in normalized coordinates and decode once for the environment."""
    proposal_kwargs = {}
    if hasattr(policy_model.config, "num_candidates"):
        proposal_kwargs["num_candidates"] = policy_model.config.num_candidates
    obs, model_actions, model_state = feature_model.extract_residual_obs(
        {k: v for k, v in env_obs.items() if k != "privileged_state"},
        visual_encoder=visual_encoder,
        **proposal_kwargs,
    )
    if "privileged_state" in env_obs:
        obs = {**obs, "privileged_state": env_obs["privileged_state"].detach().clone()}
    normalized_actions, result = policy_model.predict_action_batch(obs, mode=mode)
    chunk_len, action_dim = normalized_actions.shape[1:]
    model_actions = model_actions.clone()
    model_actions[:, :chunk_len, :action_dim] = normalized_actions
    actions = feature_model.decode_actions(model_actions, model_state)
    next_obs = obs
    if final_obs is not None:
        next_obs, _, _ = feature_model.extract_residual_obs(
            {k: v for k, v in final_obs.items() if k != "privileged_state"},
            visual_encoder=visual_encoder,
            **proposal_kwargs,
        )
    keys = RESIDUAL_OBS_KEYS + (("candidates",) if "candidates" in obs else ())
    for key in keys:
        result["forward_inputs"]["residual_" + key] = obs[key]
        result["forward_inputs"]["residual_next_" + key] = next_obs[key]
    # Cache state from the same observation boundary; the frozen VLA never sees it.
    for prefix, source in (
        ("residual_", env_obs),
        ("residual_next_", env_obs if final_obs is None else final_obs),
    ):
        if "privileged_state" in source:
            result["forward_inputs"][prefix + "privileged_state"] = (
                source["privileged_state"].detach().clone()
            )
    return actions, result


def random_min_q(
    values: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Take a shared random pair without replacement, as in the reference SAC."""
    if values.shape[-1] < 2:
        raise ValueError(f"Expected at least two critics, got {values.shape[-1]}.")
    indices = torch.randperm(
        values.shape[-1], device=values.device, generator=generator
    )[:2]
    return values.index_select(-1, indices).amin(-1, keepdim=True)


def residual_critic_loss(
    model: Any,
    target_model: Any,
    batch: dict,
    alpha: torch.Tensor,
    gamma: float,
    backup_entropy: bool,
    generator: torch.Generator | None = None,
    *,
    bootstrap_time_limits: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use one discount per chunk with optional time-limit bootstrapping."""
    with torch.no_grad():
        action, log_prob, _ = model(forward_type=ForwardType.SAC, obs=batch["next_obs"])
        values = target_model(
            forward_type=ForwardType.SAC_Q, obs=batch["next_obs"], actions=action
        )
        target_q = random_min_q(values, generator)
        if backup_entropy:
            target_q = target_q - alpha.detach() * log_prob.sum(-1, keepdim=True)
        reward = (
            batch["rewards"].float().reshape(action.shape[0], -1).sum(-1, keepdim=True)
        )
        terminated = (
            batch["terminations"]
            .bool()
            .reshape(action.shape[0], -1)
            .any(-1, keepdim=True)
        )
        if not bootstrap_time_limits:
            terminated = terminated | batch["truncations"].bool().reshape(
                action.shape[0], -1
            ).any(-1, keepdim=True)
        target = reward + gamma * (~terminated) * target_q
    values = model(
        forward_type=ForwardType.SAC_Q, obs=batch["curr_obs"], actions=batch["actions"]
    )
    loss = F.mse_loss(values, target.expand_as(values))
    return loss, {
        "q_data": values.detach().mean().item(),
        "target_q": target.mean().item(),
        "td_error_abs": (values.detach() - target).abs().mean().item(),
    }


def residual_actor_loss(
    model: Any,
    obs: dict,
    alpha: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    actor_parameters: list[torch.nn.Parameter] | None = None,
    q_reduction: str = "random_min",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return the actor objective and optional pre-clipping component grad ratio.

    The ratio uses only actor parameters and weights the entropy gradient by
    detached alpha, with both terms evaluated on the same sampled actions.
    """

    if q_reduction not in {"random_min", "mean"}:
        raise ValueError(f"Expected random_min or mean Q reduction, got {q_reduction}.")

    def actor_terms():
        actions, log_prob, _ = model(forward_type=ForwardType.SAC, obs=obs)
        values = model(
            forward_type=ForwardType.SAC_Q,
            obs=obs,
            actions=actions,
            detach_encoder=True,
        )
        q = (
            values.mean(-1, keepdim=True)
            if q_reduction == "mean"
            else random_min_q(values, generator)
        )
        return q, log_prob.sum(-1, keepdim=True)

    metrics = {}
    if actor_parameters is not None:
        # FSDP requires a fresh forward/backward for each component gradient.
        device = actor_parameters[0].device
        devices = [] if device.type == "cpu" else [device.index]
        parameters = list(model.parameters())
        saved_grads = [p.grad for p in parameters]
        generator_state = generator.get_state() if generator is not None else None
        norms = []
        try:
            for component in ("q", "entropy"):
                model.zero_grad(set_to_none=True)
                if generator_state is not None:
                    generator.set_state(generator_state)
                with torch.random.fork_rng(devices=devices, device_type=device.type):
                    q, log_prob = actor_terms()
                    term = -q.mean() if component == "q" else log_prob.mean()
                    term.backward()
                    norms.append(
                        torch.stack(
                            [p.grad.detach().square().sum() for p in actor_parameters]
                        )
                        .sum()
                        .sqrt()
                    )
        finally:
            model.zero_grad(set_to_none=True)
            for parameter, grad in zip(parameters, saved_grads, strict=True):
                parameter.grad = grad
            if generator_state is not None:
                generator.set_state(generator_state)
        q_norm, entropy_norm = norms
        weighted_norm = alpha.detach() * entropy_norm
        if not torch.isfinite(torch.stack((q_norm, weighted_norm))).all():
            raise FloatingPointError("Expected finite actor component gradient norms.")
        ratio = q_norm / weighted_norm.clamp_min(torch.finfo(q_norm.dtype).tiny)
        metrics["actor_q_to_entropy_grad_ratio"] = ratio.item()
    q, log_prob = actor_terms()
    return (alpha.detach() * log_prob - q).mean(), log_prob, metrics


@dataclass
class ResidualUpdateBudget:
    """Count optimizer updates against newly collected action-chunk transitions."""

    utd: int = 20
    warmup_transitions: int = 1024
    warmup_updates: int = 2000
    critic_warmup_updates: int = 0
    max_updates_per_iteration: int = 400
    collected: int = 0
    warmup_collected: int | None = None
    updates: int = 0

    def add(self, transitions: int) -> None:
        self.collected += transitions

    def pending(self) -> int:
        """Return the full update debt for the collected decisions."""
        if self.collected < self.warmup_transitions:
            return 0
        if self.warmup_collected is None:
            self.warmup_collected = self.collected
        allowed = self.warmup_updates + self.utd * (
            self.collected - self.warmup_collected
        )
        return max(0, allowed - self.updates)

    def available(self) -> int:
        """Apply the per-call cap without discarding pending updates."""
        return min(self.max_updates_per_iteration, self.pending())

    def restore(self, state: dict) -> None:
        """Restore counters for the same schedule, retaining the current call cap."""
        for key in (
            "utd",
            "warmup_transitions",
            "warmup_updates",
            "critic_warmup_updates",
        ):
            saved = state.get(key, 0) if key == "critic_warmup_updates" else state[key]
            if saved != getattr(self, key):
                raise ValueError(
                    f"Expected checkpoint {key}={getattr(self, key)}, got {saved}."
                )
        self.collected = state["collected"]
        self.warmup_collected = state["warmup_collected"]
        self.updates = state["updates"]


def validate_residual_cfg(cfg: Any) -> None:
    """Validate the supported single-learner residual SAC execution contract."""
    import math

    from rlinf.models.embodiment.mlp_policy.sac_residual_policy import SACResidualConfig

    model_cfg = cfg.actor.model
    config_cls = SACResidualConfig
    if cfg.algorithm.loss_type == "expo":
        from rlinf.models.embodiment.mlp_policy.expo_policy import EXPOConfig

        config_cls = EXPOConfig
        if cfg.algorithm.backup_entropy:
            raise ValueError("EXPO requires backup_entropy=False.")
    settings = config_cls(
        **{
            key: model_cfg[key]
            for key in config_cls.__dataclass_fields__
            if key in model_cfg
        }
    )
    settings.validate()
    dinov3_cfg = cfg.rollout.get("residual_dinov3")
    if dinov3_cfg is not None:
        from rlinf.models.embodiment.mlp_policy.dinov3_encoder import DINOv3Config

        vision = DINOv3Config(**dict(dinov3_cfg))
        vision.validate()
        if settings.visual_dim != vision.output_dim:
            raise ValueError(
                f"Expected actor.model.visual_dim={vision.output_dim} for DINOv3, "
                f"got {settings.visual_dim}."
            )
    algorithm = cfg.algorithm
    limit = cfg.runner.get("max_critic_updates")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Expected positive integer runner.max_critic_updates.")
    save_interval = cfg.runner.get("save_critic_interval")
    if save_interval is not None and (
        type(save_interval) is not int or save_interval < 1 or limit is None
    ):
        raise ValueError(
            "Expected positive integer runner.save_critic_interval with max_critic_updates."
        )
    val_interval = cfg.runner.get("val_critic_interval")
    if val_interval is not None and (
        type(val_interval) is not int or val_interval < 1 or limit is None
    ):
        raise ValueError(
            "Expected positive integer runner.val_critic_interval with max_critic_updates."
        )
    temperature = algorithm.entropy_tuning
    settings.target_entropy(temperature.get("target_entropy"))
    if model_cfg.model_type != cfg.algorithm.loss_type or model_cfg.precision not in {
        "fp32",
        "32",
        32,
    }:
        raise ValueError(
            "Residual SAC requires model_type=sac_residual and FP32 heads."
        )
    fsdp = cfg.actor.fsdp_config
    if (
        fsdp.strategy != "fsdp"
        or fsdp.sharding_strategy != "no_shard"
        or not fsdp.use_orig_params
    ):
        raise ValueError(
            "Residual SAC requires FSDP no_shard with use_orig_params=True."
        )
    if (
        algorithm.get("target_update_freq", 1) != 1
        or algorithm.get("target_update_type", "all") != "all"
    ):
        raise ValueError(
            "Residual SAC updates both target projections and critics after every critic step."
        )
    if cfg.actor.get("compile_model", False):
        raise ValueError("Residual SAC currently requires eager actor training.")
    if fsdp.get("amp_autocast", {}).get("enabled", False) or fsdp.get(
        "grad_scaler", {}
    ).get("enabled", False):
        raise ValueError(
            "Residual SAC requires FP32 training without AMP or gradient scaling."
        )
    for dtype in fsdp.mixed_precision.values():
        if dtype not in {None, "fp32", "32", 32}:
            raise ValueError("Residual SAC requires FP32 FSDP precision.")
    if cfg.actor.micro_batch_size != cfg.actor.global_batch_size:
        raise ValueError("Residual SAC requires micro_batch_size=global_batch_size.")
    if temperature.alpha_type != "exp":
        raise ValueError("Residual SAC requires exponential entropy temperature.")
    if (
        algorithm.bootstrap_type != "standard"
        or type(algorithm.backup_entropy) is not bool
    ):
        raise ValueError(
            "Residual SAC requires standard terminal masking and boolean backup_entropy."
        )
    if type(algorithm.get("bootstrap_time_limits", True)) is not bool:
        raise ValueError("Expected boolean algorithm.bootstrap_time_limits.")
    for split in ("train", "eval"):
        env = cfg.env[split]
        if env.get("reward_mode") == "peg_staged_dense":
            from rlinf.envs.sim.maniskill.peg_dense_reward import PegDenseRewardConfig

            PegDenseRewardConfig(**dict(env.peg_dense_reward))
            if not algorithm.get("bootstrap_time_limits", True):
                raise ValueError(
                    "peg_staged_dense requires bootstrap_time_limits=True."
                )
            if env.use_rel_reward or env.ignore_terminations:
                raise ValueError(
                    "peg_staged_dense requires use_rel_reward=False and ignore_terminations=False."
                )
    if cfg.runner.get("use_training_pipeline", False) or cfg.runner.get(
        "enable_decoupled_mode", False
    ):
        raise ValueError(
            "Residual SAC requires chunk collection without training pipeline or decoupled mode."
        )
    if cfg.actor.get("enable_offload", False) or cfg.rollout.get(
        "enable_offload", False
    ):
        raise ValueError(
            "Residual SAC currently requires actor and rollout offload disabled."
        )
    if cfg.rollout.get("enable_torch_compile", False) or cfg.rollout.get(
        "enable_cuda_graph", False
    ):
        raise ValueError("Residual SAC currently requires eager rollout inference.")
    if not cfg.rollout.get("collect_final_values", True):
        raise ValueError(
            "Residual SAC requires collect_final_values=True for terminal features."
        )
    if not cfg.rollout.get("collect_transitions", False):
        raise ValueError("Residual SAC requires rollout.collect_transitions=True.")
    if algorithm.get("demo_buffer") or cfg.rollout.get("expert_model"):
        raise ValueError(
            "Residual SAC expects online replay without an intervention expert."
        )
    if algorithm.replay_buffer.get("enable_preload", False):
        raise ValueError(
            "Residual SAC requires replay preloading disabled for checkpoint resume."
        )
    schedule = algorithm.residual_schedule
    for key in (
        "utd",
        "warmup_transitions",
        "warmup_updates",
        "max_updates_per_iteration",
    ):
        if type(schedule[key]) is not int or schedule[key] < 1:
            raise ValueError(
                f"Expected positive integer residual_schedule.{key}, got {schedule[key]}."
            )
    critic_warmup = schedule.get("critic_warmup_updates", 0)
    if type(critic_warmup) is not int or critic_warmup < 0:
        raise ValueError(
            "Expected nonnegative integer residual_schedule.critic_warmup_updates."
        )
    if algorithm.replay_buffer.min_buffer_size > schedule.warmup_transitions:
        raise ValueError("Replay min_buffer_size must not exceed warmup_transitions.")
    if (
        type(algorithm.critic_actor_ratio) is not int
        or algorithm.critic_actor_ratio < 1
    ):
        raise ValueError("Expected positive integer critic_actor_ratio.")
    for name in ("gamma", "tau"):
        if not 0 < algorithm[name] <= 1:
            raise ValueError(f"Expected {name} in (0, 1], got {algorithm[name]}.")
    for value in (
        temperature.initial_alpha,
        temperature.optim.lr,
        temperature.optim.clip_grad,
        cfg.actor.optim.lr,
        cfg.actor.optim.clip_grad,
        cfg.actor.critic_optim.lr,
        cfg.actor.critic_optim.clip_grad,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "Expected finite positive residual SAC learning rates, alpha and gradient bounds."
            )
    base = cfg.rollout.residual_base_model
    if base.model_type != "openpi" or base.openpi.task != "eval":
        raise ValueError("Residual SAC requires a frozen OpenPI eval base model.")
    if (
        base.action_dim != settings.action_dim
        or base.num_action_chunks != settings.num_action_chunks
    ):
        raise ValueError(
            "Residual and base policy action dimensions and execution horizons must match."
        )
    representation = base.openpi.config_name
    contracts = {
        "pi05_rlt_maniskill_joint": (9, 8, "rlt_openpi_joint", "pd_joint_delta_pos"),
        "pi05_maniskill_eef_rot6d": (10, 10, "openpi_eef_rot6d", "pd_ee_delta_pose"),
    }
    if representation not in contracts:
        raise ValueError(
            f"Expected a supported residual ManiSkill representation, got {representation}."
        )
    state_dim, action_dim, wrap_mode, control_mode = contracts[representation]
    if (settings.state_dim, settings.action_dim, base.openpi.action_env_dim) != (
        state_dim,
        action_dim,
        action_dim,
    ):
        raise ValueError(
            f"Expected {representation} state_dim={state_dim}, action_dim=action_env_dim={action_dim}."
        )
    if (
        base.openpi.action_chunk != settings.num_action_chunks
        or base.openpi.action_horizon < settings.num_action_chunks
    ):
        raise ValueError(
            "Expected base action_chunk to match residual chunks within action_horizon."
        )
    if base.openpi.get("state_indices"):
        raise ValueError("Residual ManiSkill requires the complete configured state.")
    for split in ("train", "eval"):
        env_cfg = cfg.env[split]
        if bool(env_cfg.get("critic_privileged_obs", False)) != bool(
            settings.privileged_state_dim
        ):
            raise ValueError(
                f"Expected env.{split}.critic_privileged_obs to match actor.model.privileged_state_dim."
            )
        if (
            settings.privileged_state_dim
            and representation != "pi05_maniskill_eef_rot6d"
        ):
            raise ValueError("Privileged peg critic requires pi05_maniskill_eef_rot6d.")
        if (
            representation == "pi05_maniskill_eef_rot6d"
            and env_cfg.init_params.sim_backend not in {"gpu", "physx_cuda"}
        ):
            raise ValueError(
                f"Expected GPU simulation and GPU IK for env.{split} EEF rot6d."
            )
        if (
            env_cfg.wrap_obs_mode != wrap_mode
            or env_cfg.init_params.control_mode != control_mode
        ):
            raise ValueError(
                f"Expected env.{split} wrap_obs_mode={wrap_mode} and control_mode={control_mode} for {representation}."
            )
    if representation == "pi05_maniskill_eef_rot6d":
        from pathlib import Path

        from rlinf.models.embodiment.mlp_policy.sac_residual_policy import (
            load_eef_rot6d_stats,
        )

        stats_path = base.openpi_data.norm_stats_path
        if (
            not settings.eef_rot6d_norm_stats_path
            or Path(settings.eef_rot6d_norm_stats_path).resolve()
            != Path(stats_path).resolve()
        ):
            raise ValueError(
                "Residual and base EEF rot6d policies must use the same norm_stats.json."
            )
        load_eef_rot6d_stats(stats_path)
    elif settings.eef_rot6d_norm_stats_path:
        raise ValueError("Joint residual SAC must not use EEF rot6d normalization.")
    if base.openpi.get("use_rlt", False):
        raise ValueError("Residual SAC requires base.openpi.use_rlt=False.")
