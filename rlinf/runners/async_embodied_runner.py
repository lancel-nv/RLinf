# Copyright 2025 The RLinf Authors.
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

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.runners.async_weight_sync_mixin import AsyncWeightSyncMixin
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_utils import CumulativeSuccessRate, compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
        AsyncEmbodiedDAGGERFSDPPolicy,
    )
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )


class AsyncEmbodiedRunner(AsyncWeightSyncMixin, EmbodiedRunner):
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union["AsyncEmbodiedSACFSDPPolicy", "AsyncEmbodiedDAGGERFSDPPolicy"],
        rollout: "AsyncMultiStepRolloutWorker",
        env: "AsyncEnvWorker",
        reward: "EmbodiedRewardWorker",
        critic=None,
    ):
        super().__init__(cfg, actor, rollout, env, reward, critic)

        # Data channels
        self.env_metric_channel = Channel.create("EnvMetric")
        self.rollout_metric_channel = Channel.create("RolloutMetric")

        self.init_weight_sync_state()
        self.collection_success = CumulativeSuccessRate()

    def init_workers(self):
        super().init_workers()
        resume_dir = self.cfg.runner.get("resume_dir")
        if (
            self.cfg.algorithm.loss_type not in {"sac_residual", "expo"}
            or not resume_dir
        ):
            return
        path = Path(resume_dir) / "collection_metrics.json"
        if path.exists():
            self.collection_success = CumulativeSuccessRate(
                **json.loads(path.read_text())
            )
        else:
            self.logger.warning(
                "Checkpoint has no collection success counts; cumulative SR starts at resume."
            )

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if self.cfg.algorithm.loss_type not in {"sac_residual", "expo"}:
            return
        directory = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "checkpoints"
            / f"global_step_{self.global_step}"
        )
        (directory / "collection_metrics.json").write_text(
            json.dumps(asdict(self.collection_success), indent=2) + "\n"
        )

    def _collection_metrics(self, results: list[dict]) -> dict[str, float]:
        if self.cfg.algorithm.loss_type not in {"sac_residual", "expo"}:
            return {}
        for result in results:
            outcomes = result.get("env", {}).get("env/success_once")
            if outcomes is not None:
                self.collection_success.update(outcomes)
        rate = self.collection_success.rate
        return {} if rate is None else {"env/collected_success_rate": rate}

    def get_env_metrics(self) -> tuple[dict, list[dict], list[dict]]:
        results: list[dict] = []
        while True:
            try:
                result = self.env_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        collection_metrics = self._collection_metrics(results)
        if not results:
            return collection_metrics, [], []

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results, metric_field="time"
        )
        env_metrics, ranked_env_metrics_list = self._process_ranked_eval_results(
            results, metric_field="env"
        )
        env_metrics.update(collection_metrics)
        if not env_metrics:
            return {**time_metrics}, ranked_time_metrics_list, ranked_env_metrics_list

        return (
            {**env_metrics, **time_metrics},
            ranked_time_metrics_list,
            ranked_env_metrics_list,
        )

    def get_rollout_metrics(self) -> tuple[dict, list[dict]]:
        results: list[dict] = []
        while True:
            try:
                result = self.rollout_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}, []

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results, metric_field="time"
        )
        return time_metrics, ranked_time_metrics_list

    def evaluate(self):
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if env_decoupled_mode:
            # The rollout-side loop is unbounded by design. Ensure one service
            # exists and reuse it across every validation cycle.
            self.rollout.ensure_evaluate_service(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            ).wait()
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        if not env_decoupled_mode:
            rollout_handle: Handle = self.rollout.evaluate(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            )
        env_results = env_handle.wait()
        if not env_decoupled_mode:
            rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        critic_limit = self.cfg.runner.get("max_critic_updates")
        if critic_limit is not None and (
            self.cfg.algorithm.loss_type not in {"sac_residual", "expo"}
            or type(critic_limit) is not int
            or critic_limit < 1
        ):
            raise ValueError(
                "max_critic_updates requires residual SAC and a positive integer."
            )
        save_critic_interval = self.cfg.runner.get("save_critic_interval")
        if save_critic_interval is not None and (
            critic_limit is None
            or type(save_critic_interval) is not int
            or save_critic_interval < 1
        ):
            raise ValueError(
                "save_critic_interval requires a positive integer and max_critic_updates."
            )
        val_critic_interval = self.cfg.runner.get("val_critic_interval")
        start_critic_step = None
        start_time = time.time()
        self.env.set_global_step(self.global_step).wait()
        # Rollout must start from the actor's exact initial/resumed weights, so
        # this first sync always blocks even when overlap is enabled.
        self.update_rollout_weights()

        env_handle: Handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
            metric_channel=self.env_metric_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            actor_channel=self.actor_channel,
            metric_channel=self.rollout_metric_channel,
        )
        if self.reward is not None:
            reward_handle: Handle = self.reward.compute_rewards_async(
                input_channel=self.reward_channel,
                output_channel=self.env_channel,
            )
        actor_handle: Handle = self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel
        )

        while critic_limit is not None or self.global_step < self.max_steps:
            # Use the step we're ABOUT to run as the profiling key, mirroring
            # ``EmbodiedRunner.run`` which gates before ``self.global_step += 1``.
            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)
            skip_step = False
            reached_critic_limit = False
            with self.timer("step"):
                actor_training_handle: Handle = self.actor.run_training()
                actor_result = actor_training_handle.wait()
                if not actor_result[0]:
                    skip_step = True

                if not skip_step:
                    self.global_step += 1
                    if self.global_step % self.weight_sync_interval == 0:
                        self.update_rollout_weights(no_wait=self.sync_weight_no_wait)

                    training_metrics = {
                        f"train/{k}": v
                        for k, v in self._aggregate_numeric_metrics(
                            actor_result
                        ).items()
                    }

                    if critic_limit is not None:
                        critic_step = int(training_metrics["train/sac/critic_updates"])
                        if start_critic_step is None:
                            start_critic_step = critic_step - int(
                                training_metrics["train/sac/updates_this_call"]
                            )
                        reached_critic_limit = critic_step >= critic_limit
                    # A critic budget owns termination; runner steps still index checkpoints.
                    progress_limit = (
                        (
                            self.global_step
                            if reached_critic_limit
                            else self.global_step + 1
                        )
                        if critic_limit is not None
                        else self.max_steps
                    )
                    run_val, save_model, _ = check_progress(
                        self.global_step,
                        progress_limit,
                        self.cfg.runner.val_check_interval,
                        self.cfg.runner.save_interval,
                        1.0,
                        run_time_exceeded=False,
                    )
                    if save_critic_interval is not None:
                        save_model = (
                            critic_step % save_critic_interval == 0
                            or reached_critic_limit
                        )
                    if val_critic_interval is not None:
                        run_val = (
                            critic_step % val_critic_interval == 0
                            or reached_critic_limit
                        )
                    if save_model:
                        self._save_checkpoint()
                    eval_metrics = {}
                    if run_val:
                        with self.timer("eval"):
                            if val_critic_interval is not None:
                                self.update_rollout_weights(no_wait=False)
                            eval_metrics = self.evaluate()
                            eval_metrics = {
                                f"eval/{k}": v for k, v in eval_metrics.items()
                            }

            if skip_step:
                self.timer.consume_durations()
                if profiled_step is not None:
                    self._close_profiling_window(profiled_step)
                time.sleep(1.0)
                continue

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            if self.actor_channel is not None:
                # Total outstanding across consumers, matching the pre-dispatcher
                # meaning of this series.
                training_metrics["train/replay_channel_qsize"] = (
                    self.actor_channel.qsize()
                )
            actor_training_time_metrics, actor_time_metrics_per_rank = (
                actor_training_handle.consume_durations(return_per_rank=True)
            )
            actor_training_time_metrics = {
                f"time/actor/{k}": v for k, v in actor_training_time_metrics.items()
            }
            time_metrics.update(actor_training_time_metrics)
            env_metrics, env_time_metrics_per_rank, env_metrics_per_rank = (
                self.get_env_metrics()
            )
            rollout_metrics, rollout_time_metrics_per_rank = self.get_rollout_metrics()

            self.metric_logger.log(time_metrics, self.global_step)
            self.metric_logger.log(env_metrics, self.global_step)
            self.metric_logger.log(rollout_metrics, self.global_step)
            self.metric_logger.log(training_metrics, self.global_step)
            self.metric_logger.log(eval_metrics, self.global_step)
            self._log_ranked_metrics(
                metrics_list=actor_result,
                step=self.global_step,
                prefix="train",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/actor",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
                add_prefix=False,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=self.global_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
                add_prefix=False,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/rollout",
                worker_group_name=self.rollout.worker_group_name,
                add_prefix=False,
            )

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            self.print_metrics_table_async(
                critic_step - 1 if critic_limit is not None else self.global_step - 1,
                critic_limit if critic_limit is not None else self.max_steps,
                start_time,
                logging_metrics,
                start_critic_step if critic_limit is not None else start_step,
            )

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)
            if reached_critic_limit:
                break

        # Let any in-flight non-blocking sync land before the workers go away.
        self.drain_pending_rollout_weight_sync()

        self.env.stop().wait()
        self.rollout.stop().wait()
        self.actor.stop().wait()
        if self.reward is not None:
            self.reward.stop().wait()
            reward_handle.wait()
        env_handle.wait()
        rollout_handle.wait()
        actor_handle.wait()
        self._finish_run()
