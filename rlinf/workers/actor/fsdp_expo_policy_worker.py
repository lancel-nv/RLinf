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

"""EXPO objectives on the asynchronous residual SAC learner lifecycle."""

from rlinf.algorithms.expo import expo_actor_loss, expo_critic_loss
from rlinf.workers.actor.fsdp_sac_residual_policy_worker import (
    AsyncSACResidualFSDPPolicy,
)


class AsyncEXPOFSDPPolicy(AsyncSACResidualFSDPPolicy):
    """Reuse replay, budgets, optimizers and checkpoints with internal target heads."""

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        super().setup_model_and_optimizer(initialize_target=False)
        self.param_names_need_sync.extend(self.model.target_parameter_names())

    def soft_update_target_model(self, tau=None) -> None:
        self.model.update_targets(self.cfg.algorithm.tau if tau is None else tau)

    def forward_critic(self, batch):
        return expo_critic_loss(
            self.model,
            batch,
            self.cfg.algorithm.gamma,
            self.critic_sample_generator,
            bootstrap_time_limits=self.cfg.algorithm.get("bootstrap_time_limits", True),
        )

    def forward_actor(self, batch, actor_parameters):
        return expo_actor_loss(
            self.model,
            batch,
            self.entropy_temp.compute_alpha(),
            self.critic_sample_generator,
            actor_parameters=actor_parameters,
        )
