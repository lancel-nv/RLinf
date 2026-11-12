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

"""EXPO objectives using cached next-state proposals and online action edits."""

from typing import Any

import torch
import torch.nn.functional as F

from rlinf.algorithms.sac_residual import residual_actor_loss
from rlinf.models.embodiment.base_policy import ForwardType


def expo_critic_loss(
    model: Any,
    batch: dict,
    gamma: float,
    generator: torch.Generator | None = None,
    *,
    bootstrap_time_limits: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select the next action, then draw an independent target-Q pair for backup."""
    with torch.no_grad():
        action, indices = model(obs=batch["next_obs"], generator=generator)
        subset = torch.randperm(
            model.config.num_q_heads, device=action.device, generator=generator
        )[:2]
        values = model(
            forward_type=ForwardType.SAC_Q,
            obs=batch["next_obs"],
            actions=action,
            target=True,
            q_indices=subset,
        )
        next_q = values.amin(-1, keepdim=True)
        reward = (
            batch["rewards"].float().reshape(action.shape[0], -1).sum(-1, keepdim=True)
        )
        terminal = (
            batch["terminations"]
            .bool()
            .reshape(action.shape[0], -1)
            .any(-1, keepdim=True)
        )
        if not bootstrap_time_limits:
            terminal = terminal | batch["truncations"].bool().reshape(
                action.shape[0], -1
            ).any(-1, keepdim=True)
        target = reward + gamma * (~terminal) * next_q
    values = model(
        forward_type=ForwardType.SAC_Q, obs=batch["curr_obs"], actions=batch["actions"]
    )
    loss = F.mse_loss(values, target.expand_as(values))
    return loss, {
        "q_data": values.detach().mean().item(),
        "target_q": target.mean().item(),
        "td_error_abs": (values.detach() - target).abs().mean().item(),
        "target_edit_fraction": (indices >= batch["next_obs"]["candidates"].shape[1])
        .float()
        .mean()
        .item(),
    }


def expo_actor_loss(
    model: Any,
    batch: dict,
    alpha: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    actor_parameters: list[torch.nn.Parameter] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Edit the executed replay action and maximize the online ensemble mean."""
    obs = {**batch["curr_obs"], "base_action": batch["actions"]}
    return residual_actor_loss(
        model,
        obs,
        alpha,
        generator,
        actor_parameters=actor_parameters,
        q_reduction="mean",
    )
