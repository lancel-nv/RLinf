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

"""EXPO selection over frozen proposals and bounded edits."""

from copy import deepcopy
from dataclasses import dataclass

import torch

from rlinf.models.embodiment.mlp_policy.sac_residual_policy import (
    SACResidualConfig,
    SACResidualPolicy,
)


@dataclass
class EXPOConfig(SACResidualConfig):
    """Candidate geometry; selection always uses the target projection and critics."""

    num_candidates: int = 8
    num_edits: int = 8

    def validate(self) -> None:
        super().validate()
        if type(self.num_candidates) is not int or self.num_candidates < 1:
            raise ValueError("Expected positive integer num_candidates.")
        if (
            type(self.num_edits) is not int
            or not 0 <= self.num_edits <= self.num_candidates
        ):
            raise ValueError("Expected 0 <= num_edits <= num_candidates.")
        if self.privileged_state_dim or self.actor_use_privileged_state:
            raise ValueError(
                "EXPO expects visual and proprioceptive inputs without privileged state."
            )


class EXPOPolicy(SACResidualPolicy):
    """Own online heads and frozen target heads in one synchronized checkpoint."""

    def __init__(self, config: EXPOConfig):
        super().__init__(config)
        self.target_encoder = deepcopy(self.encoder).requires_grad_(False)
        self.target_q_head = deepcopy(self.q_head).requires_grad_(False)

    def target_feature(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Project cached observations with the Polyak target encoder."""
        return torch.cat(
            (
                self.target_encoder["state"](
                    self._canonical_input(obs["state"].float(), "state")
                ),
                self.target_encoder["visual"](obs["visual"].float()),
            ),
            dim=-1,
        )

    def _target_q(
        self,
        feature: torch.Tensor,
        actions: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_input = self._canonical_input(
            actions.float().reshape(
                -1, self.config.num_action_chunks, self.config.action_dim
            ),
            "actions",
        ).flatten(1)
        inputs = torch.cat((feature, action_input), -1)
        indices = range(len(self.target_q_head)) if indices is None else indices
        return torch.cat([self.target_q_head[int(i)](inputs) for i in indices], -1)

    def sac_q_forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        detach_encoder: bool = False,
        target: bool = False,
        q_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Evaluate online Q heads, or the requested subset of frozen target heads."""
        if target:
            return self._target_q(self.target_feature(obs), actions, q_indices)
        return super().sac_q_forward(
            obs, actions, detach_encoder=detach_encoder, **kwargs
        )

    @torch.no_grad()
    def select(
        self, obs: dict[str, torch.Tensor], *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose the highest random-pair target minimum among proposals and edits.

        Both training and evaluation sample edits and critic subsets. Reproducible
        evaluation requires fixed proposal noise and selection RNG seeds.
        """
        candidates = obs["candidates"].float()
        expected = (self.config.num_candidates, self.config.chunk_dim)
        if candidates.ndim != 3 or candidates.shape[1:] != expected:
            raise ValueError(
                f"Expected candidate shape [B, {expected[0]}, {expected[1]}], got {tuple(candidates.shape)}."
            )
        batch, count, width = candidates.shape
        feature = self.target_feature(obs)
        edits = self.config.num_edits
        if edits:
            edit_feature = (
                feature[:, None].expand(-1, edits, -1).reshape(batch * edits, -1)
            )
            edited, _, _ = self.sac_forward(
                {"base_action": candidates[:, :edits].reshape(batch * edits, width)},
                feature=edit_feature,
                generator=generator,
            )
            candidates = torch.cat((candidates, edited.reshape(batch, edits, width)), 1)
        feature = (
            feature[:, None]
            .expand(-1, count + edits, -1)
            .reshape(batch * (count + edits), -1)
        )
        subset = torch.randperm(
            self.config.num_q_heads, device=feature.device, generator=generator
        )[:2]
        scores = (
            self._target_q(feature, candidates.reshape(-1, width), subset)
            .amin(-1)
            .reshape(batch, -1)
        )
        indices = scores.argmax(-1)
        return candidates[
            torch.arange(batch, device=candidates.device), indices
        ], indices

    def default_forward(self, obs, generator=None, **kwargs):
        return self.select(obs, generator=generator)

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        """Update only target projections and critics; the editor stays online."""
        if not 0 < tau <= 1:
            raise ValueError(f"Expected tau in (0, 1], got {tau}.")
        for online, target in (
            (self.encoder, self.target_encoder),
            (self.q_head, self.target_q_head),
        ):
            for p, q in zip(online.parameters(), target.parameters(), strict=True):
                q.lerp_(p, tau)

    def target_parameter_names(self) -> list[str]:
        """Return frozen target parameters that must also reach rollout workers."""
        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith(("target_encoder.", "target_q_head."))
        ]

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "train", **kwargs):
        obs = {key: value.to(self.edit_scale.device) for key, value in env_obs.items()}
        action, indices = self.select(obs, generator=kwargs.get("generator"))
        return action.reshape(
            -1, self.config.num_action_chunks, self.config.action_dim
        ), {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": action,
                "model_action": action,
                "expo_selected_index": indices,
            },
        }
