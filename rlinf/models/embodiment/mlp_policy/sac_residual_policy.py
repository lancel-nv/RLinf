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

"""Residual action-chunk policy conditioned on frozen VLA features."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.torch_rot6d import canonicalize_rot6d


@dataclass
class SACResidualConfig:
    """Dimensions and residual bounds in normalized model-action coordinates."""

    state_dim: int = 9
    visual_dim: int = 2048
    state_proj_dim: int = 128
    visual_proj_dim: int = 256
    privileged_state_dim: int = 0
    privileged_proj_dim: int = 128
    actor_use_privileged_state: bool = False
    action_dim: int = 8
    num_action_chunks: int = 10
    hidden_size: int = 256
    depth: int = 3
    num_q_heads: int = 10
    edit_scale: float | Sequence[float] = 0.2
    initial_log_std: float = -1.5
    action_clipping: str = "none"
    eef_rot6d_norm_stats_path: str | None = None

    @property
    def chunk_dim(self) -> int:
        return self.action_dim * self.num_action_chunks

    @property
    def edit_scales(self) -> list[float]:
        if isinstance(self.edit_scale, (int, float)):
            scales = [float(self.edit_scale)] * self.action_dim
        else:
            scales = list(self.edit_scale)
        if len(scales) != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} per-action edit scales, got {len(scales)}."
            )
        return scales * self.num_action_chunks

    def target_entropy(self, value: float | None = None) -> float:
        """Resolve entropy over the executed chunk, including residual scaling."""
        scales = self.edit_scales
        target = sum(math.log(s) - 0.5 for s in scales) if value is None else value
        bound = sum(math.log(2 * s) for s in scales)
        if not math.isfinite(target) or target > bound:
            raise ValueError(
                f"Expected finite target entropy <= {bound}, got {target}."
            )
        return float(target)

    def validate(self) -> None:
        """Reject dimensions or bounds that cannot define a residual policy."""
        for name in (
            "state_dim",
            "visual_dim",
            "state_proj_dim",
            "visual_proj_dim",
            "action_dim",
            "privileged_proj_dim",
            "num_action_chunks",
            "hidden_size",
            "depth",
            "num_q_heads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Expected positive integer {name}, got {value}.")
        if self.eef_rot6d_norm_stats_path and (
            self.state_dim != 10 or self.action_dim != 10
        ):
            raise ValueError("EEF rot6d requires state_dim=action_dim=10.")
        from rlinf.data.schema.peg_privileged import PEG_PRIVILEGED_DIM

        if type(
            self.privileged_state_dim
        ) is not int or self.privileged_state_dim not in (0, PEG_PRIVILEGED_DIM):
            raise ValueError(
                f"Expected privileged_state_dim=0 or {PEG_PRIVILEGED_DIM}, got {self.privileged_state_dim}."
            )
        if type(self.actor_use_privileged_state) is not bool:
            raise ValueError("Expected boolean actor_use_privileged_state.")
        if self.actor_use_privileged_state and not self.privileged_state_dim:
            raise ValueError(
                "actor_use_privileged_state requires privileged_state_dim > 0."
            )
        if self.num_q_heads < 2:
            raise ValueError("Residual SAC requires at least two Q heads.")
        if any(not math.isfinite(s) or s <= 0 for s in self.edit_scales):
            raise ValueError("Expected finite positive residual edit scales.")
        if (
            not math.isfinite(self.initial_log_std)
            or not -5 <= self.initial_log_std <= 2
        ):
            raise ValueError("Expected initial_log_std in [-5, 2].")
        if self.action_clipping not in {"none", "final"}:
            raise ValueError("Expected action_clipping='none' or 'final'.")


def load_eef_rot6d_stats(path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load pi05 quantiles for the 10-D EEF state and delta-rotation actions."""
    stats = json.loads(Path(path).read_text())["norm_stats"]
    result = {}
    for key in ("state", "actions"):
        values = [
            torch.tensor(stats[key][name], dtype=torch.float32)
            for name in ("q01", "q99")
        ]
        if any(
            value.shape != (10,) or not torch.isfinite(value).all() for value in values
        ):
            raise ValueError(
                f"Expected finite 10-D EEF rot6d {key} quantiles in {path}."
            )
        low, high = values
        if (high < low).any():
            raise ValueError(f"Expected ordered EEF rot6d {key} quantiles in {path}.")
        result[key] = (low[3:9], high[3:9])
    return result


def _mlp(input_dim: int, output_dim: int, cfg: SACResidualConfig) -> nn.Sequential:
    layers = []
    for _ in range(cfg.depth):
        layers.extend(
            [
                nn.Linear(input_dim, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.ReLU(),
            ]
        )
        input_dim = cfg.hidden_size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class SACResidualPolicy(nn.Module, BasePolicy):
    """FP32 residual actor and Q ensemble; the rollout worker owns the frozen VLA.

    The critic trains shared projections. Actor updates detach their features;
    actor_use_privileged_state also exposes the privileged branch to the actor.
    """

    def __init__(self, config: SACResidualConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = nn.ModuleDict(
            {
                "state": nn.Sequential(
                    nn.Linear(config.state_dim, config.state_proj_dim),
                    nn.LayerNorm(config.state_proj_dim),
                ),
                "visual": nn.Sequential(
                    nn.Linear(config.visual_dim, config.visual_proj_dim),
                    nn.LayerNorm(config.visual_proj_dim),
                ),
            }
        )
        fused_dim = config.state_proj_dim + config.visual_proj_dim
        actor_dim = fused_dim
        if config.actor_use_privileged_state:
            actor_dim += config.privileged_proj_dim
        self.editor = _mlp(actor_dim + config.chunk_dim, 2 * config.chunk_dim, config)
        with torch.no_grad():
            self.editor[-1].weight.zero_()
            self.editor[-1].bias.zero_()
            self.editor[-1].bias[config.chunk_dim :].fill_(config.initial_log_std)
        critic_dim = fused_dim
        if config.privileged_state_dim:
            self.privileged_encoder = nn.Sequential(
                nn.Linear(config.privileged_state_dim, config.privileged_proj_dim),
                nn.LayerNorm(config.privileged_proj_dim),
            )
            critic_dim += config.privileged_proj_dim
        self.q_head = nn.ModuleList(
            _mlp(critic_dim + config.chunk_dim, 1, config)
            for _ in range(config.num_q_heads)
        )
        self.register_buffer("edit_scale", torch.tensor(config.edit_scales))
        if config.eef_rot6d_norm_stats_path:
            stats = load_eef_rot6d_stats(config.eef_rot6d_norm_stats_path)
            for key in ("state", "actions"):
                low, high = stats[key]
                self.register_buffer(key + "_rot_offset", (high + low + 1e-6) / 2)
                self.register_buffer(key + "_rot_scale", (high - low + 1e-6) / 2)

    def _canonical_input(self, value: torch.Tensor, key: str) -> torch.Tensor:
        if not self.config.eef_rot6d_norm_stats_path:
            return value
        rotation = value[..., 3:9] * getattr(self, key + "_rot_scale") + getattr(
            self, key + "_rot_offset"
        )
        rotation = canonicalize_rot6d(rotation)
        return torch.cat((value[..., :3], rotation, value[..., 9:]), dim=-1)

    def _feature(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                self.encoder["state"](
                    self._canonical_input(obs["state"].float(), "state")
                ),
                self.encoder["visual"](obs["visual"].float()),
            ],
            dim=-1,
        )

    def _privileged_feature(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        from rlinf.data.schema.peg_privileged import canonicalize_peg_privileged_state

        if "privileged_state" not in obs:
            raise ValueError(
                "Privileged policy requires privileged_state in observations."
            )
        privileged = canonicalize_peg_privileged_state(obs["privileged_state"].float())
        return self.privileged_encoder(privileged)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Residual SAC does not support {forward_type}.")

    def default_forward(self, **kwargs):
        return self.sac_forward(**kwargs)

    def sac_forward(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
        feature: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Sample a bounded edit; return composed actions and per-coordinate density."""
        if feature is None:
            feature = self._feature(obs)
            if self.config.actor_use_privileged_state:
                feature = torch.cat((feature, self._privileged_feature(obs)), -1)
        feature = feature.detach()
        base = obs["base_action"].float().reshape(feature.shape[0], -1)
        base_input = self._canonical_input(
            base.reshape(-1, self.config.num_action_chunks, self.config.action_dim),
            "actions",
        ).flatten(1)
        mean, log_std = self.editor(torch.cat((feature, base_input), dim=-1)).chunk(
            2, -1
        )
        log_std = log_std.clamp(-5, 2)
        noise = (
            torch.zeros_like(mean)
            if deterministic
            else torch.randn(
                mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
            )
        )
        raw = mean + log_std.exp() * noise
        edit = raw.tanh() * self.edit_scale
        log_prob = -0.5 * (noise.square() + math.log(2 * math.pi)) - log_std
        log_prob -= 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_prob -= self.edit_scale.log()
        action = base + edit
        if self.config.action_clipping == "final":
            action = action.clamp(-1, 1)
        return action, log_prob, None

    def sac_q_forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        detach_encoder: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Evaluate all critics on the composed normalized action chunk."""
        feature = self._feature(obs)
        if self.config.privileged_state_dim:
            feature = torch.cat((feature, self._privileged_feature(obs)), -1)
        if detach_encoder:
            feature = feature.detach()
        actions = self._canonical_input(
            actions.float().reshape(
                -1, self.config.num_action_chunks, self.config.action_dim
            ),
            "actions",
        )
        inputs = torch.cat((feature, actions.reshape(feature.shape[0], -1)), dim=-1)
        return torch.cat([head(inputs) for head in self.q_head], dim=-1)

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "train", **kwargs):
        """Return normalized actions; rollout decodes them through the frozen VLA."""
        device = self.edit_scale.device
        obs = {key: value.to(device) for key, value in env_obs.items()}
        action, log_prob, _ = self.sac_forward(obs, deterministic=mode == "eval")
        return action.reshape(
            -1, self.config.num_action_chunks, self.config.action_dim
        ), {
            "prev_logprobs": log_prob,
            "prev_values": None,
            "forward_inputs": {"action": action, "model_action": action},
        }
