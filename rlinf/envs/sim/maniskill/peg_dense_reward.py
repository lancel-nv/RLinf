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

"""Three bounded distance costs for reaching, aligning and inserting a peg."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PegDenseRewardConfig:
    """Distances and scales are metres; grasp bounds use peg-local axes."""

    grasp_min_peg_m: tuple[float, float, float]
    grasp_max_peg_m: tuple[float, float, float]
    reach_scale_m: float = 0.05
    alignment_scale_m: float = 0.05
    depth_scale_m: float = 0.10
    alignment_tolerance_m: float = 0.002
    axis_tolerance_deg: float = 5.0

    def __post_init__(self) -> None:
        for name in ("grasp_min_peg_m", "grasp_max_peg_m"):
            values = getattr(self, name)
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Expected finite 3-D {name}, got {values}.")
        if any(lo >= hi for lo, hi in zip(self.grasp_min_peg_m, self.grasp_max_peg_m)):
            raise ValueError(
                "Expected grasp_min_peg_m < grasp_max_peg_m on every axis."
            )
        for name in (
            "reach_scale_m",
            "alignment_scale_m",
            "depth_scale_m",
            "alignment_tolerance_m",
            "axis_tolerance_deg",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Expected finite positive {name}, got {value}.")
        if self.axis_tolerance_deg >= 90:
            raise ValueError("Expected axis_tolerance_deg < 90 degrees.")


def compute_peg_dense_reward(
    tcp_peg: torch.Tensor,
    head_hole: torch.Tensor,
    peg_axis_hole: torch.Tensor,
    box_half_depth: torch.Tensor,
    is_grasped: torch.Tensor,
    success: torch.Tensor,
    config: PegDenseRewardConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return negative distance costs in [-3,0], with success overridden to zero.

    Locked alignment and depth costs equal one. Unlocked costs use d/(d+scale),
    so unlocking alone cannot lower reward. Gates use the current state.
    Hole +X points into the box; the task's success plane is X=-0.015 m.
    """
    lower = tcp_peg.new_tensor(config.grasp_min_peg_m)
    upper = tcp_peg.new_tensor(config.grasp_max_peg_m)
    outside = (lower - tcp_peg).clamp_min(0) + (tcp_peg - upper).clamp_min(0)
    reach_distance = torch.linalg.vector_norm(outside, dim=-1)
    lateral = torch.linalg.vector_norm(head_hole[:, 1:], dim=-1)
    angle = torch.acos(peg_axis_hole[:, 0].clamp(-1, 1))
    # Express axis error as a tip displacement over the peg's half-length.
    axis_displacement = box_half_depth[:, None] * (
        peg_axis_hole - peg_axis_hole.new_tensor([1.0, 0.0, 0.0])
    )
    alignment_distance = torch.linalg.vector_norm(
        torch.cat((head_hole[:, 1:], axis_displacement), dim=-1), dim=-1
    )
    depth_remaining = (-0.015 - head_hole[:, 0]).clamp_min(0)
    grasped = is_grasped.bool()
    aligned = (
        grasped
        & (lateral <= config.alignment_tolerance_m)
        & (angle <= math.radians(config.axis_tolerance_deg))
    )
    reach_cost = reach_distance / (reach_distance + config.reach_scale_m)
    alignment_cost = torch.where(
        grasped,
        alignment_distance / (alignment_distance + config.alignment_scale_m),
        torch.ones_like(reach_cost),
    )
    depth_cost = torch.where(
        aligned,
        depth_remaining / (depth_remaining + config.depth_scale_m),
        torch.ones_like(reach_cost),
    )
    success = success.bool()
    reach_cost = torch.where(success, 0.0, reach_cost)
    alignment_cost = torch.where(success, 0.0, alignment_cost)
    depth_cost = torch.where(success, 0.0, depth_cost)
    reward = -(reach_cost + alignment_cost + depth_cost)
    stage = torch.where(success, 3, grasped.long() + aligned.long())
    return reward, {
        "dense_reward": reward,
        "dense_stage": stage,
        "dense_grasp_cost": reach_cost,
        "dense_alignment_cost": alignment_cost,
        "dense_depth_cost": depth_cost,
        "dense_grasped": grasped,
        "dense_aligned": aligned,
        "grasp_region_error_mm": reach_distance * 1000,
        "alignment_error_mm": lateral * 1000,
        "alignment_distance_mm": alignment_distance * 1000,
        "alignment_axis_error_deg": torch.rad2deg(angle),
        "depth_remaining_mm": depth_remaining * 1000,
        "insertion_depth_mm": (head_hole[:, 0] + box_half_depth).clamp_min(0) * 1000,
    }
