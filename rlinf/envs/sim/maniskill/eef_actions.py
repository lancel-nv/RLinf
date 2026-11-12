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

"""Decode EEF dataset actions at the ManiSkill controller boundary."""

import torch

from rlinf.utils.torch_rot6d import rot6d_to_matrix


def decode_eef_actions(actions: torch.Tensor) -> torch.Tensor:
    """Map raw xyz/rot6d/gripper labels to normalized GPU EEF controller commands.

    Translation and gripper are already controller coordinates. Intrinsic XYZ
    rotation uses ManiSkill's signed rotation scale, -0.1 radians.
    """
    if actions.shape[-1] != 10 or not torch.isfinite(actions).all():
        raise ValueError(
            f"Expected finite EEF actions ending in 10, got {actions.shape}."
        )
    matrix = rot6d_to_matrix(actions[..., 3:9])
    pitch = torch.asin(matrix[..., 0, 2].clamp(-1, 1))
    roll = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    yaw = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    singular = torch.cos(pitch).abs() < 1e-6
    roll = torch.where(
        singular, torch.atan2(matrix[..., 2, 1], matrix[..., 1, 1]), roll
    )
    yaw = torch.where(singular, torch.zeros_like(yaw), yaw)
    angles = torch.stack((roll, pitch, yaw), -1) / -0.1
    return torch.cat((actions[..., :3], angles, actions[..., 9:10]), -1)
