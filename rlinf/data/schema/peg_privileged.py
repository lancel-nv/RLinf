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

"""Shared tensor layout for privileged Panda peg-insertion observations."""

import torch

from rlinf.utils.torch_rot6d import canonicalize_rot6d

PEG_PRIVILEGED_FIELDS = (
    ("peg_head_hole", 9),
    ("margins", 3),
    ("tcp_peg", 9),
    ("finger_peg_and_peg_box_forces", 9),
    ("is_grasping", 1),
)
PEG_PRIVILEGED_DIM = sum(width for _, width in PEG_PRIVILEGED_FIELDS)
PEG_PRIVILEGED_ROT6D_STARTS = (3, 15)


def canonicalize_peg_privileged_state(state: torch.Tensor) -> torch.Tensor:
    """Canonicalize all physical rot6d blocks before the critic projection."""
    if state.shape[-1] != PEG_PRIVILEGED_DIM:
        raise ValueError(
            f"Expected {PEG_PRIVILEGED_DIM}-D peg privileged state, got {state.shape}."
        )
    indices = torch.tensor(PEG_PRIVILEGED_ROT6D_STARTS, device=state.device)[:, None]
    indices = (indices + torch.arange(6, device=state.device)).flatten()
    rotations = state.index_select(-1, indices).reshape(*state.shape[:-1], -1, 6)
    result = state.clone()
    result[..., indices] = canonicalize_rot6d(rotations).flatten(-2)
    return result
