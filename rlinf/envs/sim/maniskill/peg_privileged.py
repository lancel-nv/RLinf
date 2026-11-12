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

"""Read privileged Panda peg-insertion state from ManiSkill."""

from typing import Any

import torch

from rlinf.data.schema.peg_privileged import (
    PEG_PRIVILEGED_DIM,
    canonicalize_peg_privileged_state,
)
from rlinf.utils.torch_rot6d import quaternion_wxyz_to_rot6d


def extract_peg_privileged_state(task: Any) -> torch.Tensor:
    """Return the 31-D task-relative state for both residual actor and critic.

    Pose translations and signed success margins are in millimetres. Rotations
    are canonical column-based rot6d. Pairwise contact forces are in hole axes,
    compressed as sign(f) * log1p(abs(f) / 10 N). Contacts and grasp flags are
    zero until the first physics step after each environment resets. No absolute
    poses are included.
    """
    agent = task.agent
    # GPU reset leaves contact impulses stale until the first physics step.
    contacts_valid = task.elapsed_steps > 0
    hole_inv = task.box_hole_pose.inv()
    head_hole = hole_inv * task.peg_head_pose
    tcp_peg = task.peg.pose.inv() * agent.tcp.pose
    hole_rotation = hole_inv.to_transformation_matrix()[..., :3, :3]

    def pose(value):
        return torch.cat(
            (value.p.float() * 1000, quaternion_wxyz_to_rot6d(value.q.float())), -1
        )

    xyz = head_hole.p
    radius = task.box_hole_radii[:, None]
    margins = torch.cat((xyz[:, :1] + 0.015, radius - xyz[:, 1:].abs()), -1) * 1000
    forces = []
    for a, b in (
        (agent.finger1_link, task.peg),
        (agent.finger2_link, task.peg),
        (task.peg, task.box),
    ):
        world_force = task.scene.get_pairwise_contact_forces(a, b).float()
        world_force = world_force.masked_fill(~contacts_valid[:, None], 0.0)
        force = (hole_rotation @ world_force.unsqueeze(-1)).squeeze(-1) / 10
        forces.append(force.sign() * force.abs().log1p())
    state = torch.cat(
        (
            pose(head_hole),
            margins,
            pose(tcp_peg),
            *forces,
            (agent.is_grasping(task.peg) & contacts_valid).float()[:, None],
        ),
        -1,
    ).float()
    if state.shape != (task.num_envs, PEG_PRIVILEGED_DIM):
        raise ValueError(
            f"Expected [{task.num_envs}, {PEG_PRIVILEGED_DIM}] privileged state, got {state.shape}."
        )
    return canonicalize_peg_privileged_state(state).detach().clone()
