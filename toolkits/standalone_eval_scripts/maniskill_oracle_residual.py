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

"""Physical residual oracle for the GPU EEF peg-insertion evaluation contract."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OracleConfig:
    """Root-frame limits are metres and rotation-vector radians per control step."""

    grasp_center: tuple[float, float, float]
    position_limit: tuple[float, float, float]
    rotation_limit: tuple[float, float, float]
    kp: float = 1.0
    kr: float = 1.0
    reach_radius: float = 0.05


def oracle_correction(
    tcp_peg: np.ndarray,
    head_hole: np.ndarray,
    peg_rotation: np.ndarray,
    hole_rotation: np.ndarray,
    root_rotation: np.ndarray,
    grasped: np.ndarray,
    config: OracleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bounded position/rotation corrections and stage 0/off, 1/reach, 2/align.

    Input rotations map local vectors to world axes. There is no insertion
    drive or gripper override; pi05 remains responsible for those actions.
    """
    error = np.asarray(config.grasp_center) - tcp_peg
    reach = np.linalg.norm(error, axis=-1) < config.reach_radius
    lateral = -head_hole.copy()
    lateral[:, 0] = 0
    position_world = config.kp * np.where(
        grasped[:, None],
        np.einsum("bij,bj->bi", hole_rotation, lateral),
        np.einsum("bij,bj->bi", peg_rotation, error),
    )
    rotation_world = config.kr * np.cross(peg_rotation[:, :, 0], hole_rotation[:, :, 0])
    root_inverse = root_rotation.swapaxes(-1, -2)
    position = np.einsum("bij,bj->bi", root_inverse, position_world)
    rotation = np.einsum("bij,bj->bi", root_inverse, rotation_world)
    stage = np.where(grasped, 2, np.where(reach, 1, 0))
    position[stage == 0] = 0
    rotation[~grasped] = 0
    return (
        np.clip(position, -np.asarray(config.position_limit), config.position_limit),
        np.clip(rotation, -np.asarray(config.rotation_limit), config.rotation_limit),
        stage,
    )


def controller_physical(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode the installed +/-0.1 GPU controller, including rotation norm clipping."""
    from scipy.spatial.transform import Rotation

    angles = action[:, 3:6].astype(np.float64)
    angles = (
        -0.1 * angles / np.maximum(np.linalg.norm(angles, axis=-1, keepdims=True), 1)
    )
    return np.clip(action[:, :3], -1, 1) * 0.1, Rotation.from_euler(
        "XYZ", angles
    ).as_matrix()


def apply_physical_correction(
    base_action: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    rotation_limit: tuple[float, float, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode physical corrections without editing normalized rot6d coordinates.

    Translation is added before controller clipping, preserving its exact
    normalized-edit budget even if the proposal is saturated. Rotation is
    composed on SO(3) in root axes, then encoded through intrinsic XYZ / -0.1.
    A line search enforces component limits after controller norm clipping.
    Zero correction returns the original command bit for bit.
    """
    from scipy.spatial.transform import Rotation

    action = base_action.copy()
    action[:, :3] += position / 0.1
    base_position, base_rotation = controller_physical(base_action)
    changed = np.any(rotation != 0, axis=-1)
    limit = np.asarray(rotation_limit)
    scale = np.ones(len(action))
    for _ in range(24):
        target = (
            Rotation.from_rotvec(rotation * scale[:, None]).as_matrix() @ base_rotation
        )
        angles = Rotation.from_matrix(target).as_euler("XYZ")
        action[changed, 3:6] = (angles / -0.1)[changed]
        _, actual_rotation = controller_physical(action)
        effective = Rotation.from_matrix(
            actual_rotation @ base_rotation.swapaxes(-1, -2)
        ).as_rotvec()
        invalid = changed & np.any(np.abs(effective) > limit + 2e-9, axis=-1)
        if not invalid.any():
            break
        scale[invalid] *= 0.5
    if invalid.any():
        action[invalid, 3:6] = base_action[invalid, 3:6]
    # Preserve inactive and zero-oracle rows without a decode/re-encode round trip.
    unchanged = ~np.any(position != 0, axis=-1) & ~changed
    action[unchanged] = base_action[unchanged]
    actual_position, actual_rotation = controller_physical(action)
    effective = Rotation.from_matrix(
        actual_rotation @ base_rotation.swapaxes(-1, -2)
    ).as_rotvec()
    return action, {
        "effective_position": actual_position - base_position,
        "effective_rotation": effective,
        "rotation_line_scale": scale,
        "actual_rotation": actual_rotation,
    }


def rot6d_box_feasible(
    raw_rotation: np.ndarray, half_width: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Certify target rotations have a Gram–Schmidt preimage inside a raw 6-D box.

    Preimages are [s*r1, t*r1+u*r2], s,u>0. Check the first column's interval
    and all vertices of the second column's 2-D feasible polygon. This is a
    sufficient certificate for post-controller rotations; saturated Euler
    inputs can admit additional preimages not covered by this certificate.
    """
    lower, upper = raw_rotation - half_width, raw_rotation + half_width
    first, second = target[:, :, 0], target[:, :, 1]
    nonzero = np.abs(first) > 1e-12
    denominator = np.where(nonzero, first, 1)
    a, b = lower[:, :3] / denominator, upper[:, :3] / denominator
    low = np.maximum(
        np.max(np.where(nonzero, np.minimum(a, b), -np.inf), axis=-1), 1e-6
    )
    high = np.min(np.where(nonzero, np.maximum(a, b), np.inf), axis=-1)
    feasible_first = (low <= high + 1e-10) & np.all(
        nonzero | ((lower[:, :3] <= 1e-10) & (upper[:, :3] >= -1e-10)), axis=-1
    )
    matrix = np.stack((first, second), axis=-1)
    matrix = np.concatenate(
        (matrix, -matrix, np.broadcast_to([[[0.0, -1.0]]], (len(first), 1, 2))), axis=1
    )
    bounds = np.concatenate(
        (upper[:, 3:], -lower[:, 3:], np.full((len(first), 1), -1e-6)), axis=1
    )
    feasible_second = np.zeros(len(first), dtype=bool)
    for i in range(7):
        for j in range(i):
            x, y = matrix[:, i], matrix[:, j]
            det = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
            valid = np.abs(det) > 1e-12
            safe_det = np.where(valid, det, 1)
            t = (bounds[:, i] * y[:, 1] - bounds[:, j] * x[:, 1]) / safe_det
            u = (x[:, 0] * bounds[:, j] - y[:, 0] * bounds[:, i]) / safe_det
            point = np.stack((t, u), axis=-1)
            feasible_second |= valid & np.all(
                np.einsum("bki,bi->bk", matrix, point) <= bounds + 1e-10, axis=-1
            )
    return feasible_first & feasible_second
