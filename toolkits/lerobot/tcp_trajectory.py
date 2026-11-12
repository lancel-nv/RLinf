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

"""Cartesian timing for demonstration paths, independent of joint waypoint density."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TCPMotionConfig:
    """Shared translation and geodesic rotation limits for every motion phase."""

    linear_speed_m_s: float = 0.04
    angular_speed_rad_s: float = 0.25
    control_freq: int = 10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Expected finite positive {name}, got {value}.")

    def metadata(self) -> dict:
        """Return the timing contract stored with every collected episode."""
        return {"version": "cartesian_path_v1", **asdict(self)}

    def validate_resume(self, entries: list[dict]) -> None:
        """Reject mixing datasets with different or unrecorded motion timing."""
        if any(entry.get("tcp_motion") != self.metadata() for entry in entries):
            raise ValueError(
                "Resume requires the same recorded tcp_motion settings; "
                "use a new dataset path for old or differently timed demonstrations."
            )


def retime_tcp_path(poses: np.ndarray, config: TCPMotionConfig) -> np.ndarray:
    """Sample an SE(3) polyline at the control rate, excluding its starting pose.

    Each segment takes max(distance / linear speed, angle / angular speed).
    Sampling spans original waypoint boundaries without per-waypoint dwell.
    The exact endpoint is included; its final interval can be shorter than dt.
    Gripper holds are executed separately and must not pass through this function.
    """
    from scipy.spatial.transform import Rotation, Slerp

    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(f"Expected a nonempty (N, 4, 4) pose path, got {poses.shape}.")
    if not np.isfinite(poses).all():
        raise ValueError("Expected finite TCP poses.")
    matrices = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(matrices @ matrices.transpose(0, 2, 1), np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(matrices), 1, atol=1e-5)
    ):
        raise ValueError("Expected homogeneous TCP poses with proper rotations.")
    if len(poses) == 1:
        return poses[:0].copy()
    rotations = Rotation.from_matrix(matrices)
    distances = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    angles = (rotations[1:] * rotations[:-1].inv()).magnitude()
    durations = np.maximum(
        distances / config.linear_speed_m_s, angles / config.angular_speed_rad_s
    )
    keep = np.r_[True, durations > 1e-12]
    poses = poses[keep]
    if len(poses) == 1:
        return poses[:0].copy()
    times = np.r_[0.0, np.cumsum(durations[keep[1:]])]
    ticks = int(np.floor(times[-1] * config.control_freq + 1e-10))
    samples = np.arange(1, ticks + 1) / config.control_freq
    # Floating-point accumulation must not add a nearly stationary final tick.
    samples = np.r_[samples[samples < times[-1] - 1e-12], times[-1]]
    result = np.repeat(np.eye(4)[None], len(samples), axis=0)
    for axis in range(3):
        result[:, axis, 3] = np.interp(samples, times, poses[:, axis, 3])
    result[:, :3, :3] = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(
        samples
    ).as_matrix()
    result[-1] = poses[-1]
    return result


def bounded_tcp_delta(
    current: np.ndarray, target: np.ndarray, config: TCPMotionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return root-frame translation and rotation increments within one tick's limits."""
    from scipy.spatial.transform import Rotation

    delta = target[:3, 3] - current[:3, 3]
    rotvec = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    delta = delta / max(
        1.0, np.linalg.norm(delta) * config.control_freq / config.linear_speed_m_s
    )
    rotvec = rotvec / max(
        1.0, np.linalg.norm(rotvec) * config.control_freq / config.angular_speed_rad_s
    )
    return delta, Rotation.from_rotvec(rotvec).as_matrix()


def tcp_pose_error(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return translation distance in metres and geodesic angle in radians."""
    from scipy.spatial.transform import Rotation

    return (
        float(np.linalg.norm(target[:3, 3] - current[:3, 3])),
        float(Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).magnitude()),
    )
