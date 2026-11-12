#!/usr/bin/env python3
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

"""Collect 10 mm peg demonstrations using GPU simulation and GPU EEF IK."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rlinf.data.storage.lerobot import add_frame_to_dataset
from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
    PEG_INSERTION_SIDE_HARD_OBSERVER_WIDE_WRIST_ENV_ID,
    extract_openpi_eef_states,
    validate_eef_gpu_controller,
)
from rlinf.utils.logging import get_logger
from toolkits.lerobot import collect_maniskill_peg_lerobot_joint as common
from toolkits.lerobot.tcp_trajectory import (
    TCPMotionConfig,
    bounded_tcp_delta,
    retime_tcp_path,
    tcp_pose_error,
)

LOG = get_logger()
ENV_ID = PEG_INSERTION_SIDE_HARD_OBSERVER_WIDE_WRIST_ENV_ID


@dataclass
class ReferenceSegment:
    """Joint reference for one movement or an explicit gripper hold."""

    phase: str
    hold_gripper: bool = False
    actions: list[np.ndarray] = field(default_factory=list)


def solve_gpu_reference(
    env: Any,
    seed: int,
    on_phase: Callable[[str, bool], None] | None = None,
    **_: Any,
) -> Any:
    """Plan on CPU using GPU scene poses, and execute joint references on GPU."""
    import sapien
    import trimesh
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb,
        get_actor_obb,
    )
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    env.reset(seed=seed)
    task = env.unwrapped
    if not task.scene.gpu_sim_enabled:
        raise ValueError("Reference execution requires GPU simulation.")
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=task.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    try:
        # Low-level SAPIEN actor poses are stale under GPU simulation.
        local_obb = get_actor_obb(task.peg, to_world_frame=False)
        obb = trimesh.primitives.Box(
            extents=local_obb.primitive.extents.copy(),
            transform=task.peg.pose.to_transformation_matrix()[0].cpu().numpy()
            @ local_obb.primitive.transform,
        )
        approaching = np.array([0, 0, -1])
        closing = task.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=closing, depth=0.025
        )
        initial_peg_pose = task.peg.pose
        half_length = task.peg_half_sizes[0, 0].item()
        grasp_pose = task.agent.build_grasp_pose(
            approaching, grasp["closing"], grasp["center"]
        )
        grasp_pose = grasp_pose * sapien.Pose(
            [-max(0.05, half_length / 2 + 0.01), 0, 0]
        )

        def phase(name: str, hold_gripper: bool = False) -> None:
            if on_phase is not None:
                on_phase(name, hold_gripper)

        def move(pose, name: str):
            phase(name)
            result = planner.move_to_pose_with_screw(pose)
            if isinstance(result, (int, np.integer)) and result == -1:
                raise ReferenceFailure("MPlib failed to plan a screw path.")
            return result

        move(grasp_pose * sapien.Pose([0, 0, -0.05]), "approach")
        move(grasp_pose, "grasp_approach")
        phase("grasp_hold", hold_gripper=True)
        planner.close_gripper()
        insert_pose = task.goal_pose * initial_peg_pose.inv() * grasp_pose
        offset = sapien.Pose([-0.01 - half_length, 0, 0])
        pre_insert_pose = insert_pose * offset
        move(pre_insert_pose, "transport")
        for index in range(3):
            pre_insert_pose = (
                task.goal_pose * offset * task.peg.pose.inv() * pre_insert_pose
            )
            move(pre_insert_pose, f"align_{index}")
        return move(insert_pose * sapien.Pose([0.05, 0, 0]), "insert")
    finally:
        planner.close()


class ReferenceFailure(RuntimeError):
    """The planner failed or exhausted the episode horizon."""


def initial_scene(env: Any) -> dict[str, np.ndarray]:
    """Snapshot geometry and reset state for reference/replay consistency."""
    task = env.unwrapped
    return {
        "qpos": common._to_numpy(task.agent.robot.qpos).copy(),
        "peg": common._to_numpy(task.peg.pose.raw_pose).copy(),
        "goal": common._to_numpy(task.goal_pose.raw_pose).copy(),
        "peg_half_sizes": common._to_numpy(task.peg_half_sizes).copy(),
        "box": common._to_numpy(task.box.pose.raw_pose).copy(),
        "hole_offset": common._to_numpy(task.box_hole_offsets.raw_pose).copy(),
        "hole_radius": common._to_numpy(task.box_hole_radii).copy(),
    }


def collect_reference(
    env: Any, seed: int
) -> tuple[dict[str, np.ndarray], list[ReferenceSegment]] | None:
    """Return joint targets only when their GPU reference execution succeeds."""
    segments: list[ReferenceSegment] = []
    initial = None
    reset, step = env.reset, env.step

    def reset_hook(*args, **kwargs):
        nonlocal initial
        result = reset(*args, **kwargs)
        initial = initial_scene(env)
        return result

    def on_phase(name: str, hold_gripper: bool) -> None:
        segments.append(ReferenceSegment(name, hold_gripper))

    def step_hook(action):
        if not segments:
            raise ReferenceFailure("Expected a named phase before reference execution.")
        segments[-1].actions.append(
            common._to_numpy(action).astype(np.float32).reshape(-1).copy()
        )
        result = step(action)
        if common._bool_scalar(result[3]):
            raise ReferenceFailure("Reference reached max_episode_steps.")
        return result

    env.reset, env.step = reset_hook, step_hook
    try:
        result = solve_gpu_reference(env, seed, on_phase=on_phase)
        if not common._solver_success(result):
            return None
        return initial, segments
    except ReferenceFailure as error:
        LOG.info("Reference rejected seed=%s: %s", seed, error)
        return None
    finally:
        env.reset, env.step = reset, step


def record(obs: dict[str, Any]) -> common.FrameRecord:
    """Own CPU copies of the pre-action observation and EEF state."""
    state = (
        extract_openpi_eef_states(
            obs, batch_size=1, device=obs["agent"]["qpos"].device
        )[0]
        .cpu()
        .numpy()
        .copy()
    )
    images = {
        name: {"rgb": camera["rgb"].detach().cpu().clone()}
        for name, camera in obs["sensor_data"].items()
        if "rgb" in camera
    }
    return common.FrameRecord(
        obs={"sensor_data": images},
        state=state,
        qpos=common._to_numpy(obs["agent"]["qpos"]).copy(),
    )


def replay(
    env: Any,
    seed: int,
    initial: dict[str, np.ndarray],
    segments: list[ReferenceSegment],
    motion: TCPMotionConfig,
) -> tuple[list[common.FrameRecord], list[np.ndarray], float, dict] | None:
    """Retime TCP paths and execute them through the GPU IK used by rollout."""
    from mani_skill.utils.geometry.rotation_conversions import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
    )

    from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
        ManiSkillEEFRot6dOutputs,
    )
    from rlinf.utils.rot6d import matrix_to_rot6d

    obs, _ = env.reset(seed=seed)
    validate_eef_gpu_controller(env)
    if env.unwrapped.control_freq != motion.control_freq:
        raise ValueError("TCP timing must match the environment control frequency.")
    for key, value in initial_scene(env).items():
        np.testing.assert_allclose(value, initial[key], atol=2e-6, err_msg=key)
    records, outputs = [record(obs)], []
    arm = env.unwrapped.agent.controller.controllers["arm"]
    chain = arm.kinematics.pk_chain
    max_error = 0.0
    decoder = ManiSkillEEFRot6dOutputs()
    speeds: dict[str, list[list[float]]] = {}

    def pose() -> np.ndarray:
        return arm.ee_pose_at_base.to_transformation_matrix()[0].cpu().numpy().copy()

    def execute(target: np.ndarray, gripper: float, phase: str) -> tuple[bool, bool]:
        nonlocal max_error
        current = pose()
        delta_pos, delta_matrix = bounded_tcp_delta(current, target, motion)
        delta_rot = matrix_to_euler_angles(
            torch.as_tensor(delta_matrix, device=arm.device, dtype=torch.float32)[None],
            "XYZ",
        )
        low, high = arm.action_space_low[:3], arm.action_space_high[:3]
        pos = torch.as_tensor(delta_pos, device=arm.device, dtype=torch.float32)[None]
        pos = 2 * (pos - low) / (high - low) - 1
        rot = delta_rot / arm.config.rot_lower
        if bool((pos.abs() > 1 + 1e-6).any() or (rot.norm(dim=-1) > 1 + 1e-6).any()):
            raise ValueError("TCP speed exceeds the EEF controller action limits.")
        output = torch.cat(
            (pos, rot, torch.tensor([[gripper]], device=arm.device)), dim=-1
        )[0]
        scaled = arm._clip_and_scale_action(output[None, :6])
        rotation = matrix_to_rot6d(
            euler_angles_to_matrix(scaled[:, 3:6], "XYZ").cpu().numpy()
        )[0]
        label = np.concatenate(
            (output[:3].cpu().numpy(), rotation, output[-1:].cpu().numpy())
        ).astype(np.float32)
        decoded = decoder({"actions": label[None]})["actions"][0]
        error = float(np.abs(decoded - output.cpu().numpy()).max())
        max_error = max(max_error, error)
        np.testing.assert_allclose(decoded, output.cpu().numpy(), atol=2e-6)
        obs, _, terminated, truncated, info = env.step(
            torch.from_numpy(decoded).to(arm.device)
        )
        outputs.append(label)
        records.append(record(obs))
        command_pose = np.eye(4)
        command_pose[:3, 3], command_pose[:3, :3] = delta_pos, delta_matrix
        commanded = tcp_pose_error(np.eye(4), command_pose)
        measured = tcp_pose_error(current, pose())
        speeds.setdefault(phase, []).append(
            [value * motion.control_freq for value in (*commanded, *measured)]
        )
        return common._bool_scalar(info.get("success", False)), (
            common._bool_scalar(terminated) or common._bool_scalar(truncated)
        )

    def result() -> tuple[list[common.FrameRecord], list[np.ndarray], float, dict]:
        names = (
            "command_linear_m_s",
            "command_angular_rad_s",
            "measured_linear_m_s",
            "measured_angular_rad_s",
        )
        report = {}
        for phase, values in speeds.items():
            array = np.asarray(values)
            report[phase] = {"steps": len(values)}
            for index, name in enumerate(names):
                report[phase][name] = {
                    "mean": float(array[:, index].mean()),
                    "p50": float(np.median(array[:, index])),
                    "max": float(array[:, index].max()),
                }
        return records, outputs, max_error, report

    for segment in segments:
        if not segment.actions:
            continue
        actions = np.stack(segment.actions)
        if not np.all(actions[:, -1] == actions[0, -1]):
            raise ValueError("Expected a constant gripper command within each phase.")
        if segment.hold_gripper:
            # Keep the explicit close duration; duplicate motion waypoints are not holds.
            targets = np.repeat(pose()[None], len(actions), axis=0)
        else:
            target_q = torch.as_tensor(actions[:, :7], device=arm.device)
            reference = chain.forward_kinematics(target_q).get_matrix().cpu().numpy()
            targets = retime_tcp_path(np.concatenate((pose()[None], reference)), motion)
        for target in targets:
            success, done = execute(target, float(actions[0, -1]), segment.phase)
            if success:
                return result()
            if done:
                return None
        if segment.hold_gripper or not len(targets):
            continue
        # Finish a phase before closing the gripper or starting the next path.
        while True:
            distance, angle = tcp_pose_error(pose(), targets[-1])
            if distance <= 0.001 and angle <= 0.01:
                break
            success, done = execute(
                targets[-1], float(actions[-1, -1]), segment.phase + "_settle"
            )
            if success:
                return result()
            if done:
                return None
    return None


def parse_args() -> argparse.Namespace:
    """Parse and validate GPU EEF collection settings."""
    parser = common.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        repo_id="local/maniskill_peginsertionside_10mm_eef_rot6d_gpu",
        num_episodes=400,
        sim_backend="gpu",
        max_episode_steps=500,
        target_control_mode="pd_ee_delta_pose",
        image_writer_processes=0,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--tcp-speed-m-s",
        type=float,
        default=0.04,
        help="TCP linear speed shared by all movement phases (m/s).",
    )
    parser.add_argument(
        "--tcp-angular-speed-rad-s",
        type=float,
        default=0.25,
        help="TCP angular speed limit shared by all phases (rad/s).",
    )
    args = parser.parse_args()
    try:
        TCPMotionConfig(
            args.tcp_speed_m_s, args.tcp_angular_speed_rad_s, args.control_freq
        )
    except ValueError as error:
        parser.error(str(error))
    if args.sim_backend not in {"gpu", "physx_cuda"}:
        parser.error("EEF demonstration execution requires --sim-backend gpu.")
    if (
        args.target_control_mode != "pd_ee_delta_pose"
        or args.solver_control_mode != "pd_joint_pos"
    ):
        parser.error("Expected pd_joint_pos reference and pd_ee_delta_pose execution.")
    if args.overwrite:
        parser.error("Use a new dataset path or --resume for EEF collection.")
    if (
        args.fps != args.control_freq
        or args.sim_freq % args.control_freq
        or args.sim_freq < args.control_freq * 8
    ):
        parser.error(
            "Expected fps=control_freq and sim_freq a multiple at least 8 times control_freq."
        )
    return args


def main() -> None:
    torch.set_num_threads(4)
    args = parse_args()
    motion = TCPMotionConfig(
        args.tcp_speed_m_s, args.tcp_angular_speed_rad_s, args.control_freq
    )
    output = common._resolve_output_path(args.repo_id)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Dataset exists: {output}; use a new path or --resume.")
    if (output / "source_manifest.json").exists():
        raise FileExistsError(f"Dataset already completed: {output}")
    solver = common._build_env(args, control_mode="pd_joint_pos", env_id=ENV_ID)
    env = None
    dataset = None
    entries, saved, attempts = [], 0, 0
    start = time.monotonic()
    try:
        env = common._build_env(args, control_mode="pd_ee_delta_pose", env_id=ENV_ID)
        contract = validate_eef_gpu_controller(env)
        for item in (solver, env):
            if (
                item.unwrapped.hole_clearance != 0.01
                or item.unwrapped.success_clearance != 0.01
            ):
                raise ValueError("Expected 10 mm geometry and success clearance.")
        if args.resume:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            entries = json.loads(output.with_suffix(".episodes.json").read_text())
            progress = json.loads(output.with_suffix(".progress.json").read_text())
            motion.validate_resume(entries)
            dataset = LeRobotDataset(repo_id=args.repo_id, root=output)
            if dataset.features["state"]["shape"] != (10,) or dataset.features[
                "actions"
            ]["shape"] != (10,):
                raise ValueError("Resume requires state/action dimensions 10.")
            if (
                dataset.meta.total_episodes != len(entries)
                or len(entries) != progress["saved"]
            ):
                raise ValueError("Dataset and progress episode counts disagree.")
            if args.fixed_peg_half_length is not None:
                expected = [
                    args.fixed_peg_half_length,
                    args.fixed_peg_half_width,
                    args.fixed_peg_half_width,
                ]
                for entry in entries:
                    np.testing.assert_allclose(
                        entry["initial_scene"]["peg_half_sizes"], expected, atol=1e-7
                    )
            if any(e["sim_backend"] != "gpu" or e["seed"] < args.seed for e in entries):
                raise ValueError(
                    "Resume requires GPU episodes from the same seed range."
                )
            dataset.start_image_writer(
                num_processes=args.image_writer_processes,
                num_threads=args.image_writer_threads,
            )
            saved, attempts = len(entries), progress["attempts"]
            start -= progress["elapsed_seconds"]
        while saved < args.num_episodes and attempts < args.max_attempts:
            seed = args.seed + attempts
            attempts += 1
            reference = collect_reference(solver, seed)
            if reference is None:
                continue
            result = replay(env, seed, *reference, motion=motion)
            if result is None:
                LOG.info("GPU EEF replay rejected seed=%s", seed)
                continue
            records, actions, error, speed_report = result
            main_camera = common._select_camera(
                records[0].obs, args.main_camera, common.MAIN_CAMERA_CANDIDATES, "main"
            )
            wrist_camera = common._select_camera(
                records[0].obs,
                args.wrist_camera,
                common.WRIST_CAMERA_CANDIDATES,
                "wrist",
            )
            frames = common._build_frames(
                records=records,
                actions=actions,
                task=args.task,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )
            if dataset is None:
                dataset = common._create_dataset(
                    repo_id=args.repo_id,
                    image_shape=frames[0]["image"].shape,
                    wrist_image_shape=frames[0]["wrist_image"].shape,
                    fps=args.fps,
                    image_writer_threads=args.image_writer_threads,
                    image_writer_processes=args.image_writer_processes,
                    state_dim=10,
                    action_dim=10,
                )
            for frame in frames:
                add_frame_to_dataset(dataset, frame)
            dataset.save_episode()
            if args.save_videos and saved < 3:
                common._write_episode_video(
                    frames,
                    video_dir=common._video_output_dir(args.repo_id, args.video_dir),
                    episode_index=saved,
                    seed=seed,
                    fps=args.fps,
                )
            entries.append(
                {
                    "episode_index": saved,
                    "seed": seed,
                    "frames": len(frames),
                    "replay_success": True,
                    "sim_backend": "gpu",
                    "ik_backend": contract["ik_backend"],
                    "controller_roundtrip_max_error": error,
                    "tcp_motion": motion.metadata(),
                    "tcp_speed_by_phase": speed_report,
                    "initial_scene": {
                        key: value.tolist() for key, value in reference[0].items()
                    },
                }
            )
            saved += 1
            progress = {
                "saved": saved,
                "target": args.num_episodes,
                "attempts": attempts,
                "frames": sum(e["frames"] for e in entries),
                "elapsed_seconds": time.monotonic() - start,
            }
            temporary = output.with_suffix(".progress.tmp")
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(output.with_suffix(".progress.json"))
            output.with_suffix(".episodes.json").write_text(
                json.dumps(entries, indent=2) + "\n"
            )
            LOG.info("PROGRESS %s", progress)
        if saved != args.num_episodes:
            raise RuntimeError(
                f"Expected {args.num_episodes} successful GPU episodes; collected {saved} in {attempts} attempts."
            )
        manifest = {
            **contract,
            "representation": "eef_rot6d_columns_v1",
            "fresh_collection": True,
            "source": "fresh GPU reference targets replayed through rot6d decoder and GPU EEF IK; success-filtered",
            "reference_planner": "cpu_mplib_with_gpu_reference_simulation",
            "tcp_motion": motion.metadata(),
            "env_id": ENV_ID,
            "clearance_m": 0.01,
            "fixed_peg_half_length": args.fixed_peg_half_length,
            "fixed_peg_half_width": args.fixed_peg_half_width,
            "reset_pose_randomization": True,
            "episodes": saved,
            "frames": sum(e["frames"] for e in entries),
            "fps": args.fps,
            "sim_freq": args.sim_freq,
            "control_freq": args.control_freq,
            "state_dim": 10,
            "action_dim": 10,
            "extra_delta_transform": False,
            "state_semantics": "world TCP xyz, first two rotation columns, total finger opening metres",
            "action_semantics": "normalized root delta xyz, root-aligned delta rotation columns, gripper command",
            "camera_names": [main_camera, wrist_camera],
            "image_shape": list(frames[0]["image"].shape),
            "successful_seed_count": len(entries),
            "max_controller_roundtrip_error": max(
                e["controller_roundtrip_max_error"] for e in entries
            ),
        }
        (output / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        LOG.info("COLLECTION_COMPLETE %s", manifest)
    finally:
        if dataset is not None and dataset.image_writer is not None:
            dataset.image_writer.wait_until_done()
        if env is not None:
            env.close()
        solver.close()


if __name__ == "__main__":
    main()
