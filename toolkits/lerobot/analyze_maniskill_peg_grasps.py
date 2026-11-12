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

"""Replay successful GPU EEF demonstrations to measure the peg-local grasp point."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rlinf.utils.logging import get_logger

LOG = get_logger()


def summarize_grasps(episodes: list[dict], range_margin_m: float = 0.005) -> dict:
    """Estimate a grasp region from equally weighted successful episode medians."""
    if not np.isfinite(range_margin_m) or range_margin_m < 0:
        raise ValueError("Expected finite nonnegative range_margin_m.")
    accepted = [e for e in episodes if e["accepted"]]
    if not accepted:
        raise ValueError("Expected at least one successful state-matched grasp replay.")
    medians = np.array([e["grasp_median_peg_m"] for e in accepted])
    bounds = np.quantile(medians, [0.1, 0.9], axis=0)
    lower, upper = bounds[0] - range_margin_m, bounds[1] + range_margin_m
    return {
        "grasp_min_peg_m": lower.tolist(),
        "grasp_max_peg_m": upper.tolist(),
        "range_margin_m": range_margin_m,
        "fraction_episode_medians_in_region": float(
            ((medians >= lower) & (medians <= upper)).all(axis=1).mean()
        ),
        "episodes": episodes,
        "accepted_episodes": len(accepted),
        "total_episodes": len(episodes),
        "grasp_point_peg_m": np.median(medians, axis=0).tolist(),
        "episode_median_q10_q90_m": np.quantile(medians, [0.1, 0.9], axis=0).tolist(),
        "frame": "peg local, origin at peg center, +X toward insertion tip",
        "estimator": "coordinate median of per-episode medians; frames with >=3 consecutive grasp detections",
    }


def main() -> None:
    """Replay dataset actions with GPU physics and GPU IK; save an auditable median."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--max-state-error", type=float, default=0.002)
    parser.add_argument("--range-margin-m", type=float, default=0.005)
    args = parser.parse_args()
    import gymnasium as gym
    import pyarrow.parquet as pq
    import torch

    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        PANDA_WIDE_WRISTCAM_UID,
        extract_openpi_eef_states,
        register_rlinf_peg_insertion_side_variants,
        validate_eef_gpu_controller,
    )
    from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
        ManiSkillEEFRot6dOutputs,
    )

    torch.set_num_threads(2)
    manifest = json.loads((args.dataset / "source_manifest.json").read_text())
    entries_path = args.dataset.with_suffix(".episodes.json")
    entries = json.loads(entries_path.read_text())[args.start : args.stop]
    if not all(e["replay_success"] for e in entries):
        raise ValueError("Expected success-filtered demonstration metadata.")
    register_rlinf_peg_insertion_side_variants()
    env = gym.make(
        manifest["env_id"],
        num_envs=1,
        obs_mode="state_dict",
        robot_uids=PANDA_WIDE_WRISTCAM_UID,
        control_mode="pd_ee_delta_pose",
        sim_backend="gpu",
        reward_mode="sparse",
        sim_config={
            "sim_freq": manifest["sim_freq"],
            "control_freq": manifest["control_freq"],
        },
        max_episode_steps=500,
        **{
            key: manifest[key]
            for key in ("fixed_peg_half_length", "fixed_peg_half_width")
            if manifest.get(key) is not None
        },
    )
    task = env.unwrapped
    contract = validate_eef_gpu_controller(env)
    decoder = ManiSkillEEFRot6dOutputs()
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for entry in entries:
            index = entry["episode_index"]
            paths = list((args.dataset / "data").glob(f"*/episode_{index:06d}.parquet"))
            if len(paths) != 1:
                raise ValueError(
                    f"Expected one parquet for episode {index}, got {paths}."
                )
            rows = pq.read_table(paths[0], columns=["state", "actions"]).to_pylist()
            states = np.array([r["state"] for r in rows])
            actions = decoder(
                {"actions": np.array([r["actions"] for r in rows], dtype=np.float32)}
            )["actions"]
            obs, _ = env.reset(seed=entry["seed"], options={"reconfigure": True})
            points, consecutive, max_error = [], 0, 0.0
            trace = []
            for frame, action in enumerate(actions):
                actual = (
                    extract_openpi_eef_states(obs, batch_size=1, device=task.device)
                    .cpu()
                    .numpy()[0]
                )
                max_error = max(max_error, float(np.abs(actual - states[frame]).max()))
                obs, _, terminated, truncated, info = env.step(
                    torch.as_tensor(action, device=task.device)
                )
                grasped = bool(task.agent.is_grasping(task.peg)[0])
                consecutive = consecutive + 1 if grasped else 0
                point = (task.peg.pose.inv() * task.agent.tcp.pose).p[0].cpu().numpy()
                head = (
                    (task.box_hole_pose.inv() * task.peg_head_pose).p[0].cpu().numpy()
                )
                trace.append(
                    {
                        "frame": frame + 1,
                        "grasped": grasped,
                        "tcp_peg_m": point.tolist(),
                        "head_hole_m": head.tolist(),
                    }
                )
                if consecutive >= 3:
                    points.append(point.copy())
                if bool(terminated[0]) or bool(truncated[0]):
                    break
            success = bool(info["success"][0])
            result = {
                "episode_index": index,
                "seed": entry["seed"],
                "dataset_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                "replay_success": success,
                "max_state_abs_error": max_error,
                "grasp_frames": len(points),
                "peg_half_sizes_m": task.peg_half_sizes[0].cpu().tolist(),
                "accepted": success
                and max_error <= args.max_state_error
                and bool(points),
                "grasp_median_peg_m": np.median(points, axis=0).tolist()
                if points
                else None,
            }
            results.append(result)
            trace_path = args.output.parent / (args.output.stem + "_traces")
            trace_path.mkdir(exist_ok=True)
            (trace_path / f"episode_{index:06d}.json").write_text(json.dumps(trace))
            payload = {
                "dataset": str(args.dataset.resolve()),
                "contract": contract,
                "max_state_error_threshold": args.max_state_error,
                "episodes": results,
            }
            if any(e["accepted"] for e in results):
                payload.update(summarize_grasps(results, args.range_margin_m))
            args.output.write_text(json.dumps(payload, indent=2))
            LOG.info(
                "episode=%d accepted=%s success=%s state_error=%.6g grasp_frames=%d",
                index,
                result["accepted"],
                success,
                max_error,
                len(points),
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
