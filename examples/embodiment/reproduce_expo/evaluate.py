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


"""Evaluate the trained GPU EEF rot6d checkpoint on 50 held-out episodes."""

import argparse
import hashlib
import json
import logging
import math
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="RL checkpoint actor directory; omit for SFT baseline",
)
parser.add_argument(
    "--updates",
    type=int,
    default=0,
    help="Expected critic updates, verified from saved budget",
)
parser.add_argument("--seed", type=int, default=430000)
parser.add_argument("--render-backend")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
if bool(args.checkpoint) != bool(args.updates):
    parser.error("--checkpoint and positive --updates must be supplied together")
if args.updates < 0:
    parser.error("--updates must be nonnegative")
OUT = args.output_dir.absolute()
CODE = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    """Hash weights without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict | list) -> None:
    """Publish a complete JSON result atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def wilson(k: int, n: int) -> list[float]:
    """Return the Wilson 95% interval for k successes in n episodes."""
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, middle - half), min(1, middle + half)]


def main() -> None:
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from rlinf.algorithms.sac_residual import predict_residual_actions
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        validate_eef_gpu_controller,
    )
    from rlinf.models.embodiment.mlp_policy.expo_policy import EXPOConfig, EXPOPolicy
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.utils.logging import get_logger

    logger = get_logger()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    torch.set_num_threads(4)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Expected an empty evaluation directory, found {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    assert (
        not cfg.actor.model.actor_use_privileged_state
        and cfg.actor.model.privileged_state_dim == 0
    )
    assert not cfg.env.eval.critic_privileged_obs
    assert (
        cfg.env.eval.init_params.fixed_peg_half_length == 0.105
        and cfg.env.eval.init_params.fixed_peg_half_width == 0.02
    )
    model = OmegaConf.create(
        OmegaConf.to_container(cfg.rollout.residual_base_model, resolve=True)
    )
    model.openpi.torch_compile = False
    ecfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval, resolve=True))
    ecfg.total_num_envs = 10
    ecfg.seed = args.seed
    ecfg.video_cfg.save_video = True
    ecfg.init_params.max_episode_steps = 100
    if args.render_backend:
        OmegaConf.update(
            ecfg, "init_params.render_backend", args.render_backend, force_add=True
        )
    OmegaConf.save(model, OUT / "model.yaml", resolve=True)
    OmegaConf.save(ecfg, OUT / "env.yaml", resolve=True)
    norm = Path(model.openpi_data.norm_stats_path)
    manifest = json.loads(norm.with_name("manifest.json").read_text())
    assert (
        manifest["episodes"] == 400
        and manifest["episodes"] == manifest["dataset"]["episodes"]
        and manifest["dataset"]["sim_backend"] == "gpu"
    )
    assert (
        manifest["dataset"]["fresh_collection"]
        and manifest["dataset"]["clearance_m"] == 0.001
    )
    assert (
        manifest["norm_stats_sha256"] == hashlib.sha256(norm.read_bytes()).hexdigest()
    )
    base = (
        get_model(model, torch_dtype=torch.bfloat16)
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    checkpoint = args.checkpoint
    weights = Path(model.model_path) / "model_state_dict/full_weights.pt"
    if args.updates:
        saved = torch.load(
            checkpoint / "residual_state_rank_0.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert saved["budget"]["updates"] == args.updates
        settings = EXPOConfig(
            **{
                k: cfg.actor.model[k]
                for k in EXPOConfig.__dataclass_fields__
                if k in cfg.actor.model
            }
        )
        policy = EXPOPolicy(settings)
        expected_buffers = {k: v.detach().clone() for k, v in policy.named_buffers()}
        weights = checkpoint / "model_state_dict/full_weights.pt"
        state = torch.load(weights, map_location="cpu", weights_only=True)
        assert all(
            torch.isfinite(v).all()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        )
        policy.load_state_dict(state, strict=True)
        for k, v in policy.named_buffers():
            torch.testing.assert_close(v, expected_buffers[k], rtol=0, atol=0)
        policy = policy.to("cuda:0").eval().requires_grad_(False)
        encoder = None
        assert torch.allclose(
            policy.edit_scale, torch.full_like(policy.edit_scale, 0.02)
        )
    env = ManiskillRLTEnv(ecfg, 10, 0, 1, None)
    records = []
    start = time.monotonic()
    initial_states = []
    try:
        contract = validate_eef_gpu_controller(env.env)
        metadata = {
            "checkpoint": str(checkpoint),
            "critic_updates": args.updates,
            "runner_step": checkpoint.parent.name if checkpoint else None,
            "weights_sha256": sha256_file(weights),
            "base_checkpoint": str(model.model_path),
            "base_checkpoint_step": 2000,
            "episodes": 50,
            "seeds": list(range(args.seed, (args.seed + 50))),
            "num_envs": 10,
            "max_steps": 100,
            "action_chunk": 10,
            "denoising_steps": 5,
            "residual_mode": "EXPO stochastic 8 proposals + 8 edits; target-Q selection"
            if args.updates
            else "base only",
            "edit_scale": 0.02,
            "visual_encoder": {
                "type": "pi05_prefix_masked_mean",
                "dim": 2048,
                "token_scope": "all_valid_image_prompt_state_tokens",
                "frozen": True,
            },
            "metric": "success at any active step before first termination or 100-step limit",
            "noise_seed_rule": "900000 + batch_offset * 100 + chunk_index",
            "contract": contract,
            "norm_stats_sha256": manifest["norm_stats_sha256"],
            "model": OmegaConf.to_container(model, resolve=True),
            "environment": OmegaConf.to_container(ecfg, resolve=True),
            "device": torch.cuda.get_device_name(0),
            "code_directory": str(CODE),
            "note": "EXPO pi05 pooled features on SFT2000, 400 fresh 1mm demonstrations; fixed scene and sampling seeds; SR updated after every finished episode.",
        }
        metadata["video_recording"] = {
            "enabled": True,
            "all_episodes": True,
            "fps": 10,
            "views": ["main_images", "wrist_images"],
            "layout": "side_by_side",
            "terminal_frame": True,
        }
        write_json(OUT / "metadata.json", metadata)
        for offset in range(0, 50, 10):
            seeds = list(range(args.seed + offset, (args.seed + 10) + offset))
            obs, _ = env.reset(seed=seeds, options={"reconfigure": True})
            assert "privileged_state" not in obs
            task = env.env.unwrapped
            assert validate_eef_gpu_controller(env.env) == contract
            assert (
                task.hole_clearance
                == task.success_clearance
                == ecfg.init_params.hole_clearance
                == 0.001
            )
            torch.testing.assert_close(
                task.peg_half_sizes,
                task.peg_half_sizes.new_tensor([[0.105, 0.02, 0.02]]).expand(10, -1),
            )
            torch.testing.assert_close(
                task.box_hole_radii, torch.full_like(task.box_hole_radii, 0.021)
            )
            h = hashlib.sha256()
            for value in [
                task.agent.robot.qpos,
                task.peg.pose.raw_pose,
                task.box.pose.raw_pose,
                task.peg_half_sizes,
                task.box_hole_radii,
            ]:
                h.update(value.cpu().numpy().tobytes())
            scene = {
                "qpos": task.agent.robot.qpos.cpu().tolist(),
                "peg_pose": task.peg.pose.raw_pose.cpu().tolist(),
                "box_pose": task.box.pose.raw_pose.cpu().tolist(),
                "peg_half_sizes": task.peg_half_sizes.cpu().tolist(),
                "hole_radii": task.box_hole_radii.cpu().tolist(),
            }
            initial_states.append(
                {
                    "seeds": seeds,
                    "initial_state_sha256": h.hexdigest(),
                    "initial_state": scene,
                }
            )
            write_json(OUT / "initial_states.json", initial_states)
            active = torch.ones(10, device=env.device, dtype=torch.bool)
            grasp = torch.zeros_like(active)
            lengths = torch.zeros(10, device=env.device, dtype=torch.long)
            writers = {
                i: imageio.get_writer(
                    OUT / f"seed_{seeds[i]}.mp4",
                    fps=10,
                    codec="libx264",
                    output_params=["-threads", "1"],
                )
                for i in range(10)
            }
            try:
                for step in range(100):
                    for i, writer in writers.items():
                        writer.append_data(
                            torch.cat(
                                [obs["main_images"][i], obs["wrist_images"][i]], dim=1
                            )
                            .cpu()
                            .numpy()
                        )
                    if step % 10 == 0:
                        noise_seed = 900000 + offset * 100 + step // 10
                        torch.manual_seed(noise_seed)
                        np.random.seed(noise_seed)
                        with torch.inference_mode():
                            if args.updates:
                                actions, policy_info = predict_residual_actions(
                                    policy,
                                    base,
                                    obs,
                                    mode="eval",
                                    visual_encoder=encoder,
                                )
                                assert not any(
                                    "privileged" in k
                                    for k in policy_info["forward_inputs"]
                                )
                            else:
                                _, model_actions, model_state = (
                                    base.extract_residual_obs(obs)
                                )
                                actions = base.decode_actions(
                                    model_actions, model_state
                                )
                        assert (
                            tuple(actions.shape) == (10, 10, 7)
                            and torch.isfinite(actions).all()
                        )
                    action = actions[:, step % 10].to(env.device).clone()
                    action[~active] = 0
                    obs, _, termination, truncation, info = env.step(
                        action, auto_reset=False
                    )
                    lengths += active.long()
                    grasp |= task.agent.is_grasping(task.peg) & active
                    success = info["success"].bool() & active
                    finished = active & (
                        success | termination.bool() | truncation.bool() | (step == 99)
                    )
                    for i in finished.nonzero().flatten().tolist():
                        records.append(
                            {
                                "seed": seeds[i],
                                "success": bool(success[i]),
                                "steps": int(lengths[i]),
                                "first_success_step": int(lengths[i])
                                if success[i]
                                else -1,
                                "grasped_once": bool(grasp[i]),
                                "terminated": bool(termination[i]),
                                "truncated": bool(truncation[i]),
                            }
                        )
                        active[i] = False
                        successes = sum(r["success"] for r in records)
                        write_json(OUT / "episodes.json", records)
                        progress = {
                            "completed": len(records),
                            "target": 50,
                            "successes": successes,
                            "success_rate": successes / len(records),
                            "wilson_95_ci": wilson(successes, len(records)),
                            "elapsed_seconds": time.monotonic() - start,
                        }
                        write_json(OUT / "progress.json", progress)
                        logger.info(
                            "EPISODE seed=%s success=%s SR=%d/%d (%.2f%%)",
                            seeds[i],
                            bool(success[i]),
                            successes,
                            len(records),
                            progress["success_rate"] * 100,
                        )
                        if i in writers:
                            writers[i].append_data(
                                torch.cat(
                                    [obs["main_images"][i], obs["wrist_images"][i]],
                                    dim=1,
                                )
                                .cpu()
                                .numpy()
                            )
                            writers.pop(i).close()
                    if not active.any():
                        break
                    if (step + 1) % 100 == 0:
                        logger.info(
                            "BATCH seed=%s step=%s active=%s",
                            seeds[0],
                            step + 1,
                            int(active.sum()),
                        )
            finally:
                for writer in writers.values():
                    writer.close()
        assert len(records) == 50 and {r["seed"] for r in records} == set(
            range(args.seed, (args.seed + 50))
        )
        assert {p.name for p in OUT.glob("seed_*.mp4")} == {
            f"seed_{seed}.mp4" for seed in range(args.seed, (args.seed + 50))
        }
        successes = sum(r["success"] for r in records)
        result = {
            "episodes": 50,
            "successes": successes,
            "success_rate": successes / 50,
            "wilson_95_ci": wilson(successes, 50),
            "grasped_once": sum(r["grasped_once"] for r in records),
            "elapsed_seconds": time.monotonic() - start,
            "critic_updates": args.updates,
            "control_contract": contract,
            "norm_stats_sha256": manifest["norm_stats_sha256"],
        }
        write_json(OUT / "summary.json", result)
        logger.info("FINISHED %s", result)
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
