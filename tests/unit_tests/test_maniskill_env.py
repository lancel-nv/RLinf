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

"""ManiSkill EEF action contracts and optional GPU preinsert integration."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.maniskill.eef_actions import decode_eef_actions
from rlinf.envs.sim.maniskill.peg_preinsert import PegPreinsertConfig
from rlinf.utils.eef_rot6d import EEFNormalizer
from rlinf.utils.torch_rot6d import canonicalize_rot6d, rot6d_to_matrix


@pytest.fixture
def eef_stats(tmp_path):
    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {
                        "q01": torch.linspace(-1.5, -0.5, 10).tolist(),
                        "q99": torch.linspace(0.8, 2.0, 10).tolist(),
                    }
                    for key in ("state", "actions")
                }
            }
        )
    )
    return str(path)


def _raw_actions(batch=3, steps=2):
    actions = torch.randn(batch, steps, 10) * 0.01
    actions[..., 3] += 1
    actions[..., 7] += 1
    actions[..., 9] = -1
    return actions


def test_eef_decoder_matches_dataset_output_and_normalization(eef_stats):
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
        ManiSkillEEFRot6dOutputs,
    )

    raw = _raw_actions()
    normalizer = EEFNormalizer(eef_stats)
    normalized = normalizer.normalize(raw, "actions")
    decoded = decode_eef_actions(normalizer.unnormalize(normalized, "actions"))
    expected = ManiSkillEEFRot6dOutputs()({"actions": raw.numpy()})["actions"]
    torch.testing.assert_close(decoded, torch.tensor(expected), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(raw, normalizer.unnormalize(normalized, "actions"))


@pytest.mark.parametrize("pitch", [-math.pi / 2, -0.1, 0.0, 0.1, math.pi / 2])
def test_eef_decoder_intrinsic_xyz_round_trip(pitch):
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    matrix = rotation.from_euler("XYZ", [0.08, pitch, -0.04]).as_matrix()
    raw = torch.zeros(1, 10, dtype=torch.float64)
    raw[:, 3:9] = torch.tensor(np.concatenate((matrix[:, 0], matrix[:, 1])))
    command = decode_eef_actions(raw)
    actual = rotation.from_euler("XYZ", command[0, 3:6].numpy() * -0.1).as_matrix()
    np.testing.assert_allclose(actual, matrix, atol=1e-6)


def test_eef_heads_ignore_redundant_rotation_coordinates(eef_stats):
    from rlinf.models.embodiment.prefix.policy import PrefixPolicy

    cfg = {"hidden_dims": [16, 16], "norm_stats_path": eef_stats}
    policy = PrefixPolicy(
        prefix_dim=4,
        proprio_dim=10,
        action_dim=10,
        num_action_chunks=2,
        ref_num_action_chunks=2,
        actor_head_cfg=OmegaConf.create(
            {**cfg, "name": "eef_rot6d_residual_mlp", "edit_scale": 0.02}
        ),
        critic_head_cfg=OmegaConf.create(
            {**cfg, "name": "eef_rot6d_multi_q_mlp", "num_q_heads": 4}
        ),
    )
    # Exercise the learned conditioning path, beyond the zero-initialized editor.
    torch.nn.init.normal_(policy.actor_head.editor[-1].weight, std=0.1)
    normalizer = EEFNormalizer(eef_stats)
    raw = _raw_actions()
    equivalent = raw.clone()
    equivalent[..., 3:6] *= 2
    equivalent[..., 6:9] = 3 * raw[..., 6:9] + raw[..., 3:6]
    a = normalizer.normalize(raw, "actions")
    b = normalizer.normalize(equivalent, "actions")
    visual = torch.randn(3, 4)
    obs_a = {
        "z_rl": visual,
        "proprio": normalizer.normalize(raw[:, 0], "state"),
        "ref_chunk": a,
    }
    obs_b = {
        "z_rl": visual,
        "proprio": normalizer.normalize(equivalent[:, 0], "state"),
        "ref_chunk": b,
    }
    out_a, lp_a, _ = policy.sac_forward(obs_a, deterministic=True)
    out_b, lp_b, _ = policy.sac_forward(obs_b, deterministic=True)
    torch.testing.assert_close(
        out_a - a.flatten(1), out_b - b.flatten(1), atol=1e-6, rtol=1e-4
    )
    torch.testing.assert_close(lp_a, lp_b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        policy.sac_q_forward(obs_a, a),
        policy.sac_q_forward(obs_b, b),
        atol=1e-6,
        rtol=1e-5,
    )
    assert (out_a - a.flatten(1)).abs().max() <= 0.020001
    loss = -policy.sac_q_forward(obs_a, out_a).mean()
    loss.backward()
    assert policy.actor_head.editor[-1].weight.grad.isfinite().all()
    assert policy.actor_head.editor[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "settings",
    [
        {"gap_m": 0},
        {"gap_m": -0.1},
        {"lateral_jitter_m": float("nan")},
        {"rotation_jitter_deg": 11},
    ],
)
def test_preinsert_validates_initial_distribution(settings):
    with pytest.raises(ValueError):
        PegPreinsertConfig(**settings)


def test_rot6d_degenerate_frames_have_finite_right_handed_rotations():
    vectors = torch.tensor(
        [[0.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 2, 0, 0]], requires_grad=True
    )
    matrix = rot6d_to_matrix(vectors)
    torch.testing.assert_close(
        matrix.transpose(-1, -2) @ matrix, torch.eye(3).expand(2, -1, -1)
    )
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(2))
    canonicalize_rot6d(vectors).sum().backward()
    assert vectors.grad.isfinite().all()


def test_preinsert_prefix_config_composes_and_builds_expo(eef_stats, monkeypatch):
    from hydra import compose, initialize_config_dir

    from rlinf.algorithms.prefix_off_policy import build_prefix_off_policy_algorithm
    from rlinf.models.embodiment.prefix import get_model

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_prefix_expo_openpi_pi05_eef_preinsert_async"
        )
    for head in ("actor_head", "critic_head"):
        cfg.actor.model[head].norm_stats_path = eef_stats
    policy = get_model(cfg.actor.model)
    algorithm = build_prefix_off_policy_algorithm(cfg)
    assert cfg.env.train.init_params.preinsert.gap_m == 0.05
    assert cfg.env.train.init_params.hole_clearance == 0.001
    assert cfg.env.train.max_episode_steps == 100
    assert cfg.env.train.reward_mode == "only_success"
    assert cfg.env.eval.video_cfg.save_video
    assert cfg.rollout.prefix_feature_model.prefix.action_space == "normalized_model"
    assert policy.flat_action_dim == 100
    assert algorithm.target_entropy == pytest.approx(100 * (math.log(0.02) - 0.5))
    raw = _raw_actions(batch=2, steps=10)
    normalizer = EEFNormalizer(eef_stats)
    obs = {
        "z_rl": torch.zeros(2, 2048),
        "proprio": normalizer.normalize(raw[:, 0], "state"),
        "ref_chunk": normalizer.normalize(raw, "actions"),
    }
    obs["candidates"] = obs["ref_chunk"][:, None].expand(-1, 8, -1, -1)
    action, _ = algorithm.select_actions(policy, policy, obs)
    assert action.shape == (2, 100)
    assert torch.isfinite(action).all()


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires ManiSkill GPU physics, renderer, and GPU IK assets.",
)
@pytest.mark.parametrize("seed", [420000, 420002])
def test_preinsert_gpu_reset_normalized_execution_and_replay(eef_stats, seed):
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from toolkits.lerobot.collect_maniskill_peg_preinsert import (
        encode_action,
        expert_action,
    )

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/env/maniskill_peg_preinsert.yaml"
    )
    if os.environ.get("RLINF_TEST_RENDER_BACKEND"):
        cfg.init_params.render_backend = os.environ["RLINF_TEST_RENDER_BACKEND"]
    cfg.normalized_actions_stats_path = eef_stats
    cfg.video_cfg.save_video = False
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    normalizer = EEFNormalizer(eef_stats).to(env.device)
    try:
        task = env.env.unwrapped
        obs, info = env.reset(seed=seed, options={"reconfigure": True})
        initial = obs["states"].clone()
        head = (task.box_hole_pose.inv() * task.peg_head_pose).p
        torch.testing.assert_close(
            -task.peg_half_sizes[:, 0] - head[:, 0],
            torch.full((1,), 0.05, device=env.device),
            atol=2e-6,
            rtol=0,
        )
        assert head[:, 1:].abs().max() <= 0.030001
        assert task.hole_clearance == 0.001
        assert not info["success"].any()
        assert obs["states"].shape == (1, 10)
        assert "privileged_state" not in obs
        commands = []
        labels = []
        for _ in range(100):
            label = torch.tensor(
                encode_action(task, expert_action(task)), device=env.device
            )[None]
            labels.append(label.cpu().numpy())
            normalized = normalizer.normalize(label, "actions")
            commands.append(normalized.clone())
            result = env.chunk_step(normalized[:, None])
            if result[2].any() or result[3].any():
                break
        success = bool(result[-1][-1]["success"].all())
        if seed == 420000:
            assert success
        final = result[0][-1]["states"].clone()
        obs, _ = env.reset(seed=seed, options={"reconfigure": True})
        torch.testing.assert_close(obs["states"], initial, atol=2e-5, rtol=0)
        for command in commands:
            result = env.chunk_step(command[:, None])
        assert bool(result[-1][-1]["success"].all()) == success
        torch.testing.assert_close(result[0][-1]["states"], final, atol=0.002, rtol=0)
    finally:
        env.env.close()

    from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
        ManiSkillEEFRot6dOutputs,
    )

    cfg.normalized_actions_stats_path = None
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    try:
        obs, _ = env.reset(seed=seed, options={"reconfigure": True})
        torch.testing.assert_close(obs["states"], initial, atol=2e-5, rtol=0)
        decoder = ManiSkillEEFRot6dOutputs()
        for label in labels:
            command = torch.tensor(
                decoder({"actions": label})["actions"], device=env.device
            )
            result = env.chunk_step(command[:, None])
        assert bool(result[-1][-1]["success"].all()) == success
        torch.testing.assert_close(result[0][-1]["states"], final, atol=0.002, rtol=0)
    finally:
        env.env.close()


def test_preinsert_sft_config_matches_eef_data_contract(monkeypatch):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/sft/config"), version_base="1.1"
    ):
        cfg = compose(config_name="maniskill_peg_preinsert_eef_sft_openpi_pi05")
    assert cfg.runner.max_steps == 2000
    assert cfg.actor.global_batch_size == 256
    assert cfg.actor.model.action_dim == cfg.actor.model.openpi.action_env_dim == 10
    assert cfg.actor.model.openpi.config_name == "pi05_maniskill_eef_rot6d"
    assert not cfg.actor.model.openpi.use_rlt
    assert cfg.actor.model.prefix.pool == "masked_mean"
    assert cfg.actor.optim.total_training_steps == 3000
    assert cfg.actor.optim.lr_warmup_steps == 1000
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    data = get_openpi_config(cfg.actor.model.openpi.config_name).data
    assert data.eef_rot6d and data.output_action_dim == 10
    assert not data.extra_delta_transform
