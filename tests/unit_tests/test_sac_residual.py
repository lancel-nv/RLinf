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

"""Residual SAC density, objectives, replay boundaries and update scheduling."""

import asyncio
import copy
import math
import os
import queue
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributions import Normal, TransformedDistribution, transforms

from rlinf.algorithms.sac_residual import (
    ResidualUpdateBudget,
    extract_residual_obs,
    random_min_q,
    residual_actor_loss,
    residual_critic_loss,
    validate_residual_cfg,
)
from rlinf.data.schema.embodied_types import (
    EnvPart,
    EnvTransition,
    PolicyOutput,
    PolicyPart,
    TrajectoryKey,
    TrajectorySource,
    TrajectoryStep,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.sac_residual_policy import (
    SACResidualConfig,
    SACResidualPolicy,
)


def make_policy():
    return SACResidualPolicy(
        SACResidualConfig(
            state_dim=3,
            visual_dim=5,
            state_proj_dim=4,
            visual_proj_dim=6,
            action_dim=2,
            num_action_chunks=3,
            hidden_size=8,
            depth=2,
            num_q_heads=4,
            edit_scale=[0.1, 0.3],
        )
    )


def make_obs(batch_size=4):
    return {
        "state": torch.randn(batch_size, 3),
        "visual": torch.randn(batch_size, 5),
        "base_action": torch.randn(batch_size, 6) * 2,
    }


def test_residual_density_includes_tanh_and_scale_once():
    policy = make_policy()
    obs = make_obs()
    actions, log_prob, _ = policy(forward_type=ForwardType.SAC, obs=obs)
    delta = actions - obs["base_action"]
    distribution = TransformedDistribution(
        Normal(torch.zeros_like(actions), torch.full_like(actions, math.exp(-1.5))),
        [transforms.TanhTransform(), transforms.AffineTransform(0, policy.edit_scale)],
    )
    torch.testing.assert_close(
        log_prob, distribution.log_prob(delta), atol=2e-5, rtol=2e-5
    )
    assert torch.all(delta.abs() <= policy.edit_scale)
    deterministic, _, _ = policy.sac_forward(obs, deterministic=True)
    torch.testing.assert_close(deterministic, obs["base_action"])
    assert (
        deterministic.abs().max() > 1
    )  # No implicit clipping of normalized proposals.


def test_residual_actor_detaches_features_but_critic_trains_them():
    policy = make_policy()
    obs = make_obs()
    loss, log_prob, _ = residual_actor_loss(policy, obs, torch.tensor(0.0002))
    loss.backward()
    assert all(p.grad is None for p in policy.encoder.parameters())
    assert policy.editor[-1].weight.grad.norm() > 0
    assert log_prob.shape == (4, 1)
    policy.zero_grad(set_to_none=True)
    policy.sac_q_forward(obs, obs["base_action"]).square().mean().backward()
    assert any(
        p.grad is not None and p.grad.norm() > 0 for p in policy.encoder.parameters()
    )
    assert all(p.grad is None for p in policy.editor.parameters())


@pytest.mark.parametrize("alpha_value", [1e-5, 2e-4])
def test_residual_actor_gradient_ratio_matches_components_without_changing_update(
    alpha_value,
):
    torch.manual_seed(61)
    policy = make_policy()
    obs = make_obs()
    params = list(policy.editor.parameters())
    alpha = torch.tensor(alpha_value, requires_grad=True)
    reference = copy.deepcopy(policy)
    ref_params = list(reference.editor.parameters())
    torch.manual_seed(71)
    actions, log_prob, _ = reference.sac_forward(obs)
    values = reference.sac_q_forward(obs, actions, detach_encoder=True)
    q = random_min_q(values, torch.Generator().manual_seed(81))
    (-q.mean()).backward(retain_graph=True)
    q_norm = torch.cat([p.grad.flatten() for p in ref_params]).norm()
    reference.zero_grad(set_to_none=True)
    (alpha.detach() * log_prob.sum(-1).mean()).backward()
    entropy_norm = torch.cat([p.grad.flatten() for p in ref_params]).norm()

    torch.manual_seed(71)
    loss, _, metrics = residual_actor_loss(
        policy, obs, alpha, torch.Generator().manual_seed(81), actor_parameters=params
    )
    assert set(metrics) == {"actor_q_to_entropy_grad_ratio"}
    assert metrics["actor_q_to_entropy_grad_ratio"] == pytest.approx(
        (q_norm / entropy_norm).item(), rel=1e-5
    )
    rng_after_monitor = torch.get_rng_state()
    assert all(p.grad is None for p in policy.parameters())
    assert alpha.grad is None
    loss.backward()
    gradients = [p.grad.clone() for p in params]
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(71)
    plain_loss, _, plain_metrics = residual_actor_loss(
        policy, obs, alpha, torch.Generator().manual_seed(81)
    )
    assert not plain_metrics
    torch.testing.assert_close(torch.get_rng_state(), rng_after_monitor, rtol=0, atol=0)
    torch.testing.assert_close(loss, plain_loss, rtol=0, atol=0)
    plain_loss.backward()
    for parameter, gradient in zip(params, gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert alpha.grad is None


@pytest.mark.parametrize("bootstrap_time_limits", [False, True])
@pytest.mark.parametrize("backup_entropy", [False, True])
def test_residual_target_matches_reference_chunk_discount_and_timeout(
    backup_entropy, bootstrap_time_limits
):
    policy = make_policy()
    target = copy.deepcopy(policy).requires_grad_(False)
    batch = {
        "curr_obs": make_obs(),
        "next_obs": make_obs(),
        "actions": torch.randn(4, 6),
        "rewards": torch.tensor(
            [[0.0, 0.0, 20.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
        ),
        "terminations": torch.tensor(
            [[False, False, True], [False] * 3, [False] * 3, [False] * 3]
        ),
        "truncations": torch.tensor(
            [[False] * 3, [False, False, True], [False] * 3, [False] * 3]
        ),
    }
    alpha = torch.tensor(0.0002)
    torch.manual_seed(32)
    rng = torch.Generator().manual_seed(41)
    actual, metrics = residual_critic_loss(
        policy,
        target,
        batch,
        alpha,
        0.99,
        backup_entropy,
        rng,
        bootstrap_time_limits=bootstrap_time_limits,
    )
    torch.manual_seed(32)
    with torch.no_grad():
        action, logp, _ = policy.sac_forward(batch["next_obs"])
        qs = target.sac_q_forward(batch["next_obs"], action)
        pair = torch.randperm(4, generator=torch.Generator().manual_seed(41))[:2]
        continuation = qs[:, pair].amin(-1, keepdim=True)
        if backup_entropy:
            continuation -= alpha * logp.sum(-1, keepdim=True)
        ended = batch["terminations"].any(-1, keepdim=True)
        if not bootstrap_time_limits:
            ended |= batch["truncations"].any(-1, keepdim=True)
        expected_target = (
            batch["rewards"].sum(-1, keepdim=True) + 0.99 * (~ended) * continuation
        )
    expected = (
        (policy.sac_q_forward(batch["curr_obs"], batch["actions"]) - expected_target)
        .square()
        .mean()
    )
    torch.testing.assert_close(actual, expected)
    assert expected_target[0].item() == 20
    torch.testing.assert_close(
        expected_target[1], (0.99 if bootstrap_time_limits else 0.0) * continuation[1]
    )
    assert metrics["target_q"] == pytest.approx(expected_target.mean().item())
    actual.backward()
    assert all(p.grad is None for p in target.parameters())


def test_residual_random_pair_is_without_replacement():
    values = torch.arange(10.0).expand(3, -1)
    rng = torch.Generator().manual_seed(77)
    expected_indices = torch.randperm(10, generator=torch.Generator().manual_seed(77))[
        :2
    ]
    torch.testing.assert_close(
        random_min_q(values, rng), values[:, expected_indices].amin(-1, keepdim=True)
    )


def test_residual_entropy_tracks_action_dimension_and_scale():
    config = SACResidualConfig()
    assert config.target_entropy() == pytest.approx(80 * (math.log(0.2) - 0.5))
    old = SACResidualConfig(action_dim=9)
    assert old.target_entropy() == pytest.approx(-189.84941211906906)
    with pytest.raises(ValueError, match="target entropy"):
        config.target_entropy(-40)
    per_axis = SACResidualConfig(
        action_dim=2, num_action_chunks=3, edit_scale=[0.1, 0.3]
    )
    assert per_axis.target_entropy() == pytest.approx(
        3 * (math.log(0.1) + math.log(0.3) - 1)
    )


def test_residual_alpha_gradient_moves_toward_requested_entropy():
    for entropy_gap in (-2.0, 2.0):
        log_alpha = torch.tensor(math.log(0.0002), requires_grad=True)
        (log_alpha.exp() * entropy_gap).backward()
        assert math.copysign(1, log_alpha.grad.item()) == math.copysign(1, entropy_gap)


def test_residual_utd_budget_survives_resume_and_retains_update_debt():
    budget = ResidualUpdateBudget()
    budget.add(1000)
    assert budget.available() == 0
    budget.add(40)
    assert budget.available() == 400
    budget.updates = 2000
    assert budget.available() == 0
    budget.add(30)
    assert budget.available() == 400
    budget.updates += 400
    restored = ResidualUpdateBudget(**asdict(budget))
    assert restored.available() == 200
    restored.updates += 200
    assert restored.available() == 0


@pytest.mark.parametrize("privileged", [False, True])
def test_residual_collector_preserves_proposal_and_normalized_actions_at_timeout(
    privileged,
):
    obs, next_obs = make_obs(1), make_obs(1)
    if privileged:
        from rlinf.data.schema.peg_privileged import PEG_PRIVILEGED_DIM

        obs["privileged_state"] = torch.randn(1, PEG_PRIVILEGED_DIM)
        next_obs["privileged_state"] = torch.randn(1, PEG_PRIVILEGED_DIM)
    normalized = obs["base_action"] + 0.02
    forward = {"action": normalized}
    forward.update({"residual_" + key: value for key, value in obs.items()})
    source = TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)
    policy = PolicyPart(
        sources=[source],
        obs={"states": torch.zeros(1, 9)},
        output=PolicyOutput(forward_inputs=forward),
    )
    env = EnvPart(
        sources=[source],
        transition=EnvTransition(
            rewards=torch.zeros(1, 3),
            terminations=torch.zeros(1, 3, dtype=torch.bool),
            truncations=torch.tensor([[False, False, True]]),
            dones=torch.tensor([[False, False, True]]),
        ),
        next_obs=next_obs,
    )
    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=env.transition.rewards,
        collect_prev_infos=False,
        collect_transitions=True,
        enable_rlt=False,
        enable_residual=True,
        include_final_value=False,
    )
    torch.testing.assert_close(step.actions, normalized)
    for key in obs:
        torch.testing.assert_close(step.curr_obs[key], obs[key])
        torch.testing.assert_close(step.next_obs[key], next_obs[key])
    extracted = extract_residual_obs(forward)
    torch.testing.assert_close(extracted["base_action"], obs["base_action"])


def load_recipe(monkeypatch, name="maniskill_sac_residual_openpi_pi05"):
    root = Path(__file__).resolve().parents[2] / "examples/embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(root))
    with initialize_config_dir(config_dir=str(root / "config"), version_base="1.3"):
        return compose(config_name=name)


def test_residual_recipe_and_hyperparameter_validation(monkeypatch):
    cfg = load_recipe(monkeypatch)
    validate_residual_cfg(cfg)
    assert cfg.algorithm.critic_actor_ratio == 20
    assert cfg.algorithm.entropy_tuning.initial_alpha == 0.0002
    assert cfg.actor.optim.adam_beta2 == 0.999
    assert not cfg.reward.use_reward_model
    assert cfg.reward.get("env_reward_weight", 1.0) == 1.0
    assert cfg.env.train.reward_mode == "only_success"
    assert not cfg.env.train.use_rel_reward
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    transition = EnvTransition(rewards=rewards)
    actual = transition.compute_rewards(
        bootstrap_values=None,
        env_reward_weight=cfg.reward.get("env_reward_weight", 1.0),
        reward_weight=cfg.reward.get("reward_weight", 1.0),
        auto_reset=cfg.env.train.auto_reset,
        bootstrap_type=cfg.algorithm.bootstrap_type,
        gamma=cfg.algorithm.gamma,
    )
    torch.testing.assert_close(actual, rewards)
    bad = OmegaConf.create(OmegaConf.to_container(cfg))
    bad.algorithm.entropy_tuning.target_entropy = -40
    with pytest.raises(ValueError, match="target entropy"):
        validate_residual_cfg(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edit_scale": 0},
        {"edit_scale": float("nan")},
        {"edit_scale": [0.2]},
        {"initial_log_std": -6},
        {"num_q_heads": 1},
    ],
)
def test_residual_rejects_invalid_model_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        SACResidualPolicy(SACResidualConfig(**kwargs))


def make_worker(worker_cls=None):
    from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
    from rlinf.workers.actor.fsdp_sac_residual_policy_worker import (
        SACResidualFSDPPolicy,
    )

    worker = object.__new__(worker_cls or SACResidualFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "gamma": 0.99,
                "tau": 0.005,
                "backup_entropy": False,
                "entropy_tuning": {"optim": {"clip_grad": 1.0}},
            },
            "actor": {"optim": {"clip_grad": 1.0}, "critic_optim": {"clip_grad": 1.0}},
        }
    )
    worker.device = torch.device("cpu")
    worker.model = make_policy()
    worker.target_model = copy.deepcopy(worker.model).requires_grad_(False)
    worker.target_model_initialized = True
    worker.target_update_type = "all"
    worker.use_dsrl = False
    worker.entropy_temp = EntropyTemperature(
        0.0002, "exp", device="cpu", dtype=torch.float32
    )
    worker.target_entropy = worker.model.config.target_entropy()
    worker.critic_sample_generator = torch.Generator().manual_seed(1)
    worker.critic_actor_ratio = 20
    worker.optimizer = torch.optim.Adam(worker.model.editor.parameters(), lr=0.0003)
    worker.qf_optimizer = torch.optim.Adam(
        [*worker.model.encoder.parameters(), *worker.model.q_head.parameters()],
        lr=0.0003,
    )
    worker.alpha_optimizer = torch.optim.Adam(
        worker.entropy_temp.parameters(), lr=0.0003
    )
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1.0
    )
    worker.qf_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.qf_optimizer, lambda _: 1.0
    )
    worker.alpha_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.alpha_optimizer, lambda _: 1.0
    )
    batch = {
        "curr_obs": make_obs(),
        "next_obs": make_obs(),
        "actions": torch.randn(4, 6),
        "rewards": torch.ones(4, 3),
        "terminations": torch.zeros(4, 3, dtype=torch.bool),
    }
    worker.buffer_dataloader_iter = iter([batch] * 20)
    return worker


def test_residual_worker_delays_actor_and_updates_temperature_with_same_sample():
    worker = make_worker()
    initial_editor = copy.deepcopy(worker.model.editor.state_dict())
    initial_alpha = worker.entropy_temp.base_alpha.detach().clone()
    for update in range(20):
        worker.update_step = update
        old_target = copy.deepcopy(worker.target_model.encoder.state_dict())
        metrics = worker.update_one_epoch()
        for key, value in worker.target_model.encoder.state_dict().items():
            torch.testing.assert_close(
                value,
                old_target[key] * 0.995
                + worker.model.encoder.state_dict()[key] * 0.005,
            )
        assert metrics["sac/actor_updated"] == float(update == 19)
        assert ("actor_q_to_entropy_grad_ratio" in metrics) == (update == 19)
        if update == 19:
            assert math.isfinite(metrics["actor_q_to_entropy_grad_ratio"])
            assert metrics["actor_q_to_entropy_grad_ratio"] >= 0
        if update < 19:
            for key, value in worker.model.editor.state_dict().items():
                torch.testing.assert_close(value, initial_editor[key], atol=0, rtol=0)
            torch.testing.assert_close(
                worker.entropy_temp.base_alpha, initial_alpha, atol=0, rtol=0
            )
    assert not torch.equal(worker.model.editor[-1].weight, initial_editor["6.weight"])
    assert not torch.equal(worker.entropy_temp.base_alpha, initial_alpha)
    assert all(p.grad is None for p in worker.model.encoder.parameters())


@pytest.mark.parametrize("expo", [False, True])
@pytest.mark.parametrize("auto_reset,expected_count", [(False, 4), (True, 6)])
def test_residual_replay_counts_decisions_and_keeps_timeout_features(
    auto_reset, expected_count, expo
):
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
    from rlinf.workers.actor.fsdp_sac_residual_policy_worker import (
        SACResidualFSDPPolicy,
    )

    worker = object.__new__(SACResidualFSDPPolicy)
    worker.cfg = OmegaConf.create({"env": {"train": {"auto_reset": auto_reset}}})
    worker.replay_buffer = TrajectoryReplayBuffer(sample_window_size=10)
    trajectory = make_trajectory()
    if expo:
        trajectory.curr_obs["candidates"] = torch.randn(3, 2, 8, 6)
        trajectory.next_obs["candidates"] = torch.randn(3, 2, 8, 6) + 100
    try:
        rows = worker.transition_rows(trajectory)
        assert len(rows) == expected_count
        torch.testing.assert_close(rows[0].actions[0, 0], trajectory.actions[0, 0])
        assert rows[0].terminations.any()
        assert rows[-1].truncations.any()
        assert not rows[-1].terminations.any()
        for key in trajectory.next_obs:
            torch.testing.assert_close(
                rows[-1].next_obs[key][0, 0], trajectory.next_obs[key][-1, 1]
            )
        worker.replay_buffer.add_trajectories(rows)
        samples = worker.replay_buffer.sample_chunks(2)
        assert samples["actions"].shape == (2, 6)
        if expo:
            assert samples["next_obs"]["candidates"].shape == (2, 8, 6)
            assert samples["next_obs"]["candidates"].mean() > 90
        valid_actions = torch.cat([row.actions.flatten(0, 1) for row in rows])
        assert (
            (samples["actions"][:, None] == valid_actions[None]).all(-1).any(-1).all()
        )
    finally:
        worker.replay_buffer.close()


@pytest.mark.parametrize(
    "override,match",
    [
        ({"rollout": {"collect_final_values": False}}, "terminal features"),
        ({"algorithm": {"target_update_freq": 2}}, "every critic step"),
        ({"runner": {"save_critic_interval": 0}}, "save_critic_interval"),
        (
            {"runner": {"save_critic_interval": 100000, "max_critic_updates": None}},
            "save_critic_interval",
        ),
        (
            {"actor": {"fsdp_config": {"mixed_precision": {"param_dtype": "bf16"}}}},
            "FP32",
        ),
    ],
)
def test_residual_rejects_unsupported_execution_settings(monkeypatch, override, match):
    cfg = OmegaConf.merge(
        OmegaConf.to_container(load_recipe(monkeypatch), resolve=True), override
    )
    with pytest.raises(ValueError, match=match):
        validate_residual_cfg(cfg)


def make_trajectory():
    from rlinf.data.schema.embodied_types import Trajectory

    trajectory = Trajectory(max_episode_length=9)
    trajectory.actions = torch.arange(36.0).reshape(3, 2, 6)
    trajectory.rewards = torch.zeros(3, 2, 3)
    trajectory.terminations = torch.zeros(4, 2, 3, dtype=torch.bool)
    trajectory.truncations = torch.zeros_like(trajectory.terminations)
    trajectory.terminations[1, 0, -1] = True
    trajectory.truncations[3, 1, -1] = True
    trajectory.dones = trajectory.terminations | trajectory.truncations
    trajectory.curr_obs = {
        key: value.reshape(3, 2, -1) for key, value in make_obs(6).items()
    }
    trajectory.next_obs = {
        key: value.reshape(3, 2, -1) for key, value in make_obs(6).items()
    }
    return trajectory


def test_residual_resume_keeps_current_call_cap_and_pending_updates():
    saved = ResidualUpdateBudget(max_updates_per_iteration=10000)
    saved.add(1100)
    assert saved.available() == 2000
    saved.updates = 2000
    saved.add(100)
    restored = ResidualUpdateBudget(max_updates_per_iteration=400)
    restored.restore(asdict(saved))
    assert restored.pending() == saved.pending() == 2000
    assert restored.available() == 400
    assert restored.updates == 2000
    assert restored.warmup_collected == 1100
    restored.updates += 400
    assert restored.pending() == 1600


@pytest.mark.parametrize("setting", ["utd", "warmup_transitions", "warmup_updates"])
def test_residual_resume_rejects_changed_credit_schedule(setting):
    saved = asdict(ResidualUpdateBudget())
    saved[setting] += 1
    with pytest.raises(ValueError, match=setting):
        ResidualUpdateBudget().restore(saved)


def test_residual_async_recipe_preserves_algorithm_and_separates_devices(monkeypatch):
    cfg = load_recipe(monkeypatch, "maniskill_sac_residual_openpi_pi05_async")
    validate_residual_cfg(cfg)
    assert cfg.cluster.component_placement.actor == "0-0"
    assert cfg.cluster.component_placement.rollout == "1-1"
    assert cfg.cluster.component_placement.env == "1-1"
    assert cfg.actor.sync_weight_no_wait
    assert cfg.runner.weight_sync_interval == 1
    assert cfg.runner.max_critic_updates == 800000
    assert cfg.algorithm.residual_schedule.max_updates_per_iteration == 400
    assert cfg.algorithm.residual_schedule.utd == 20
    assert cfg.algorithm.critic_actor_ratio == 20
    assert not cfg.algorithm.backup_entropy
    assert cfg.actor.global_batch_size == 256
    for mode in ("train", "eval"):
        assert cfg.env[mode].init_params.id == "PegInsertionSideHardClearance-v1"
        assert cfg.env[mode].reward_mode == "only_success"
        assert not cfg.env[mode].use_rel_reward
    assert not cfg.reward.use_reward_model
    assert cfg.reward.get("env_reward_weight", 1.0) == 1.0


@pytest.mark.parametrize(
    "interval_key", ["save_critic_interval", "val_critic_interval"]
)
@pytest.mark.parametrize("critic_limit", [None, 47])
@pytest.mark.parametrize("save_interval", [None, 25])
def test_residual_async_training_yields_and_consumes_exact_transition_credits(
    tmp_path, monkeypatch, critic_limit, save_interval, interval_key
):
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
    from rlinf.scheduler import Worker
    from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType
    from rlinf.workers.actor.fsdp_sac_residual_policy_worker import (
        AsyncSACResidualFSDPPolicy,
    )

    # Exercise the real learner, replay and single-rank metric collective on CPU.
    monkeypatch.setattr(Worker, "torch_platform", torch.cpu)
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "gloo").as_uri(), rank=0, world_size=1
    )
    worker = make_worker(AsyncSACResidualFSDPPolicy)
    worker.cfg.env = {"train": {"auto_reset": False}}
    worker.cfg.runner = {
        "max_critic_updates": critic_limit,
        interval_key: save_interval,
    }
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker._timer_metrics = {}
    worker.demo_buffer = None
    worker.replay_buffer = TrajectoryReplayBuffer(sample_window_size=32)
    worker.update_budget = ResidualUpdateBudget(
        utd=2, warmup_transitions=4, warmup_updates=40, max_updates_per_iteration=20
    )
    worker.update_step = 0
    worker._recv_queue = queue.Queue()

    def batches():
        while True:
            yield worker.replay_buffer.sample_chunks(4)

    worker.buffer_dataloader_iter = batches()
    initial_editor = worker.model.editor[-1].weight.detach().clone()

    async def exercise():
        assert await worker.run_training() == {}
        assert worker.update_step == 0
        worker._recv_queue.put(make_trajectory())
        received_at = []

        async def receive_during_updates():
            await asyncio.sleep(0)
            received_at.append(worker.update_step)
            worker._recv_queue.put(make_trajectory())

        receiver = asyncio.create_task(receive_during_updates())
        first = await worker.run_training()
        await receiver
        assert 0 < received_at[0] < 20
        assert worker._recv_queue.qsize() == 1
        assert first["replay/collected_transitions"] == 4
        assert first["sac/critic_updates"] == 20
        assert math.isfinite(first["actor_q_to_entropy_grad_ratio"])
        assert first["sac/pending_updates"] == 20
        assert not torch.equal(worker.model.editor[-1].weight, initial_editor)
        second = await worker.run_training()
        assert second["replay/collected_transitions"] == 8
        second_step = save_interval or 40
        assert second["sac/critic_updates"] == second_step
        assert second["sac/pending_updates"] == 48 - second_step
        expected = critic_limit or 48
        while worker.update_step < expected:
            previous = worker.update_step
            result = await worker.run_training()
            assert result["sac/updates_this_call"] == worker.update_step - previous
            assert result["sac/pending_updates"] == 48 - worker.update_step
            assert all(math.isfinite(value) for value in result.values())
        final = await worker.run_training()
        assert final == (
            {"sac/critic_updates": expected, "sac/updates_this_call": 0}
            if critic_limit
            else {}
        )
        assert worker.update_step == worker.update_budget.updates == expected
        assert int(next(iter(worker.qf_optimizer.state.values()))["step"]) == expected

    try:
        asyncio.run(exercise())
    finally:
        worker.replay_buffer.close()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_residual_rejects_invalid_critic_update_limits(monkeypatch, limit):
    cfg = OmegaConf.merge(
        load_recipe(monkeypatch, "maniskill_sac_residual_openpi_pi05_async"),
        {"runner": {"max_critic_updates": limit}},
    )
    with pytest.raises(ValueError, match="max_critic_updates"):
        validate_residual_cfg(cfg)


class DinoPatchBackbone(torch.nn.Module):
    """Stand-in for the external DINOv3 API with spatially identifiable tokens."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward_features(self, images):
        patches = torch.nn.functional.avg_pool2d(images[:, :1], 16)
        tokens = patches.flatten(2).transpose(1, 2).repeat(1, 1, 384)
        return {"x_norm_patchtokens": tokens * self.scale}


def make_dino_encoder():
    from rlinf.models.embodiment.mlp_policy.dinov3_encoder import (
        DINOv3Config,
        FrozenDINOv3Encoder,
    )

    return FrozenDINOv3Encoder(
        DINOv3Config(
            repo_path="/unused/dinov3",
            weights_path="/unused/weights.pth",
            image_height=32,
            image_width=32,
        ),
        backbone=DinoPatchBackbone(),
    )


def test_dino_features_preserve_batch_camera_and_spatial_order_without_prompt():
    encoder = make_dino_encoder()
    quadrants = torch.tensor(
        [[[0, 64], [128, 255]], [[32, 96], [160, 224]]], dtype=torch.uint8
    )
    image = quadrants.repeat_interleave(16, 1).repeat_interleave(16, 2)
    image = image.unsqueeze(-1).expand(-1, -1, -1, 3)
    obs = {
        "main_images": image,
        "wrist_images": image.flip(0),
        "task_descriptions": ["insert the peg", "a different prompt"],
    }
    visual = encoder(obs)
    assert visual.shape == (2, 3072)
    expected = ((quadrants.float() / 255.0 - 0.485) / 0.229).flatten(1)
    expected = expected[:, None, :].expand(-1, 384, -1).reshape(2, -1)
    torch.testing.assert_close(visual, torch.cat([expected, expected.flip(0)], 1))
    obs["task_descriptions"] = ["ignore the images"] * 2
    torch.testing.assert_close(encoder(obs), visual, atol=0, rtol=0)
    obs["main_images"], obs["wrist_images"] = obs["wrist_images"], obs["main_images"]
    swapped = encoder(obs)
    torch.testing.assert_close(swapped[:, :1536], visual[:, 1536:])
    torch.testing.assert_close(swapped[:, 1536:], visual[:, :1536])


def test_dino_byte_and_float_rgb_agree_and_features_train_only_sac_heads():
    encoder = make_dino_encoder()
    encoder.train()
    assert not encoder.training and not encoder.backbone.training
    images = torch.ones(2, 32, 32, 3, dtype=torch.uint8)
    obs = {"main_images": images, "wrist_images": images * 255}
    visual = encoder(obs)
    float_obs = {key: value.float() / 255 for key, value in obs.items()}
    torch.testing.assert_close(encoder(float_obs), visual)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(encoder(obs), visual, atol=0, rtol=0)
    assert visual.dtype == torch.float32 and not visual.requires_grad
    policy = SACResidualPolicy(
        SACResidualConfig(
            state_dim=3,
            visual_dim=3072,
            state_proj_dim=4,
            visual_proj_dim=6,
            action_dim=2,
            num_action_chunks=3,
            hidden_size=8,
            depth=2,
            num_q_heads=4,
        )
    )
    policy_obs = {
        "state": torch.randn(2, 3),
        "visual": visual,
        "base_action": torch.randn(2, 6),
    }
    policy.sac_q_forward(
        policy_obs, policy_obs["base_action"]
    ).square().mean().backward()
    assert policy.encoder["visual"][0].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    policy.zero_grad(set_to_none=True)
    loss, _, _ = residual_actor_loss(policy, policy_obs, torch.tensor(1e-5))
    loss.backward()
    assert all(p.grad is None for p in policy.encoder.parameters())
    assert policy.editor[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "bad,match",
    [
        (None, "RGB camera"),
        (torch.zeros(2, 3, 32, 32), "shaped"),
        (torch.zeros(1, 32, 32, 3), "batch sizes"),
        (torch.full((2, 32, 32, 3), float("nan")), "finite RGB"),
        (torch.full((2, 32, 32, 3), -1.0), "finite RGB"),
    ],
)
def test_dino_rejects_missing_or_invalid_camera(bad, match):
    encoder = make_dino_encoder()
    obs = {"main_images": torch.zeros(2, 32, 32, 3), "wrist_images": bad}
    with pytest.raises(ValueError, match=match):
        encoder(obs)


def test_dino_recipe_preserves_sac_settings_and_validates_feature_width(monkeypatch):
    from rlinf.models.embodiment.mlp_policy.dinov3_encoder import DINOv3Config

    cfg = load_recipe(monkeypatch, "maniskill_sac_residual_openpi_pi05_dinov3_async")
    baseline = load_recipe(monkeypatch, "maniskill_sac_residual_openpi_pi05_async")
    validate_residual_cfg(cfg)
    vision = DINOv3Config(**dict(cfg.rollout.residual_dinov3))
    assert vision.output_dim == cfg.actor.model.visual_dim == 3072
    assert list(vision.camera_keys) == ["main_images", "wrist_images"]
    assert (vision.image_height, vision.image_width) == (224, 320)
    assert (vision.spatial_pool_height, vision.spatial_pool_width) == (2, 2)
    assert cfg.algorithm == baseline.algorithm
    assert cfg.env == baseline.env
    assert cfg.rollout.residual_base_model == baseline.rollout.residual_base_model
    assert cfg.actor.model.edit_scale == 0.02
    assert cfg.algorithm.entropy_tuning.initial_alpha == 1e-5
    assert cfg.actor.global_batch_size == 256
    assert cfg.runner.max_critic_updates == 800000
    cfg.actor.model.visual_dim = 2048
    with pytest.raises(ValueError, match="visual_dim=3072"):
        validate_residual_cfg(cfg)
    cfg.actor.model.visual_dim = 3072
    cfg.rollout.residual_dinov3.camera_keys = ["main_images", "main_images"]
    with pytest.raises(ValueError, match="unique keys"):
        validate_residual_cfg(cfg)


@pytest.fixture
def eef_stats(tmp_path):
    import json

    path = tmp_path / "norm_stats.json"
    stats = {
        key: {
            "q01": [-2.0 - i / 10 for i in range(10)],
            "q99": [1.0 + i / 10 for i in range(10)],
        }
        for key in ("state", "actions")
    }
    path.write_text(json.dumps({"norm_stats": stats}))
    return path


def test_rot6d_canonicalization_invariance_and_degenerate_frames():
    from rlinf.utils.torch_rot6d import (
        canonicalize_rot6d,
        quaternion_wxyz_to_rot6d,
        rot6d_to_matrix,
    )

    rotation = canonicalize_rot6d(torch.randn(20, 6))
    equivalent = torch.cat(
        (rotation[:, :3] * 3, rotation[:, 3:] * 2 + rotation[:, :3] * 4), -1
    )
    torch.testing.assert_close(canonicalize_rot6d(equivalent), rotation)
    values = torch.cat((rotation, torch.zeros(2, 6), torch.ones(2, 6)))
    matrix = rot6d_to_matrix(values)
    torch.testing.assert_close(
        matrix.transpose(-1, -2) @ matrix,
        torch.eye(3).expand(24, -1, -1),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(24))
    near_pi = torch.tensor([[1e-4, 1.0, 0.0, 0.0], [-1e-4, 1.0, 0.0, 0.0]])
    continuous = quaternion_wxyz_to_rot6d(near_pi)
    assert (continuous[0] - continuous[1]).norm() < 0.001
    torch.testing.assert_close(quaternion_wxyz_to_rot6d(-near_pi), continuous)


def test_eef_rot6d_actor_and_critic_receive_canonical_physical_rotations(eef_stats):
    import json

    from rlinf.utils.torch_rot6d import canonicalize_rot6d

    stats = json.loads(eef_stats.read_text())["norm_stats"]
    low = torch.tensor(stats["state"]["q01"])
    high = torch.tensor(stats["state"]["q99"])

    def normalize(value):
        return (value - low) / (high - low + 1e-6) * 2 - 1

    policy = SACResidualPolicy(
        SACResidualConfig(
            state_dim=10,
            action_dim=10,
            visual_dim=5,
            hidden_size=16,
            num_q_heads=2,
            num_action_chunks=2,
            eef_rot6d_norm_stats_path=str(eef_stats),
        )
    )
    value = torch.randn(3, 10)
    value[:, 3:9] = canonicalize_rot6d(value[:, 3:9])
    equivalent = value.clone()
    equivalent[:, 3:6] *= 3
    equivalent[:, 6:9] = value[:, 6:9] * 2 + value[:, 3:6] * 4
    obs = {
        "state": normalize(value),
        "visual": torch.randn(3, 5),
        "base_action": normalize(value).repeat(1, 2),
    }
    other = dict(
        obs, state=normalize(equivalent), base_action=normalize(equivalent).repeat(1, 2)
    )
    actor_inputs = []
    handle = policy.editor.register_forward_pre_hook(
        lambda _, args: actor_inputs.append(args[0].detach())
    )
    policy.sac_forward(obs, deterministic=True)
    policy.sac_forward(other, deterministic=True)
    handle.remove()
    torch.testing.assert_close(actor_inputs[0], actor_inputs[1], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actor_inputs[0][:, -20:].reshape(3, 2, 10)[:, :, 3:9],
        value[:, None, 3:9].expand(-1, 2, -1),
        atol=1e-6,
        rtol=1e-6,
    )
    q = policy.sac_q_forward(obs, obs["base_action"])
    q_other = policy.sac_q_forward(other, other["base_action"])
    torch.testing.assert_close(q, q_other, atol=2e-6, rtol=2e-6)
    actions = obs["base_action"].clone().requires_grad_()
    policy.sac_q_forward(obs, actions).sum().backward()
    assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "mismatch",
    [
        None,
        "train",
        "eval",
        "state",
        "action",
        "base",
        "chunks",
        "stats",
        "stat_dims",
        "train_cpu",
        "eval_cpu",
    ],
)
def test_eef_rot6d_recipe_rejects_mixed_representations(
    monkeypatch, eef_stats, tmp_path, mismatch
):
    import json

    cfg = load_recipe(
        monkeypatch, "maniskill_sac_residual_openpi_pi05_dinov3_eef_async"
    )
    cfg.actor.model.eef_rot6d_norm_stats_path = str(eef_stats)
    cfg.rollout.residual_base_model.openpi_data.norm_stats_path = str(eef_stats)
    if mismatch is None:
        validate_residual_cfg(cfg)
        assert cfg.actor.model.action_dim == cfg.actor.model.state_dim == 10
        return
    if mismatch in ("train", "eval"):
        cfg.env[mismatch].wrap_obs_mode = "rlt_openpi_joint"
        cfg.env[mismatch].init_params.control_mode = "pd_joint_delta_pos"
    elif mismatch in ("train_cpu", "eval_cpu"):
        cfg.env[mismatch.removesuffix("_cpu")].init_params.sim_backend = "physx_cpu"
    elif mismatch == "state":
        cfg.actor.model.state_dim = 9
    elif mismatch == "action":
        cfg.actor.model.action_dim = 8
    elif mismatch == "base":
        cfg.rollout.residual_base_model.openpi.action_env_dim = 7
    elif mismatch == "chunks":
        cfg.rollout.residual_base_model.openpi.action_chunk = 5
    elif mismatch == "stats":
        cfg.actor.model.eef_rot6d_norm_stats_path = str(tmp_path / "other.json")
    elif mismatch == "stat_dims":
        stats = json.loads(eef_stats.read_text())
        stats["norm_stats"]["actions"]["q01"] = [0] * 7
        eef_stats.write_text(json.dumps(stats))
    with pytest.raises(ValueError):
        validate_residual_cfg(cfg)


def make_privileged_policy(actor_use_privileged_state=False):
    from rlinf.data.schema.peg_privileged import PEG_PRIVILEGED_DIM

    cfg = asdict(make_policy().config)
    cfg.update(
        privileged_state_dim=PEG_PRIVILEGED_DIM,
        privileged_proj_dim=8,
        actor_use_privileged_state=actor_use_privileged_state,
    )
    return SACResidualPolicy(SACResidualConfig(**cfg))


def test_privileged_critic_changes_q_but_not_actor_and_requires_replay_state():
    from rlinf.data.schema.peg_privileged import PEG_PRIVILEGED_DIM

    torch.manual_seed(107)
    policy = make_privileged_policy()
    # A nonzero editor verifies isolation after actor training as well as at init.
    torch.nn.init.normal_(policy.editor[-1].weight, std=0.1)
    obs = make_obs()
    baseline, _, _ = policy.sac_forward(obs, deterministic=True)
    obs["privileged_state"] = torch.randn(4, PEG_PRIVILEGED_DIM, requires_grad=True)
    actions, _, _ = policy.sac_forward(obs, deterministic=True)
    torch.testing.assert_close(actions, baseline, atol=0, rtol=0)
    q = policy.sac_q_forward(obs, actions)
    changed = dict(obs, privileged_state=obs["privileged_state"] + 0.5)
    assert not torch.allclose(q, policy.sac_q_forward(changed, actions))
    actor_loss, _, metrics = residual_actor_loss(
        policy,
        obs,
        torch.tensor(1e-5),
        actor_parameters=list(policy.editor.parameters()),
    )
    actor_loss.backward()
    assert math.isfinite(metrics["actor_q_to_entropy_grad_ratio"])
    assert obs["privileged_state"].grad is None
    assert all(p.grad is None for p in policy.privileged_encoder.parameters())
    assert all(p.grad is None for p in policy.encoder.parameters())
    assert policy.editor[-1].weight.grad.norm() > 0
    policy.zero_grad(set_to_none=True)
    policy.sac_q_forward(obs, obs["base_action"]).square().mean().backward()
    assert policy.privileged_encoder[0].weight.grad.norm() > 0
    assert policy.encoder["visual"][0].weight.grad.norm() > 0
    assert all(p.grad is None for p in policy.editor.parameters())
    with pytest.raises(ValueError, match="requires privileged_state"):
        policy.sac_q_forward(make_obs(), actions)


@pytest.mark.parametrize("actor_privileged", [False, True])
def test_privileged_critic_canonicalizes_every_pose_rotation(actor_privileged):
    from rlinf.data.schema.peg_privileged import (
        PEG_PRIVILEGED_DIM,
        PEG_PRIVILEGED_ROT6D_STARTS,
        canonicalize_peg_privileged_state,
    )

    policy = make_privileged_policy(actor_privileged)
    torch.nn.init.normal_(policy.editor[-1].weight, std=0.1)
    original = canonicalize_peg_privileged_state(torch.randn(4, PEG_PRIVILEGED_DIM))
    equivalent = original.clone()
    for start in PEG_PRIVILEGED_ROT6D_STARTS:
        first, second = original[:, start : start + 6].chunk(2, -1)
        equivalent[:, start : start + 6] = torch.cat(
            (first * 3, second * 2 + first * 4), -1
        )
    obs = dict(make_obs(), privileged_state=original)
    changed = dict(obs, privileged_state=equivalent)
    torch.testing.assert_close(
        policy.sac_q_forward(obs, obs["base_action"]),
        policy.sac_q_forward(changed, obs["base_action"]),
    )
    torch.testing.assert_close(
        policy.sac_forward(obs, deterministic=True)[0],
        policy.sac_forward(changed, deterministic=True)[0],
    )
    with pytest.raises(ValueError, match="privileged state"):
        canonicalize_peg_privileged_state(torch.zeros(4, PEG_PRIVILEGED_DIM - 1))


@pytest.mark.parametrize("actor_privileged", [False, True])
def test_privileged_target_uses_next_state_and_bootstraps_timeout(actor_privileged):
    from rlinf.data.schema.peg_privileged import PEG_PRIVILEGED_DIM

    torch.manual_seed(127)
    policy = make_privileged_policy(actor_privileged)
    target = copy.deepcopy(policy).requires_grad_(False)
    obs = dict(make_obs(), privileged_state=torch.randn(4, PEG_PRIVILEGED_DIM))
    nxt = dict(make_obs(), privileged_state=torch.randn(4, PEG_PRIVILEGED_DIM))
    batch = {
        "curr_obs": obs,
        "next_obs": nxt,
        "actions": obs["base_action"],
        "rewards": torch.zeros(4, 1),
        "terminations": torch.zeros(4, 1, dtype=torch.bool),
        "truncations": torch.ones(4, 1, dtype=torch.bool),
    }
    torch.manual_seed(131)
    _, metrics = residual_critic_loss(
        policy,
        target,
        batch,
        torch.tensor(1e-5),
        0.99,
        False,
        torch.Generator().manual_seed(137),
    )
    torch.manual_seed(131)
    with torch.no_grad():
        actions, _, _ = policy.sac_forward(nxt)
        expected = 0.99 * random_min_q(
            target.sac_q_forward(nxt, actions), torch.Generator().manual_seed(137)
        )
    assert metrics["target_q"] == pytest.approx(expected.mean().item())
    nxt.pop("privileged_state")
    with pytest.raises(ValueError, match="requires privileged_state"):
        residual_critic_loss(policy, target, batch, torch.tensor(1e-5), 0.99, False)


def test_privileged_recipe_matches_state_contract_and_random_geometry(
    monkeypatch, eef_stats
):
    from rlinf.data.schema.peg_privileged import PEG_PRIVILEGED_DIM

    cfg = load_recipe(
        monkeypatch, "maniskill_sac_residual_openpi_pi05_dinov3_eef_privileged_async"
    )
    cfg.actor.model.eef_rot6d_norm_stats_path = str(eef_stats)
    cfg.rollout.residual_base_model.openpi_data.norm_stats_path = str(eef_stats)
    validate_residual_cfg(cfg)
    assert cfg.actor.model.privileged_state_dim == PEG_PRIVILEGED_DIM
    assert cfg.actor.model.actor_use_privileged_state
    for split in ("train", "eval"):
        assert cfg.env[split].critic_privileged_obs
        assert cfg.env[split].reward_mode == "peg_staged_dense"
        assert not cfg.env[split].use_rel_reward
        assert len(cfg.env[split].peg_dense_reward.grasp_min_peg_m) == 3
        assert len(cfg.env[split].peg_dense_reward.grasp_max_peg_m) == 3
        assert cfg.env[split].peg_dense_reward.alignment_tolerance_m == 0.002
        assert cfg.env[split].init_params.get("fixed_peg_half_length") is None
        assert cfg.env[split].init_params.get("fixed_peg_half_width") is None
    assert not cfg.algorithm.backup_entropy
    assert cfg.algorithm.entropy_tuning.initial_alpha == 1e-5
    assert cfg.actor.model.edit_scale == 0.02
    cfg.env.eval.critic_privileged_obs = False
    with pytest.raises(ValueError, match="critic_privileged_obs"):
        validate_residual_cfg(cfg)


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires ManiSkill, GPU physics and GPU IK; set RLINF_TEST_MANISKILL_GPU=1.",
)
@pytest.mark.parametrize("fixed_geometry", [False, True])
def test_privileged_maniskill_observation_geometry_and_reset(
    monkeypatch, fixed_geometry
):
    from rlinf.data.schema.peg_privileged import (
        PEG_PRIVILEGED_DIM,
    )
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from rlinf.envs.sim.maniskill.peg_privileged import extract_peg_privileged_state

    cfg = load_recipe(
        monkeypatch, "maniskill_sac_residual_openpi_pi05_dinov3_eef_privileged_async"
    ).env.train
    if fixed_geometry:
        OmegaConf.update(
            cfg, "init_params.fixed_peg_half_length", 0.105, force_add=True
        )
        OmegaConf.update(cfg, "init_params.fixed_peg_half_width", 0.02, force_add=True)
    cfg.total_num_envs = 4
    cfg.max_episode_steps = 2
    cfg.init_params.max_episode_steps = 2
    env = ManiskillRLTEnv(cfg, 4, 0, 1, None)
    try:
        for first_seed in (1100, 2200):
            obs, _ = env.reset(
                seed=list(range(first_seed, first_seed + 4)),
                options={"reconfigure": True},
            )
            task = env.env.unwrapped
            state = obs["privileged_state"]
            assert (
                state.shape == (4, PEG_PRIVILEGED_DIM) and torch.isfinite(state).all()
            )
            if fixed_geometry:
                torch.testing.assert_close(
                    task.peg_half_sizes,
                    state.new_tensor([[0.105, 0.02, 0.02]]).expand(4, -1),
                )
            else:
                assert task.peg_half_sizes[:, 0].std() > 0.001
                assert task.peg_half_sizes[:, 1].std() > 0.0001
                assert (
                    (task.peg_half_sizes[:, 0] >= 0.085)
                    & (task.peg_half_sizes[:, 0] <= 0.125)
                ).all()
                assert (
                    (task.peg_half_sizes[:, 1:] >= 0.015)
                    & (task.peg_half_sizes[:, 1:] <= 0.025)
                ).all()
            torch.testing.assert_close(
                task.box_hole_radii, task.peg_half_sizes[:, 1] + 0.01
            )
            assert task.box.pose.p.std(0).max() > 0.001
            # Place the head exactly at the depth threshold with millimetre offsets.
            from mani_skill.utils.structs.pose import Pose

            original_peg = Pose.create(task.peg.pose.raw_pose.clone())
            head_offset = state.new_tensor([[-0.015, 0.002, -0.003]]).expand(4, -1)
            task.peg.set_pose(
                task.box_hole_pose
                * Pose.create_from_pq(p=head_offset)
                * task.peg_head_offsets.inv()
            )
            aligned = extract_peg_privileged_state(task)
            torch.testing.assert_close(
                aligned[:, :3],
                state.new_tensor([[-15.0, 2.0, -3.0]]).expand(4, -1),
                atol=1e-3,
                rtol=0,
            )
            torch.testing.assert_close(
                aligned[:, 3:9],
                state.new_tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).expand(4, -1),
                atol=1e-5,
                rtol=0,
            )
            expected_margins = torch.stack(
                (
                    torch.zeros_like(task.box_hole_radii),
                    task.box_hole_radii * 1000 - 2,
                    task.box_hole_radii * 1000 - 3,
                ),
                -1,
            )
            torch.testing.assert_close(
                aligned[:, 9:12], expected_margins, atol=1e-3, rtol=0
            )
            task.peg.set_pose(original_peg)
            saved = state.clone()
            action = state.new_zeros((4, 7))
            action[:, 0] = 0.2
            action[:, -1] = 1
            next_obs, _, _, _, _ = env.step(action, auto_reset=False)
            torch.testing.assert_close(state, saved)
            torch.testing.assert_close(
                next_obs["privileged_state"], extract_peg_privileged_state(task)
            )
            assert not torch.equal(state, next_obs["privileged_state"])
            observations, _, _, truncations, _ = env.chunk_step(
                action[:, None].expand(-1, 4, -1)
            )
            assert truncations[:, 0].all() and not truncations[:, 1:].any()
            for observation in observations[1:]:
                torch.testing.assert_close(
                    observation["privileged_state"], observations[0]["privileged_state"]
                )
            frozen, _, _, _, _ = env.chunk_step(action[:, None].expand(-1, 4, -1))
            torch.testing.assert_close(
                frozen[-1]["privileged_state"], observations[0]["privileged_state"]
            )
    finally:
        env.env.close()


def test_privileged_actor_uses_state_and_keeps_projection_critic_owned():
    from rlinf.data.schema.peg_privileged import PEG_PRIVILEGED_DIM

    torch.manual_seed(211)
    policy = make_privileged_policy(True)
    obs = dict(make_obs(), privileged_state=torch.randn(4, PEG_PRIVILEGED_DIM))
    # Zero initialization must still preserve the frozen base action exactly.
    torch.testing.assert_close(
        policy.sac_forward(obs, deterministic=True)[0], obs["base_action"]
    )
    torch.nn.init.normal_(policy.editor[-1].weight, std=0.1)
    changed = dict(obs, privileged_state=obs["privileged_state"] + 0.7)
    actions = policy.sac_forward(obs, deterministic=True)[0]
    assert not torch.allclose(
        actions, policy.sac_forward(changed, deterministic=True)[0]
    )
    assert torch.all((actions - obs["base_action"]).abs() <= policy.edit_scale + 1e-6)
    loss, _, metrics = residual_actor_loss(
        policy,
        obs,
        torch.tensor(1e-5),
        actor_parameters=list(policy.editor.parameters()),
    )
    loss.backward()
    assert math.isfinite(metrics["actor_q_to_entropy_grad_ratio"])
    assert policy.editor[0].weight.grad.norm() > 0
    assert all(p.grad is None for p in policy.privileged_encoder.parameters())
    assert all(p.grad is None for p in policy.encoder.parameters())
    with pytest.raises(ValueError, match="requires privileged_state"):
        policy.sac_forward(make_obs())
    with pytest.raises(ValueError, match="requires privileged_state"):
        policy.predict_action_batch(make_obs(), mode="eval")
    restored = make_privileged_policy(True)
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(
        actions, restored.sac_forward(obs, deterministic=True)[0]
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"actor_use_privileged_state": True},
        {"actor_use_privileged_state": "true", "privileged_state_dim": 31},
    ],
)
def test_privileged_actor_rejects_invalid_contract(settings):
    with pytest.raises(ValueError, match="actor_use_privileged_state"):
        SACResidualConfig(**settings).validate()


@pytest.mark.parametrize("width", [30, 32, 415])
def test_compact_privileged_rejects_incompatible_layout(width):
    with pytest.raises(ValueError, match="privileged_state_dim"):
        SACResidualConfig(privileged_state_dim=width).validate()


def test_peg_dense_reward_costs_unlock_without_penalizing_progress():
    from rlinf.envs.sim.maniskill.peg_dense_reward import (
        PegDenseRewardConfig,
        compute_peg_dense_reward,
    )

    cfg = PegDenseRewardConfig(
        grasp_min_peg_m=(-0.01, -0.01, -0.01), grasp_max_peg_m=(0.01, 0.01, 0.01)
    )
    tcp = torch.tensor([[0.06, 0.0, 0.0]]).repeat(4, 1)
    head = torch.tensor(
        [
            [-0.115, 0.05, 0.0],
            [-0.115, 0.05, 0.0],
            [-0.115, 0.0, 0.0],
            [-0.015, 0.0, 0.0],
        ]
    )
    axis = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
    grasp = torch.tensor([False, True, True, True])
    success = torch.tensor([False, False, False, True])
    reward, info = compute_peg_dense_reward(
        tcp, head, axis, torch.full((4,), 0.1), grasp, success, cfg
    )
    torch.testing.assert_close(reward, torch.tensor([-2.5, -2.0, -1.0, 0.0]))
    assert info["dense_stage"].tolist() == [0, 1, 2, 3]
    assert info["dense_grasped"][1] and info["grasp_region_error_mm"][1] > 0
    assert (reward.diff() > 0).all()
    torch.testing.assert_close(
        reward,
        -(
            info["dense_grasp_cost"]
            + info["dense_alignment_cost"]
            + info["dense_depth_cost"]
        ),
    )
    dropped, _ = compute_peg_dense_reward(
        tcp,
        head,
        axis,
        torch.full((4,), 0.1),
        torch.zeros_like(grasp),
        torch.zeros_like(success),
        cfg,
    )
    assert (dropped <= reward).all()


def test_peg_dense_reward_improves_each_distance_and_uses_axis_alignment():
    from rlinf.envs.sim.maniskill.peg_dense_reward import (
        PegDenseRewardConfig,
        compute_peg_dense_reward,
    )

    cfg = PegDenseRewardConfig(
        grasp_min_peg_m=(-0.01,) * 3, grasp_max_peg_m=(0.01,) * 3
    )
    no = torch.zeros(3, dtype=torch.bool)
    axis = torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1)
    half = torch.full((3,), 0.1)
    tcp = torch.tensor([[0.2, 0.0, 0.0], [0.1, 0.0, 0.0], [0.02, 0.0, 0.0]])
    head = torch.tensor([[-0.2, 0.1, 0.0]]).repeat(3, 1)
    reach, _ = compute_peg_dense_reward(tcp, head, axis, half, no, no, cfg)
    assert (reach.diff() > 0).all()
    head[:, 1] = torch.tensor([0.1, 0.05, 0.01])
    alignment, _ = compute_peg_dense_reward(
        torch.zeros_like(tcp), head, axis, half, ~no, no, cfg
    )
    assert (alignment.diff() > 0).all()
    head[:, 1] = 0
    head[:, 0] = torch.tensor([-0.215, -0.115, -0.025])
    depth, _ = compute_peg_dense_reward(
        torch.zeros_like(tcp), head, axis, half, ~no, no, cfg
    )
    assert (depth.diff() > 0).all()
    reversed_axis, info = compute_peg_dense_reward(
        torch.zeros_like(tcp), head, -axis, half, ~no, no, cfg
    )
    assert not info["dense_aligned"].any()
    assert (reversed_axis < depth).all()
    far, _ = compute_peg_dense_reward(tcp * 1e4, head, axis, half, no, no, cfg)
    assert ((far >= -3) & (far <= 0)).all()


def test_peg_dense_reward_unlock_is_nonnegative_and_stalls_never_pay():
    from rlinf.envs.sim.maniskill.peg_dense_reward import (
        PegDenseRewardConfig,
        compute_peg_dense_reward,
    )

    cfg = PegDenseRewardConfig(
        grasp_min_peg_m=(-0.01,) * 3, grasp_max_peg_m=(0.01,) * 3
    )
    rng = torch.Generator().manual_seed(97)
    tcp = torch.randn(128, 3, generator=rng)
    head = torch.randn(128, 3, generator=rng)
    axis = torch.nn.functional.normalize(torch.randn(128, 3, generator=rng), dim=-1)
    no = torch.zeros(128, dtype=torch.bool)
    args = (tcp, head, axis, torch.full((128,), 0.1))
    before, _ = compute_peg_dense_reward(*args, no, no, cfg)
    after, _ = compute_peg_dense_reward(*args, ~no, no, cfg)
    assert (after >= before).all()
    assert (after <= 0).all()
    rewards = after[:, None].expand(-1, 10)
    replay = EnvTransition(rewards=rewards).compute_rewards(
        bootstrap_values=None,
        env_reward_weight=20,
        reward_weight=1,
        auto_reset=False,
        bootstrap_type="standard",
        gamma=0.99,
    )
    torch.testing.assert_close(replay, rewards)
    assert (replay.sum(1) < 0).all()


def test_peg_grasp_region_has_flat_interior_without_blocking_grasp():
    from rlinf.envs.sim.maniskill.peg_dense_reward import (
        PegDenseRewardConfig,
        compute_peg_dense_reward,
    )

    cfg = PegDenseRewardConfig(
        grasp_min_peg_m=(-0.075, -0.005, -0.005), grasp_max_peg_m=(-0.05, 0.005, 0.005)
    )
    tcp = torch.tensor(
        [
            [-0.075, -0.005, 0.0],
            [-0.05, 0.005, 0.0],
            [-0.06, 0.0, 0.0],
            [-0.049, 0.0, 0.0],
        ]
    )
    reward, info = compute_peg_dense_reward(
        tcp,
        torch.tensor([[-0.2, 0.02, 0.0]]).repeat(4, 1),
        torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1),
        torch.full((4,), 0.1),
        torch.ones(4, dtype=torch.bool),
        torch.zeros(4, dtype=torch.bool),
        cfg,
    )
    torch.testing.assert_close(reward[:3], reward[0].expand(3))
    torch.testing.assert_close(
        info["grasp_region_error_mm"],
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert info["dense_stage"].tolist() == [1, 1, 1, 1]
    assert reward[3] < reward[0]


@pytest.mark.parametrize(
    "settings",
    [
        {"grasp_min_peg_m": (0, 0)},
        {"grasp_min_peg_m": (0, float("nan"), 0)},
        {"grasp_max_peg_m": (-0.01, 0.01, 0.01)},
        {"reach_scale_m": 0},
        {"alignment_scale_m": float("nan")},
        {"depth_scale_m": -1},
        {"alignment_tolerance_m": 0},
        {"axis_tolerance_deg": 90},
    ],
)
def test_peg_dense_reward_rejects_invalid_settings(settings):
    from rlinf.envs.sim.maniskill.peg_dense_reward import PegDenseRewardConfig

    with pytest.raises(ValueError):
        PegDenseRewardConfig(
            **dict(
                {"grasp_min_peg_m": (-0.01,) * 3, "grasp_max_peg_m": (0.01,) * 3},
                **settings,
            )
        )


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires ManiSkill GPU physics and GPU IK.",
)
@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("time_limit", [5, 6])
def test_peg_dense_reward_gpu_chunk_timeout_and_reset(
    monkeypatch, auto_reset, time_limit
):
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv

    cfg = load_recipe(
        monkeypatch, "maniskill_sac_residual_openpi_pi05_dinov3_eef_privileged_async"
    ).env.train
    cfg.max_episode_steps = time_limit
    cfg.init_params.max_episode_steps = time_limit
    cfg.auto_reset = auto_reset
    cfg.total_num_envs = 2
    env = ManiskillRLTEnv(cfg, 2, 0, 1, None)
    try:
        env.reset(seed=1100)
        actions = torch.zeros(2, 3, 7, device=env.device)
        _, rewards, terminations, truncations, infos = env.chunk_step(actions)
        for j in range(3):
            torch.testing.assert_close(rewards[:, j], infos[j]["dense_reward"])
        assert (rewards < 0).all()
        first_return = rewards.sum(1)
        torch.testing.assert_close(infos[-1]["episode"]["return"], first_return)
        assert not (terminations | truncations).any()
        obs, rewards, terminations, truncations, infos = env.chunk_step(actions)
        final = infos[-1]["final_info"] if auto_reset else infos[-1]
        assert (rewards[:, : time_limit - 3] < 0).all()
        assert (rewards[:, time_limit - 3 :] == 0).all()
        assert truncations[:, time_limit - 4].all() and truncations.sum().item() == 2
        assert not terminations.any()
        torch.testing.assert_close(
            final["episode"]["return"], first_return + rewards.sum(1)
        )
        if auto_reset:
            assert infos[-1]["_final_observation"].all()
            assert not torch.equal(
                obs[-1]["states"], infos[-1]["final_observation"]["states"]
            )
            _, rewards, _, _, infos = env.chunk_step(actions)
            torch.testing.assert_close(infos[-1]["episode"]["return"], rewards.sum(1))
            return
        _, frozen, _, _, _ = env.chunk_step(actions)
        assert (frozen == 0).all()
        env.reset(options={"env_idx": torch.tensor([0], device=env.device)})
        _, rewards, _, _, infos = env.chunk_step(actions)
        assert (rewards[1] == 0).all()
        torch.testing.assert_close(infos[-1]["episode"]["return"][0], rewards[0].sum())
        env.reset(seed=2200)
        _, reward, _, _, info = env.step(actions[:, 0], auto_reset=False)
        torch.testing.assert_close(reward, info["dense_reward"])
        assert (reward < 0).all()
    finally:
        env.env.close()


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("algorithm.bootstrap_time_limits", False, "bootstrap_time_limits"),
        ("env.eval.ignore_terminations", True, "ignore_terminations"),
    ],
)
def test_peg_dense_recipe_rejects_inconsistent_returns(
    monkeypatch, eef_stats, path, value, match
):
    cfg = load_recipe(
        monkeypatch, "maniskill_sac_residual_openpi_pi05_dinov3_eef_privileged_async"
    )
    cfg.actor.model.eef_rot6d_norm_stats_path = str(eef_stats)
    cfg.rollout.residual_base_model.openpi_data.norm_stats_path = str(eef_stats)
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError, match=match):
        validate_residual_cfg(cfg)


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires GPU physics and GPU EEF IK.",
)
def test_fixed_peg_collection_matches_reference_and_replay_geometry():
    from toolkits.lerobot.collect_maniskill_peg_lerobot_eef import ENV_ID, initial_scene
    from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (
        _build_env,
        build_parser,
    )

    args = build_parser().parse_args(
        [
            "--sim-backend",
            "gpu",
            "--fixed-peg-half-length",
            "0.105",
            "--fixed-peg-half-width",
            "0.02",
        ]
    )
    reference_scenes = []
    for mode in ("pd_joint_pos", "pd_ee_delta_pose"):
        env = _build_env(args, control_mode=mode, env_id=ENV_ID)
        try:
            for index, seed in enumerate((650000, 650001)):
                env.reset(seed=seed, options={"reconfigure": True})
                scene = initial_scene(env)
                torch.testing.assert_close(
                    torch.as_tensor(scene["peg_half_sizes"]),
                    torch.tensor([0.105, 0.02, 0.02]),
                )
                torch.testing.assert_close(
                    torch.as_tensor(scene["hole_radius"]), torch.tensor(0.03)
                )
                assert not scene["hole_offset"][:3].any()
                if mode == "pd_joint_pos":
                    reference_scenes.append(scene)
                else:
                    for key, value in scene.items():
                        torch.testing.assert_close(
                            torch.as_tensor(value),
                            torch.as_tensor(reference_scenes[index][key]),
                        )
            assert not (reference_scenes[0]["box"] == reference_scenes[1]["box"]).all()
            assert not (reference_scenes[0]["peg"] == reference_scenes[1]["peg"]).all()
        finally:
            env.close()


def test_physical_oracle_frames_gates_and_gripper():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from toolkits.standalone_eval_scripts.maniskill_oracle_residual import (
        OracleConfig,
        apply_physical_correction,
        controller_physical,
        oracle_correction,
    )

    identity = np.repeat(np.eye(3)[None], 3, axis=0)
    config = OracleConfig((0, 0, 0), (0.001,) * 3, (0.002,) * 3)
    peg = identity.copy()
    peg[2] = Rotation.from_euler("z", 0.2).as_matrix()
    tcp = np.array([[0.1, 0, 0], [0.01, 0, 0], [0, 0, 0]])
    head = np.array([[0, 0, 0], [0, 0, 0], [-0.1, 0.02, -0.03]])
    grasp = np.array([False, False, True])
    position, rotation, stage = oracle_correction(
        tcp, head, peg, identity, identity, grasp, config
    )
    np.testing.assert_array_equal(stage, [0, 1, 2])
    np.testing.assert_allclose(
        position, [[0, 0, 0], [-0.001, 0, 0], [0, -0.001, 0.001]]
    )
    np.testing.assert_allclose(rotation, [[0, 0, 0], [0, 0, 0], [0, 0, -0.002]])
    base = np.zeros((3, 7), dtype=np.float32)
    base[:, -1] = [-1, 0.5, 1]
    action, diagnostics = apply_physical_correction(
        base, position, rotation, config.rotation_limit
    )
    np.testing.assert_array_equal(action[:, -1], base[:, -1])
    np.testing.assert_array_equal(action[0], base[0])
    assert action[2, 5] > 0  # Installed controller uses a negative rotation scale.
    p, r = controller_physical(action)
    np.testing.assert_allclose(p, position, atol=1e-9)
    np.testing.assert_allclose(Rotation.from_matrix(r).as_rotvec(), rotation, atol=1e-9)
    assert np.max(abs(diagnostics["effective_rotation"])) <= 0.002 + 1e-9
    root = np.repeat(Rotation.from_euler("z", np.pi / 2).as_matrix()[None], 3, axis=0)
    p_root, _, _ = oracle_correction(tcp, head, peg, identity, root, grasp, config)
    np.testing.assert_allclose(p_root[1], [0, 0.001, 0], atol=1e-12)


def test_physical_oracle_saturated_base_and_zero_identity():
    import numpy as np

    from toolkits.standalone_eval_scripts.maniskill_oracle_residual import (
        apply_physical_correction,
    )

    base = np.random.default_rng(5).normal(size=(128, 7)).astype(np.float32)
    zero = np.zeros((128, 3))
    action, _ = apply_physical_correction(base, zero, zero, (0.001,) * 3)
    np.testing.assert_array_equal(action, base)
    action, stats = apply_physical_correction(
        base, np.full_like(zero, 0.0003), np.full_like(zero, 0.001), (0.001,) * 3
    )
    np.testing.assert_array_equal(action[:, -1], base[:, -1])
    assert np.max(abs(stats["effective_position"])) <= 0.0003 + 2e-8
    assert np.max(abs(stats["effective_rotation"])) <= 0.001 + 2e-9


def test_rot6d_oracle_feasibility_certificate():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from toolkits.standalone_eval_scripts.maniskill_oracle_residual import (
        rot6d_box_feasible,
    )

    identity = np.eye(3)[None]
    raw = np.array([[1.0, 0, 0, 0, 1, 0]])
    assert rot6d_box_feasible(raw, np.full(6, 0.001), identity).all()
    rotated = Rotation.from_euler("z", 0.1).as_matrix()[None]
    assert not rot6d_box_feasible(raw, np.full(6, 0.001), rotated).any()
    raw = np.array([[2.0, 0, 0, 0.5, 3, 0]])
    assert rot6d_box_feasible(raw, np.full(6, 0.0001), identity).all()


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires ManiSkill GPU physics and GPU IK.",
)
@pytest.mark.parametrize("chunk_auto_reset", [False, True])
def test_peg_contact_observations_after_reset(monkeypatch, chunk_auto_reset):
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from rlinf.envs.sim.maniskill.peg_privileged import extract_peg_privileged_state

    cfg = load_recipe(
        monkeypatch, "maniskill_sac_residual_openpi_pi05_dinov3_eef_privileged_async"
    ).env.train
    cfg.total_num_envs = 2
    cfg.max_episode_steps = 3
    cfg.init_params.max_episode_steps = 3
    cfg.auto_reset = True
    env = ManiskillRLTEnv(cfg, 2, 0, 1, None)
    try:
        env.reset(seed=1100)
        task = env.env.unwrapped
        # The vendor contact buffer may retain arbitrary previous-episode values.
        monkeypatch.setattr(
            task.scene,
            "get_pairwise_contact_forces",
            lambda *args: torch.ones((2, 3), device=env.device),
        )
        monkeypatch.setattr(
            task.agent,
            "is_grasping",
            lambda *args, **kwargs: torch.ones(2, dtype=torch.bool, device=env.device),
        )

        obs, info = env.reset(seed=2200)
        assert (env.elapsed_steps == 0).all()
        assert (obs["privileged_state"][:, 21:] == 0).all()
        assert not info["dense_grasped"].any()
        torch.testing.assert_close(
            obs["privileged_state"], extract_peg_privileged_state(task)
        )
        initial_state = obs["privileged_state"].clone()
        action = torch.zeros((2, 7), device=env.device)
        action[:, -1] = 1
        stepped, _, _, _, info = env.step(action, auto_reset=False)
        assert stepped["privileged_state"][:, -1].all()
        assert (stepped["privileged_state"][:, 21:30].abs().sum(-1) > 0).all()
        assert info["dense_grasped"].all()
        torch.testing.assert_close(obs["privileged_state"], initial_state)

        obs, info = env.reset(options={"env_idx": torch.tensor([0], device=env.device)})
        assert env.elapsed_steps.tolist() == [0, 1]
        assert (obs["privileged_state"][0, 21:] == 0).all()
        assert not info["dense_grasped"][0] and info["dense_grasped"][1]
        torch.testing.assert_close(
            obs["privileged_state"][1],
            stepped["privileged_state"][1],
            atol=0.002,
            rtol=1e-5,
        )
        env.step(action, auto_reset=False)
        if chunk_auto_reset:
            observations, _, _, truncations, infos = env.chunk_step(
                action[:, None].expand(-1, 3, -1)
            )
            obs, info = observations[-1], infos[-1]
            assert truncations.any(1).all()
        else:
            obs, _, _, truncations, info = env.step(action)
            assert truncations.tolist() == [False, True]
        done = info["_final_observation"]
        assert (obs["privileged_state"][done, 21:] == 0).all()
        assert not info["dense_grasped"][done].any()
        final_state = info["final_observation"]["privileged_state"]
        assert final_state[done, -1].all()
        assert (final_state[done, 21:30].abs().sum(-1) > 0).all()
        assert info["final_info"]["dense_grasped"][done].all()
        assert (env.elapsed_steps[done] == 0).all()
        if (~done).any():
            assert obs["privileged_state"][~done, -1].all()
        obs, _, _, _, info = env.step(action, auto_reset=False)
        assert obs["privileged_state"][:, -1].all()
        assert info["dense_grasped"].all()
    finally:
        env.env.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"gap_m": 0},
        {"gap_m": -0.01},
        {"lateral_jitter_m": float("nan")},
        {"rotation_jitter_deg": 11},
    ],
)
def test_peg_preinsert_rejects_invalid_config(settings):
    from rlinf.envs.sim.maniskill.peg_preinsert import PegPreinsertConfig

    with pytest.raises(ValueError):
        PegPreinsertConfig(**settings)


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires ManiSkill GPU physics and GPU IK.",
)
@pytest.mark.parametrize("fixed_geometry", [True, False])
@pytest.mark.parametrize("hole_clearance", [None, 0.001])
def test_peg_preinsert_reset_and_grasp(fixed_geometry, hole_clearance):
    from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles

    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/env/maniskill_peg_preinsert.yaml"
    )
    if not fixed_geometry:
        del cfg.init_params.fixed_peg_half_length
        del cfg.init_params.fixed_peg_half_width
    if hole_clearance is not None:
        cfg.init_params.hole_clearance = hole_clearance
    cfg.total_num_envs = 8
    cfg.max_episode_steps = 5
    cfg.init_params.max_episode_steps = 5
    cfg.auto_reset = True
    env = ManiskillRLTEnv(cfg, 8, 0, 1, None)
    try:
        task = env.env.unwrapped
        command = torch.zeros((8, 7), device=env.device)
        command[:, -1] = -1
        box_poses = []
        for first_seed in range(320000, 320048, 8):
            seeds = list(range(first_seed, first_seed + 8))
            obs, info = env.reset(seed=seeds, options={"reconfigure": True})
            expected_clearance = 0.01 if hole_clearance is None else hole_clearance
            assert task.hole_clearance == task.success_clearance == expected_clearance
            torch.testing.assert_close(
                task.box_hole_radii,
                task.peg_half_sizes[:, 1] + expected_clearance,
            )
            head = (task.box_hole_pose.inv() * task.peg_head_pose).p
            gap = -task.peg_half_sizes[:, 0] - head[:, 0]
            torch.testing.assert_close(
                gap, torch.full_like(gap, 0.05), atol=2e-6, rtol=0
            )
            assert (head[:, 1:].abs() <= 0.030001).all()
            rotation = matrix_to_euler_angles(
                (task.box_hole_pose.inv() * task.peg.pose).to_transformation_matrix()[
                    :, :3, :3
                ],
                "XYZ",
            )
            assert (rotation.abs() <= math.radians(3) + 1e-5).all()
            tcp = (task.peg.pose.inv() * task.agent.tcp.pose).p
            expected = torch.zeros_like(tcp)
            expected[:, 0] = -0.6 * task.peg_half_sizes[:, 0]
            torch.testing.assert_close(tcp, expected, atol=1e-4, rtol=0)
            assert not info["success"].any()
            assert (obs["privileged_state"][:, 21:] == 0).all()
            assert (env.elapsed_steps == 0).all()
            initial_qpos = task.agent.robot.qpos.clone()
            initial_peg = task.peg.pose.raw_pose.clone()
            box_poses.append(task.box.pose.raw_pose.clone())
            obs, _, _, _, info = env.step(command, auto_reset=False)
            assert task.agent.is_grasping(task.peg).all()
            assert obs["privileged_state"][:, -1].all()
            assert info["dense_grasped"].all()
            unreset_state = obs["privileged_state"][[0, 2, 4, 5, 6, 7]].clone()
            obs, _ = env.reset(
                options={"env_idx": torch.tensor([1, 3], device=env.device)}
            )
            torch.testing.assert_close(
                obs["privileged_state"][[0, 2, 4, 5, 6, 7]],
                unreset_state,
                atol=0.002,
                rtol=1e-5,
            )
            assert (obs["privileged_state"][[1, 3], 21:] == 0).all()
            env.reset(seed=seeds, options={"reconfigure": True})
            torch.testing.assert_close(
                task.agent.robot.qpos, initial_qpos, atol=1e-5, rtol=0
            )
            torch.testing.assert_close(
                task.peg.pose.raw_pose, initial_peg, atol=1e-6, rtol=0
            )
        assert torch.cat(box_poses)[:, :2].std(0).min() > 0.01
        obs, _, _, truncations, infos = env.chunk_step(
            command[:, None].expand(-1, 5, -1)
        )
        assert truncations[:, -1].all()
        assert (obs[-1]["privileged_state"][:, 21:] == 0).all()
        assert infos[-1]["final_observation"]["privileged_state"][:, -1].all()
        obs, _, _, _, _ = env.step(command, auto_reset=False)
        assert obs["privileged_state"][:, -1].all()
    finally:
        env.env.close()


def test_async_collection_success_rate_counts_each_episode_once():
    from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
    from rlinf.utils.metric_utils import CumulativeSuccessRate

    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create({"algorithm": {"loss_type": "sac_residual"}})
    runner.collection_success = CumulativeSuccessRate()
    runner.env_metric_channel = asyncio.Queue()
    assert runner.get_env_metrics() == ({}, [], [])

    # Unequal worker batches and episode lengths must not weight the SR.
    for rank, outcomes, lengths in [
        (0, [True], [10.0]),
        (1, [False, False, True], [100.0, 100.0, 20.0]),
    ]:
        runner.env_metric_channel.put_nowait(
            {
                "rank": rank,
                "env": {
                    "env/success_once": torch.tensor(outcomes),
                    "env/episode_len": torch.tensor(lengths),
                },
            }
        )
    metrics, _, ranked = runner.get_env_metrics()
    assert metrics["env/collected_success_rate"] == 0.5
    assert float(metrics["env/success_once"]) == 0.5
    assert len(ranked) == 2
    assert runner.collection_success.completed == 4
    assert runner.get_env_metrics()[0] == {"env/collected_success_rate": 0.5}
    assert runner.collection_success.completed == 4

    runner.env_metric_channel.put_nowait(
        {"rank": 0, "env": {"env/success_once": torch.tensor([True, True])}}
    )
    metrics, _, _ = runner.get_env_metrics()
    assert float(metrics["env/success_once"]) == 1.0
    assert metrics["env/collected_success_rate"] == pytest.approx(4 / 6)
    # Evaluation aggregation must not modify training collection counts.
    runner._process_ranked_eval_results(
        [{"rank": 0, "env": {"env/success_once": torch.ones(50)}}], "env"
    )
    assert runner.collection_success.completed == 6


def test_async_collection_success_rate_does_not_change_other_algorithms():
    from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
    from rlinf.utils.metric_utils import CumulativeSuccessRate

    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create({"algorithm": {"loss_type": "sac"}})
    runner.collection_success = CumulativeSuccessRate()
    runner.env_metric_channel = asyncio.Queue()
    runner.env_metric_channel.put_nowait(
        {"rank": 0, "env": {"env/success_once": torch.tensor([True])}}
    )
    metrics, _, _ = runner.get_env_metrics()
    assert "env/collected_success_rate" not in metrics
    assert runner.collection_success.completed == 0


@pytest.mark.parametrize("warmup", [20000, 20001])
def test_residual_critic_warmup_freezes_actor_and_alpha(warmup):
    worker = make_worker()
    worker.cfg.algorithm.residual_schedule = {"critic_warmup_updates": warmup}
    editor = copy.deepcopy(worker.model.editor.state_dict())
    alpha = worker.entropy_temp.base_alpha.detach().clone()
    critic = copy.deepcopy(worker.model.q_head.state_dict())
    for completed in (19, 19979, 19999, warmup):
        worker.update_step = completed
        metrics = worker.update_one_epoch()
        assert not metrics["sac/actor_updated"]
        assert not worker.optimizer.state
        assert not worker.alpha_optimizer.state
        assert worker.lr_scheduler.last_epoch == 0
        assert worker.alpha_lr_scheduler.last_epoch == 0
        for key, value in worker.model.editor.state_dict().items():
            torch.testing.assert_close(value, editor[key], atol=0, rtol=0)
        torch.testing.assert_close(
            worker.entropy_temp.base_alpha, alpha, atol=0, rtol=0
        )
    assert any(
        not torch.equal(v, critic[k])
        for k, v in worker.model.q_head.state_dict().items()
    )
    worker.update_step = 20019
    assert worker.update_one_epoch()["sac/actor_updated"] == 1
    assert int(next(iter(worker.optimizer.state.values()))["step"]) == 1
    assert int(next(iter(worker.alpha_optimizer.state.values()))["step"]) == 1


def test_residual_critic_warmup_resume_contract():
    original = ResidualUpdateBudget(critic_warmup_updates=20000, updates=12000)
    restored = ResidualUpdateBudget(critic_warmup_updates=20000)
    restored.restore(asdict(original))
    assert restored.updates == 12000
    with pytest.raises(ValueError, match="critic_warmup_updates"):
        ResidualUpdateBudget().restore(asdict(original))
    legacy = asdict(ResidualUpdateBudget())
    del legacy["critic_warmup_updates"]
    ResidualUpdateBudget().restore(legacy)


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires ManiSkill GPU physics and GPU IK.",
)
def test_peg_preinsert_reconfiguration_seed_sequence_and_grasp():
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/env/maniskill_peg_preinsert.yaml"
    )
    cfg.init_params.reconfiguration_freq = 1
    cfg.init_params.hole_clearance = 0.001
    cfg.total_num_envs = 8
    cfg.auto_reset = False
    env = ManiskillRLTEnv(cfg, 8, 0, 1, None)
    try:
        task = env.env.unwrapped
        command = torch.zeros((8, 7), device=env.device)
        command[:, -1] = -1
        seeds, poses = [], []
        for repeat in range(2):
            for episode in range(4):
                obs, info = env.reset(
                    seed=list(range(440000, 440008)) if episode == 0 else None,
                    options={},
                )
                assert info["reconfigure"]
                current_seed = task._episode_seed.copy()
                current_pose = torch.cat(
                    (
                        task.peg.pose.raw_pose,
                        task.box.pose.raw_pose,
                        task.agent.robot.qpos,
                    ),
                    -1,
                ).clone()
                if repeat == 0:
                    seeds.append(current_seed)
                    poses.append(current_pose)
                else:
                    assert current_seed.tolist() == seeds[episode].tolist()
                    torch.testing.assert_close(
                        current_pose, poses[episode], atol=1e-5, rtol=0
                    )
                assert (obs["privileged_state"][:, 21:] == 0).all()
                for _ in range(5):
                    env.step(command, auto_reset=False)
                    assert task.agent.is_grasping(task.peg).all()
        assert len({int(seed) for batch in seeds for seed in batch}) == 32
        assert not torch.equal(poses[0], poses[1])
    finally:
        env.env.close()


# EXPO shares the residual replay and optimizer lifecycle.
def make_expo_policy(num_candidates=3, num_edits=2):
    from rlinf.models.embodiment.mlp_policy.expo_policy import EXPOConfig, EXPOPolicy

    return EXPOPolicy(
        EXPOConfig(
            **asdict(make_policy().config),
            num_candidates=num_candidates,
            num_edits=num_edits,
        )
    )


def make_expo_obs(batch_size=4, num_candidates=3):
    return {
        **make_obs(batch_size),
        "candidates": torch.randn(batch_size, num_candidates, 6),
    }


@pytest.mark.parametrize("num_edits", [0, 2, 3])
def test_expo_selection_matches_reference_and_uses_target_features(num_edits):
    policy = make_expo_policy(num_edits=num_edits)
    # Separate online and target features so accidental use of online values fails.
    with torch.no_grad():
        policy.encoder["state"][0].weight.add_(2)
        policy.editor[-1].weight.normal_(std=0.1)
    obs = make_expo_obs()
    generator = torch.Generator().manual_seed(12)
    reference_rng = torch.Generator().manual_seed(12)
    feature = policy.target_feature(obs)
    candidates = obs["candidates"]
    if num_edits:
        edited, _, _ = policy.sac_forward(
            {"base_action": candidates[:, :num_edits].reshape(-1, 6)},
            feature=feature[:, None]
            .expand(-1, num_edits, -1)
            .reshape(4 * num_edits, -1),
            generator=reference_rng,
        )
        assert (
            (edited - candidates[:, :num_edits].reshape(-1, 6)).abs()
            <= policy.edit_scale + 1e-6
        ).all()
        candidates = torch.cat((candidates, edited.reshape(4, num_edits, 6)), 1)
    indices = torch.randperm(4, generator=reference_rng)[:2]
    scores = []
    for candidate in candidates.unbind(1):
        values = policy.sac_q_forward(obs, candidate, target=True)
        scores.append(values[:, indices].amin(-1))
    expected_indices = torch.stack(scores, -1).argmax(-1)
    action, selected = policy(obs=obs, generator=generator)
    torch.testing.assert_close(selected, expected_indices)
    torch.testing.assert_close(action, candidates[torch.arange(4), expected_indices])
    torch.testing.assert_close(generator.get_state(), reference_rng.get_state())
    # Evaluation also samples edits, as in gr00t EXPO.
    eval_action, result = policy.predict_action_batch(
        obs, mode="eval", generator=torch.Generator().manual_seed(12)
    )
    torch.testing.assert_close(eval_action.flatten(1), action)
    torch.testing.assert_close(
        result["forward_inputs"]["expo_selected_index"], selected
    )


def test_expo_actor_edits_replay_action_and_uses_all_online_critics():
    from rlinf.algorithms.expo import expo_actor_loss

    policy = make_expo_policy()
    obs = make_expo_obs()
    action = torch.randn(4, 6) + 5
    batch = {"curr_obs": obs, "actions": action}
    alpha = torch.tensor(1e-5, requires_grad=True)
    torch.manual_seed(91)
    actual, log_prob, _ = expo_actor_loss(policy, batch, alpha)
    torch.manual_seed(91)
    edited, per_dim, _ = policy.sac_forward({**obs, "base_action": action})
    q = policy.sac_q_forward(obs, edited, detach_encoder=True).mean(-1, keepdim=True)
    expected = (alpha.detach() * per_dim.sum(-1, keepdim=True) - q).mean()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(log_prob, per_dim.sum(-1, keepdim=True))
    actual.backward()
    assert policy.editor[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in policy.encoder.parameters())
    assert all(p.grad is None for p in policy.target_encoder.parameters())
    assert all(p.grad is None for p in policy.target_q_head.parameters())
    assert alpha.grad is None
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(91)
    monitored, _, metrics = expo_actor_loss(
        policy, batch, alpha, actor_parameters=list(policy.editor.parameters())
    )
    torch.testing.assert_close(monitored, actual)
    assert math.isfinite(metrics["actor_q_to_entropy_grad_ratio"])


@pytest.mark.parametrize("bootstrap_time_limits", [True, False])
def test_expo_backup_matches_independent_pair_and_timeout_contract(
    bootstrap_time_limits,
):
    from rlinf.algorithms.expo import expo_critic_loss

    policy = make_expo_policy()
    batch = {
        "curr_obs": make_expo_obs(),
        "next_obs": make_expo_obs(),
        "actions": torch.randn(4, 6),
        "rewards": torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        "terminations": torch.tensor(
            [[False, False, True], [False] * 3, [False] * 3, [False] * 3]
        ),
        "truncations": torch.tensor(
            [[False] * 3, [False, False, True], [False] * 3, [False] * 3]
        ),
    }
    rng = torch.Generator().manual_seed(71)
    with torch.no_grad():
        action, _ = policy(obs=batch["next_obs"], generator=rng)
        next_q = random_min_q(
            policy.sac_q_forward(batch["next_obs"], action, target=True), rng
        )
        terminal = batch["terminations"].any(-1, keepdim=True)
        if not bootstrap_time_limits:
            terminal |= batch["truncations"].any(-1, keepdim=True)
        target = batch["rewards"].sum(-1, keepdim=True) + 0.99 * (~terminal) * next_q
    values = policy.sac_q_forward(batch["curr_obs"], batch["actions"])
    actual, metrics = expo_critic_loss(
        policy,
        batch,
        0.99,
        torch.Generator().manual_seed(71),
        bootstrap_time_limits=bootstrap_time_limits,
    )
    torch.testing.assert_close(actual, (values - target).square().mean())
    assert metrics["target_q"] == pytest.approx(target.mean().item())
    actual.backward()
    assert any(p.grad is not None for p in policy.encoder.parameters())
    assert all(p.grad is None for p in policy.editor.parameters())
    assert all(p.grad is None for p in policy.target_q_head.parameters())


def test_expo_target_update_checkpoint_and_rollout_sync(tmp_path):
    from rlinf.utils.utils import collect_param_names_need_sync

    policy = make_expo_policy()
    old = copy.deepcopy(policy.target_encoder.state_dict())
    with torch.no_grad():
        for p in policy.encoder.parameters():
            p.add_(0.2)
    policy.update_targets(0.005)
    for key, value in policy.target_encoder.state_dict().items():
        torch.testing.assert_close(
            value, old[key] * 0.995 + policy.encoder.state_dict()[key] * 0.005
        )
    names = collect_param_names_need_sync(policy) + policy.target_parameter_names()
    assert set(names) == set(policy.state_dict())
    checkpoint = tmp_path / "weights.pt"
    torch.save(policy.state_dict(), checkpoint)
    restored = make_expo_policy()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    obs = make_expo_obs()
    expected = policy(obs=obs, generator=torch.Generator().manual_seed(89))
    actual = restored(obs=obs, generator=torch.Generator().manual_seed(89))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert all(not p.requires_grad for p in restored.target_encoder.parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_candidates": 0},
        {"num_candidates": 1.5},
        {"num_edits": -1},
        {"num_candidates": 2, "num_edits": 3},
    ],
)
def test_expo_rejects_invalid_candidates(kwargs):
    from rlinf.models.embodiment.mlp_policy.expo_policy import EXPOConfig

    with pytest.raises(ValueError):
        EXPOConfig(**kwargs).validate()


def test_expo_candidate_replay_boundary_keeps_terminal_observation():
    obs, final = make_expo_obs(1), make_expo_obs(1)
    forward = {"action": torch.randn(1, 6)}
    for prefix, source in (("residual_", obs), ("residual_next_", final)):
        forward.update({prefix + key: value for key, value in source.items()})
    for actual, expected in (
        (extract_residual_obs(forward), obs),
        (extract_residual_obs(forward, transition=True), final),
    ):
        assert actual.keys() == expected.keys()
        torch.testing.assert_close(actual["candidates"], expected["candidates"])


def test_expo_recipe_and_entropy_backup_rejection(monkeypatch, tmp_path):
    cfg = load_recipe(monkeypatch, "maniskill_expo_openpi_pi05_dinov3_eef_async")
    stats = tmp_path / "norm_stats.json"
    import json

    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    k: {"q01": [-1.0] * 10, "q99": [1.0] * 10}
                    for k in ("state", "actions")
                }
            }
        )
    )
    cfg.actor.model.eef_rot6d_norm_stats_path = str(stats)
    cfg.rollout.residual_base_model.openpi_data.norm_stats_path = str(stats)
    validate_residual_cfg(cfg)
    assert cfg.actor.model.num_candidates == cfg.actor.model.num_edits == 8
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 50
    assert cfg.env.train.reward_mode == "only_success"
    assert not cfg.env.train.critic_privileged_obs
    assert cfg.algorithm.residual_schedule.critic_warmup_updates == 20000
    cfg.algorithm.backup_entropy = True
    with pytest.raises(ValueError, match="backup_entropy=False"):
        validate_residual_cfg(cfg)


def test_expo_worker_reuses_delayed_actor_and_internal_target_updates():
    from rlinf.workers.actor.fsdp_expo_policy_worker import AsyncEXPOFSDPPolicy

    worker = make_worker(AsyncEXPOFSDPPolicy)
    worker.model = make_expo_policy()
    worker.target_model = None
    worker.optimizer = torch.optim.Adam(worker.model.editor.parameters(), lr=0.0003)
    worker.qf_optimizer = torch.optim.Adam(
        [*worker.model.encoder.parameters(), *worker.model.q_head.parameters()],
        lr=0.0003,
    )
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1.0
    )
    worker.qf_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.qf_optimizer, lambda _: 1.0
    )
    worker.soft_update_target_model(tau=1.0)
    batch = {
        "curr_obs": make_expo_obs(),
        "next_obs": make_expo_obs(),
        "actions": torch.randn(4, 6),
        "rewards": torch.zeros(4, 3),
        "terminations": torch.zeros(4, 3, dtype=torch.bool),
    }
    worker.buffer_dataloader_iter = iter([batch] * 20)
    for i in range(20):
        worker.update_step = i
        metrics = worker.update_one_epoch()
        assert metrics["sac/actor_updated"] == float(i == 19)
        assert 0 <= metrics["critic/target_edit_fraction"] <= 1
        assert all(p.grad is None for p in worker.model.target_encoder.parameters())
    assert int(next(iter(worker.optimizer.state.values()))["step"]) == 1
    assert int(next(iter(worker.qf_optimizer.state.values()))["step"]) == 20
    assert math.isfinite(metrics["actor_q_to_entropy_grad_ratio"])


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_residual_rejects_invalid_critic_evaluation_interval(monkeypatch, interval):
    cfg = load_recipe(monkeypatch, "maniskill_sac_residual_openpi_pi05_async")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    cfg.runner.val_critic_interval = interval
    with pytest.raises(ValueError, match="val_critic_interval"):
        validate_residual_cfg(cfg)


def test_expo_target_encoder_and_critic_canonicalize_rot6d(eef_stats):
    import json

    from rlinf.models.embodiment.mlp_policy.expo_policy import EXPOConfig, EXPOPolicy
    from rlinf.utils.torch_rot6d import canonicalize_rot6d

    stats = json.loads(eef_stats.read_text())["norm_stats"]["state"]
    low, high = (torch.tensor(stats[key]) for key in ("q01", "q99"))
    value = torch.randn(3, 10)
    value[:, 3:9] = canonicalize_rot6d(value[:, 3:9])
    equivalent = value.clone()
    equivalent[:, 3:6] *= 3
    equivalent[:, 6:9] = value[:, 6:9] * 2 + value[:, 3:6] * 4

    def normalize(x):
        return (x - low) / (high - low + 1e-6) * 2 - 1

    policy = EXPOPolicy(
        EXPOConfig(
            state_dim=10,
            action_dim=10,
            visual_dim=5,
            hidden_size=16,
            num_q_heads=2,
            num_action_chunks=2,
            eef_rot6d_norm_stats_path=str(eef_stats),
        )
    )
    obs = {"state": normalize(value), "visual": torch.randn(3, 5)}
    other = {**obs, "state": normalize(equivalent)}
    torch.testing.assert_close(
        policy.target_feature(obs), policy.target_feature(other), atol=2e-6, rtol=2e-6
    )
    a = normalize(value).repeat(1, 2)
    b = normalize(equivalent).repeat(1, 2)
    torch.testing.assert_close(
        policy.sac_q_forward(obs, a, target=True),
        policy.sac_q_forward(other, b, target=True),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_MANISKILL_GPU") != "1",
    reason="Requires ManiSkill GPU physics and GPU IK.",
)
def test_joint_peg_preinsert_reset_replay_and_grasp():
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from rlinf.envs.sim.maniskill.peg_preinsert import validate_joint_gpu_controller
    from toolkits.lerobot.collect_maniskill_peg_preinsert import expert_action

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/env/maniskill_peg_preinsert.yaml"
    )
    cfg.wrap_obs_mode = "rlt_openpi_joint"
    cfg.critic_privileged_obs = False
    cfg.init_params.control_mode = "pd_joint_delta_pos"
    cfg.init_params.hole_clearance = 0.001
    cfg.total_num_envs = 8
    cfg.max_episode_steps = cfg.init_params.max_episode_steps = 5
    cfg.auto_reset = True
    env = ManiskillRLTEnv(cfg, 8, 0, 1, None)
    try:
        task = env.env.unwrapped
        for first_seed in (320000, 320016, 320032):
            seeds = list(range(first_seed, first_seed + 8))
            obs, _ = env.reset(seed=seeds, options={"reconfigure": True})
            validate_joint_gpu_controller(env.env)
            assert obs["states"].shape == (8, 9)
            assert "privileged_state" not in obs
            head = (task.box_hole_pose.inv() * task.peg_head_pose).p
            gap = -task.peg_half_sizes[:, 0] - head[:, 0]
            torch.testing.assert_close(
                gap, torch.full_like(gap, 0.05), atol=2e-6, rtol=0
            )
            assert (head[:, 1:].abs() <= 0.030001).all()
            initial = obs["states"].clone()
            command = expert_action(task)
            assert command.shape == (8, 8) and (command.abs() <= 1).all()
            next_obs, *_ = env.step(command, auto_reset=False)
            assert task.agent.is_grasping(task.peg).all()
            expected = next_obs["states"].clone()
            env.reset(options={"env_idx": torch.tensor([1, 3], device=env.device)})
            torch.testing.assert_close(
                task.agent.robot.qpos[[0, 2, 4, 5, 6, 7]],
                expected[[0, 2, 4, 5, 6, 7]],
                atol=0.002,
                rtol=1e-5,
            )
            obs, _ = env.reset(seed=seeds, options={"reconfigure": True})
            torch.testing.assert_close(obs["states"], initial, atol=1e-5, rtol=0)
            next_obs, *_ = env.step(command, auto_reset=False)
            torch.testing.assert_close(next_obs["states"], expected, atol=0.002, rtol=0)
            hold = torch.zeros_like(command)
            hold[:, -1] = -1
            obs, _, _, truncations, _ = env.chunk_step(hold[:, None].expand(-1, 4, -1))
            assert truncations[:, -1].all()
            assert (env.elapsed_steps == 0).all()
            assert obs[-1]["states"].shape == (8, 9)
    finally:
        env.env.close()


@pytest.mark.parametrize("algorithm", ["sac_residual", "expo"])
@pytest.mark.parametrize("eef", [False, True])
@pytest.mark.parametrize("dinov3", [False, True])
def test_residual_representation_and_encoder_recipes(
    monkeypatch, eef_stats, algorithm, eef, dinov3
):
    from rlinf.models.embodiment.mlp_policy.expo_policy import EXPOConfig, EXPOPolicy

    name = f"maniskill_{algorithm}_openpi_pi05"
    name += "_dinov3" if dinov3 else ""
    name += "_eef" if eef else ""
    cfg = load_recipe(monkeypatch, name + "_async")
    if eef:
        cfg.actor.model.eef_rot6d_norm_stats_path = str(eef_stats)
        cfg.rollout.residual_base_model.openpi_data.norm_stats_path = str(eef_stats)
    validate_residual_cfg(cfg)
    model_cfg = cfg.actor.model
    assert cfg.algorithm.loss_type == model_cfg.model_type == algorithm
    assert model_cfg.state_dim == (10 if eef else 9)
    assert model_cfg.action_dim == (10 if eef else 8)
    assert model_cfg.visual_dim == (3072 if dinov3 else 2048)
    assert bool(cfg.rollout.get("residual_dinov3")) == dinov3
    config_cls, policy_cls = (
        (EXPOConfig, EXPOPolicy)
        if algorithm == "expo"
        else (SACResidualConfig, SACResidualPolicy)
    )
    settings = config_cls(
        **{
            key: model_cfg[key]
            for key in config_cls.__dataclass_fields__
            if key in model_cfg
        }
    )
    policy = policy_cls(settings)
    obs = {
        "state": torch.randn(2, settings.state_dim),
        "visual": torch.randn(2, settings.visual_dim),
        "base_action": torch.randn(2, settings.chunk_dim),
    }
    if algorithm == "expo":
        obs["candidates"] = torch.randn(2, settings.num_candidates, settings.chunk_dim)
        actions, _ = policy.select(obs)
        assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 50
        assert cfg.env.train.init_params.hole_clearance == 0.001
    else:
        actions, _, _ = policy.sac_forward(obs, deterministic=True)
    q = policy.sac_q_forward(obs, actions)
    assert actions.shape == (2, settings.chunk_dim)
    assert q.shape == (2, settings.num_q_heads)
    assert torch.isfinite(actions).all() and torch.isfinite(q).all()
