"""Tests for REINFORCE/POMO policy training (P4.6)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.decoder import CVRPPolicy
from vrp_diffusion_quantum.train.train_policy import (
    build_policy_from_config,
    constraint_matrix_prior,
    evaluate_policy,
    policy_gradient_loss,
    train_policy,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost


def _example(n_customers: int, *, seed: int, capacity: float = 12.0) -> CVRPExample:
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.random((n_customers, 2))])
    demands = np.concatenate([[0.0], rng.integers(1, 5, size=n_customers).astype(np.float64)])
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=capacity,
        depot_index=0,
        instance_id=f"cvrp{n_customers}_{seed}",
        n_customers=n_customers,
        seed=seed,
        generator_settings={},
    )
    routes: list[list[int]] = []
    current: list[int] = []
    load = 0.0
    for customer in range(n_customers):
        demand = float(demands[customer + 1])
        if load + demand > capacity and current:
            routes.append(current)
            current, load = [], 0.0
        current.append(customer)
        load += demand
    if current:
        routes.append(current)
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=len(routes),
        feasible=True,
        solver_name="unit",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def _examples(count: int = 4, n_customers: int = 6) -> list[CVRPExample]:
    return [_example(n_customers, seed=100 + index) for index in range(count)]


def _policy(**overrides: object) -> CVRPPolicy:
    torch.manual_seed(0)
    config: dict[str, object] = {
        "embedding_dim": 32,
        "global_num_layers": 1,
        "global_num_heads": 4,
        "local_num_layers": 1,
        "local_num_heads": 4,
        "feed_forward_dim": 64,
        "decoder_num_heads": 4,
    }
    config.update(overrides)
    return build_policy_from_config(config)


def test_multi_start_baseline_centres_the_advantage() -> None:
    cost = torch.tensor([1.0, 2.0, 3.0, 4.0])
    log_probability = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    loss, advantage = policy_gradient_loss(cost, log_probability, num_starts=2)

    assert advantage.tolist() == pytest.approx([-0.5, 0.5, -0.5, 0.5])
    expected = float((advantage * log_probability).mean())
    assert float(loss) == pytest.approx(expected)


def test_multi_start_baseline_requires_more_than_one_start() -> None:
    with pytest.raises(ValueError, match="num_starts >= 2"):
        policy_gradient_loss(torch.ones(2), torch.zeros(2), num_starts=1)


def test_greedy_rollout_baseline_uses_the_supplied_costs() -> None:
    cost = torch.tensor([1.0, 3.0])
    log_probability = torch.tensor([-1.0, -1.0])
    loss, advantage = policy_gradient_loss(
        cost,
        log_probability,
        num_starts=2,
        baseline_mode="greedy_rollout",
        baseline_cost=torch.tensor([2.0]),
    )
    assert advantage.tolist() == pytest.approx([-1.0, 1.0])
    assert float(loss) == pytest.approx(0.0)


def test_unknown_baseline_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported baseline_mode"):
        policy_gradient_loss(
            torch.ones(2),
            torch.zeros(2),
            num_starts=2,
            baseline_mode="ema",  # type: ignore[arg-type]
        )


def test_training_runs_without_nans_and_logs_reward_and_cost() -> None:
    policy = _policy()
    examples = _examples()
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    history = train_policy(
        policy,
        examples,
        num_epochs=2,
        learning_rate=1e-3,
        batch_size=2,
        num_starts=4,
        prior=constraint_matrix_prior,
        seed=7,
    )

    assert len(history) == 2
    for row in history:
        for key in ("train_loss", "train_cost", "train_reward", "train_entropy", "val_cost"):
            assert math.isfinite(float(row[key])), f"{key}={row[key]}"
        assert row["train_reward"] == pytest.approx(-row["train_cost"])
        assert row["train_cost"] > 0
        assert row["train_feasibility_rate"] == 1.0
        assert row["val_feasibility_rate"] == 1.0
        assert math.isfinite(float(row["gradient_norm"]))

    after = list(policy.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_training_reports_finite_metrics_for_every_ablation() -> None:
    examples = _examples(count=2)
    for overrides in (
        {"use_local_encoder": False, "use_local_pointer": False},
        {"use_savings_bias": False},
        {"use_global_pointer": False},
        {"use_context_perception": False},
    ):
        policy = _policy(**overrides)
        needs_prior = bool(overrides.get("use_local_encoder", True))
        history = train_policy(
            policy,
            examples,
            num_epochs=1,
            learning_rate=1e-3,
            batch_size=2,
            num_starts=3,
            prior=constraint_matrix_prior if needs_prior else None,
            seed=3,
        )
        assert math.isfinite(float(history[0]["train_loss"]))
        assert history[0]["val_feasibility_rate"] == 1.0


def test_training_writes_last_and_best_checkpoints(tmp_path: Path) -> None:
    policy = _policy()
    history = train_policy(
        policy,
        _examples(count=2),
        num_epochs=2,
        learning_rate=1e-3,
        batch_size=2,
        num_starts=3,
        prior=constraint_matrix_prior,
        seed=11,
        checkpoint_dir=tmp_path,
    )
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "best.pt").is_file()

    payload = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert payload["best_metric_name"] == "val_best_cost"
    assert math.isfinite(float(payload["best_metric_value"]))
    assert payload["epoch"] <= history[-1]["epoch"]


def test_greedy_rollout_baseline_trains_end_to_end() -> None:
    policy = _policy()
    history = train_policy(
        policy,
        _examples(count=2),
        num_epochs=1,
        learning_rate=1e-3,
        batch_size=2,
        num_starts=2,
        baseline_mode="greedy_rollout",
        prior=constraint_matrix_prior,
        seed=5,
    )
    assert math.isfinite(float(history[0]["train_loss"]))


def test_evaluate_policy_reports_gap_and_best_of_starts() -> None:
    policy = _policy()
    examples = _examples(count=3)
    metrics = evaluate_policy(
        policy,
        examples,
        batch_size=2,
        num_starts=4,
        prior=constraint_matrix_prior,
    )
    assert metrics["num_instances"] == 3.0
    assert metrics["best_cost"] <= metrics["cost"] + 1e-6
    assert metrics["reward"] == pytest.approx(-metrics["cost"])
    assert metrics["feasible_rate"] == 1.0
    assert math.isfinite(metrics["gap_percent"])
    assert metrics["reference_cost"] > 0


def test_local_encoder_policy_requires_a_prior() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="requires m_hat"):
        evaluate_policy(policy, _examples(count=1), batch_size=1, prior=None)


def test_training_rejects_an_empty_dataset() -> None:
    with pytest.raises(ValueError, match="empty list of examples"):
        train_policy(_policy(), [], num_epochs=1, learning_rate=1e-3)
