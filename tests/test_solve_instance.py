"""Tests for best-of-k policy inference: greedy, sampling, multi-start, augmentation (P4.7)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.augment import AUGMENT_NUM
from vrp_diffusion_quantum.data.dataset import CVRPBatch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.inference.policy_support import (
    augment_instance,
    collate_instances,
    constraint_matrix_prior,
)
from vrp_diffusion_quantum.inference.solve_instance import (
    load_policy_checkpoint,
    solve_instance,
    solve_instances,
    summarize_solutions,
)
from vrp_diffusion_quantum.models.decoder import CVRPPolicy
from vrp_diffusion_quantum.train.train_policy import build_policy_from_config
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes


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
        "use_local_encoder": False,
        "use_local_pointer": False,
    }
    config.update(overrides)
    return build_policy_from_config(config)


def test_greedy_solve_returns_a_validated_route_set() -> None:
    example = _example(8, seed=1)
    solution = solve_instance(
        _policy(),
        example.instance,
        reference_cost=example.solution.cost,
    )

    assert solution.feasible
    assert validate_routes(example.instance, solution.routes).feasible
    assert solution.cost == pytest.approx(route_cost(example.instance, solution.routes))
    assert solution.num_vehicles == len(solution.routes)
    assert solution.num_candidates == 1
    assert solution.method == "policy_greedy"
    assert solution.gap_to_reference is not None


def test_more_starts_and_augmentations_never_worsen_the_best_cost() -> None:
    policy = _policy()
    example = _example(10, seed=2)

    single = solve_instance(policy, example.instance)
    multi_start = solve_instance(policy, example.instance, num_starts=6)
    augmented = solve_instance(
        policy, example.instance, num_starts=6, num_augmentations=AUGMENT_NUM
    )

    assert multi_start.num_candidates == 6
    assert augmented.num_candidates == 6 * AUGMENT_NUM
    assert multi_start.cost <= single.cost + 1e-6
    assert augmented.cost <= multi_start.cost + 1e-6
    for solution in (single, multi_start, augmented):
        assert validate_routes(example.instance, solution.routes).feasible


def test_augmentation_is_distance_preserving_so_costs_transfer() -> None:
    example = _example(9, seed=3)
    solution = solve_instance(
        _policy(), example.instance, num_starts=4, num_augmentations=AUGMENT_NUM
    )
    variant = augment_instance(example.instance, solution.best_augmentation)

    assert solution.best_augmentation >= 0
    assert route_cost(variant, solution.routes) == pytest.approx(solution.cost, abs=1e-6)
    assert validate_routes(variant, solution.routes).feasible


def test_sampling_explores_more_candidates_and_is_reproducible() -> None:
    policy = _policy()
    example = _example(10, seed=4)
    kwargs = {"decode_mode": "sampling", "num_starts": 3, "num_samples": 4, "seed": 17}

    first = solve_instance(policy, example.instance, **kwargs)  # type: ignore[arg-type]
    second = solve_instance(policy, example.instance, **kwargs)  # type: ignore[arg-type]

    assert first.num_candidates == 12
    assert first.routes == second.routes
    assert first.cost == pytest.approx(second.cost)
    assert validate_routes(example.instance, first.routes).feasible


def test_greedy_decoding_rejects_repeated_samples() -> None:
    with pytest.raises(ValueError, match="num_samples > 1 needs sampling"):
        solve_instance(_policy(), _example(5, seed=5).instance, num_samples=3)


def test_augmentation_count_is_bounded() -> None:
    with pytest.raises(ValueError, match="num_augmentations must be in"):
        solve_instance(_policy(), _example(5, seed=6).instance, num_augmentations=AUGMENT_NUM + 1)


def test_a_set_of_instances_produces_best_of_k_feasible_routes() -> None:
    policy = _policy()
    examples = [_example(size, seed=10 + size) for size in (6, 8, 10, 12)]
    solutions = solve_instances(
        policy,
        examples,
        decode_mode="sampling",
        num_starts=4,
        num_samples=2,
        num_augmentations=4,
        seed=99,
    )

    assert len(solutions) == len(examples)
    for example, solution in zip(examples, solutions, strict=True):
        assert solution.instance_id == example.instance.instance_id
        assert solution.feasible
        assert solution.num_candidates == 4 * 2 * 4
        report = validate_routes(example.instance, solution.routes)
        assert report.feasible, report.violations
        served = sorted(customer for route in solution.routes for customer in route)
        assert served == list(range(example.instance.n_customers))
        assert solution.gap_to_reference is not None

    summary = summarize_solutions(solutions)
    assert summary["feasibility_rate"] == 1.0
    assert summary["num_instances"] == 4.0
    assert math.isfinite(summary["mean_cost"])
    assert math.isfinite(summary["mean_gap_to_reference"])


def test_unlabelled_instances_are_solved_without_a_reference() -> None:
    solutions = solve_instances(_policy(), [_example(7, seed=21).instance])
    assert solutions[0].feasible
    assert solutions[0].reference_cost is None
    assert solutions[0].gap_to_reference is None


def test_policy_with_a_local_encoder_uses_the_diffusion_prior() -> None:
    example = _example(8, seed=31)
    policy = _policy(use_local_encoder=True, use_local_pointer=True)

    def prior(batch: CVRPBatch) -> torch.Tensor:
        # Every geometric variant shares the instance's route-membership structure.
        matrix = torch.as_tensor(example.constraint_matrix, dtype=torch.float32)
        return matrix.unsqueeze(0).expand(batch.coords.shape[0], -1, -1)

    solution = solve_instance(policy, example.instance, prior=prior, num_augmentations=3)
    assert solution.feasible
    assert validate_routes(example.instance, solution.routes).feasible

    with pytest.raises(ValueError, match="requires m_hat"):
        solve_instance(policy, example.instance)


def test_collate_instances_leaves_label_fields_empty() -> None:
    example = _example(6, seed=41)
    batch = collate_instances([example.instance])
    assert bool(batch.constraint_matrix.eq(0).all())
    assert float(batch.cost[0]) == 0.0
    assert bool(batch.customer_mask.all())
    assert "routes" not in batch.metadata[0]


def test_policy_checkpoint_round_trip(tmp_path: Path) -> None:
    model_config = {
        "embedding_dim": 32,
        "global_num_layers": 1,
        "global_num_heads": 4,
        "feed_forward_dim": 64,
        "decoder_num_heads": 4,
        "use_local_encoder": False,
        "use_local_pointer": False,
    }
    torch.manual_seed(0)
    policy = build_policy_from_config(model_config)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": policy.state_dict(), "extra": {"model": model_config}}, checkpoint)

    restored, payload = load_policy_checkpoint(checkpoint)
    assert payload["extra"]["model"] == model_config

    example = _example(7, seed=51)
    original = solve_instance(policy, example.instance, num_starts=3)
    reloaded = solve_instance(restored, example.instance, num_starts=3)
    assert reloaded.routes == original.routes
    assert reloaded.cost == pytest.approx(original.cost)


def test_prior_helper_returns_the_labelled_matrix() -> None:
    example = _example(5, seed=61)
    from vrp_diffusion_quantum.data.dataset import collate_batch

    batch = collate_batch([example])
    assert torch.equal(constraint_matrix_prior(batch), batch.constraint_matrix)
