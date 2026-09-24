"""Tests for the matrix-independent Phase 5 M-ablation baselines."""

from __future__ import annotations

import numpy as np
import pytest

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.eval.baselines import clarke_wright_routes, random_constraint_matrix
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes


def _instance(n_customers: int = 5, *, seed: int = 0) -> CVRPInstance:
    # Two well-separated clusters near the depot, plus a lone far customer, hand-checkable.
    coords = np.array(
        [
            [0.0, 0.0],  # depot
            [1.0, 0.0],
            [1.1, 0.1],
            [-1.0, 0.0],
            [-1.1, -0.1],
            [5.0, 5.0],
        ]
    )[: n_customers + 1]
    demands = np.array([0.0, 2.0, 2.0, 2.0, 2.0, 2.0])[: n_customers + 1]
    return CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=5.0,
        depot_index=0,
        instance_id=f"cw_toy_{seed}",
        n_customers=n_customers,
        seed=seed,
        generator_settings={},
    )


def test_clarke_wright_routes_are_feasible_and_deterministic() -> None:
    instance = _instance(5)
    routes = clarke_wright_routes(instance)
    report = validate_routes(instance, routes)
    assert report.feasible, report.violations

    again = clarke_wright_routes(instance)
    assert routes == again


def test_clarke_wright_groups_nearby_customers() -> None:
    # Customers 0/1 (indices 0,1) are near each other and far from customer 4 (the outlier).
    instance = _instance(5)
    routes = clarke_wright_routes(instance)
    outlier_route = next(route for route in routes if 4 in route)
    assert outlier_route == [4]


def test_clarke_wright_rejects_oversized_demand() -> None:
    instance = _instance(1)
    instance.demands[1] = instance.capacity + 1.0
    with pytest.raises(ValueError, match="exceeds vehicle capacity"):
        clarke_wright_routes(instance)


def test_clarke_wright_handles_trivial_sizes() -> None:
    assert clarke_wright_routes(_instance(0)) == []
    assert clarke_wright_routes(_instance(1)) == [[0]]


def test_clarke_wright_routes_use_reference_cost_helper() -> None:
    instance = _instance(5)
    routes = clarke_wright_routes(instance)
    # Just exercises the same cost helper the routing evaluator relies on downstream.
    assert route_cost(instance, routes) > 0.0


def _example_with_matrix() -> CVRPExample:
    instance = _instance(4)
    routes = [[0, 1], [2, 3]]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=2,
        feasible=True,
        solver_name="toy",
        time_budget=None,
        seed=0,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_random_constraint_matrix_is_symmetric_zero_diagonal_binary() -> None:
    matrix = random_constraint_matrix(6, seed=1, density=0.4)
    assert matrix.shape == (6, 6)
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0.0)
    assert np.all(np.isin(matrix, [0.0, 1.0]))


def test_random_constraint_matrix_is_deterministic_given_seed() -> None:
    first = random_constraint_matrix(8, seed=42, density=0.3)
    second = random_constraint_matrix(8, seed=42, density=0.3)
    assert np.array_equal(first, second)


def test_random_constraint_matrix_diverges_across_seeds() -> None:
    first = random_constraint_matrix(8, seed=1, density=0.5)
    second = random_constraint_matrix(8, seed=2, density=0.5)
    assert not np.array_equal(first, second)


def test_random_constraint_matrix_density_matched_reproduces_true_pair_count() -> None:
    example = _example_with_matrix()
    reference = example.constraint_matrix
    expected_pairs = int(np.triu(reference, k=1).sum())
    matrix = random_constraint_matrix(4, seed=7, reference_matrix=reference)
    assert int(np.triu(matrix, k=1).sum()) == expected_pairs


def test_random_constraint_matrix_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        random_constraint_matrix(-1, seed=0)


def test_random_constraint_matrix_rejects_both_reference_and_density() -> None:
    example = _example_with_matrix()
    with pytest.raises(ValueError, match="at most one"):
        random_constraint_matrix(4, seed=0, reference_matrix=example.constraint_matrix, density=0.5)


def test_random_constraint_matrix_rejects_invalid_density() -> None:
    with pytest.raises(ValueError, match="density"):
        random_constraint_matrix(4, seed=0, density=1.5)
