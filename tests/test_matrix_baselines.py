"""Tests for deterministic heuristic constraint-matrix baselines."""

from __future__ import annotations

import numpy as np

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.matrix_baselines import (
    nearest_neighbor_cluster_matrix,
    predict_heuristic_matrices,
    random_cluster_matrix,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost


def _example() -> CVRPExample:
    instance = CVRPInstance(
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
        demands=np.array([0.0, 2.0, 2.0, 3.0, 3.0]),
        capacity=6.0,
        depot_index=0,
        instance_id="baseline_toy",
        n_customers=4,
        seed=0,
        generator_settings={"kind": "hand_checked"},
    )
    routes = [[0, 1], [2, 3]]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=2,
        feasible=True,
        solver_name="hand_checked",
        time_budget=None,
        seed=0,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_heuristic_baselines_are_binary_symmetric_zero_diagonal() -> None:
    example = _example()
    predictions = predict_heuristic_matrices(example, random_seed=7)
    assert set(predictions) == {
        "all_zero",
        "nearest_neighbor",
        "demand_aware",
        "random",
    }
    for matrix in predictions.values():
        assert matrix.shape == (4, 4)
        assert np.all(np.isin(matrix, [0.0, 1.0]))
        assert np.array_equal(matrix, matrix.T)
        assert np.all(np.diag(matrix) == 0.0)


def test_every_nonzero_baseline_produces_transitive_clusters() -> None:
    example = _example()
    predictions = predict_heuristic_matrices(example, random_seed=7)
    for name, matrix in predictions.items():
        if name == "all_zero":
            continue
        for customer in range(example.instance.n_customers):
            cluster = np.flatnonzero(matrix[customer] > 0.5)
            if len(cluster) > 1:
                assert np.all(matrix[np.ix_(cluster, cluster)] == 1.0) or np.all(
                    matrix[np.ix_(cluster, cluster)] + np.eye(len(cluster)) == 1.0
                )


def test_demand_aware_clusters_respect_capacity() -> None:
    example = _example()
    matrix = predict_heuristic_matrices(example, random_seed=7)["demand_aware"]
    demands = example.instance.customer_demands()
    for customer in range(example.instance.n_customers):
        cluster = [*np.flatnonzero(matrix[customer] > 0.5).tolist(), customer]
        assert float(np.sum(demands[sorted(set(cluster))])) <= example.instance.capacity


def test_nearest_neighbor_baseline_is_clustered_not_only_pairwise() -> None:
    example = _example()
    matrix = nearest_neighbor_cluster_matrix(example)
    assert matrix[0, 1] == 1.0
    assert matrix[2, 3] == 1.0


def test_random_baseline_is_seeded_and_reproducible() -> None:
    example = _example()
    first = random_cluster_matrix(example, seed=3)
    second = random_cluster_matrix(example, seed=3)
    different = random_cluster_matrix(example, seed=9)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
