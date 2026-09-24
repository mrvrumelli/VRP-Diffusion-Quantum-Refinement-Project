"""Regression tests for the constraint-matrix source ablation runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from vrp_diffusion_quantum.data.dataset import make_example, save_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_m_source_ablation",
    root / "scripts" / "run_m_source_ablation.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load run_m_source_ablation.py")
run_m_source_ablation = importlib.util.module_from_spec(spec)
sys.modules["run_m_source_ablation"] = run_m_source_ablation
spec.loader.exec_module(run_m_source_ablation)

from run_m_source_ablation import (  # noqa: E402
    _positive_pair_rate,
    _score_arm,
    capacity_distance_predicted_matrix,
    dense_no_mask_matrix,
    ground_truth_matrix,
    random_matrix,
    split_examples_by_size,
)


def _example(instance_id: str = "cvrp3_0", n_customers: int = 3) -> CVRPExample:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.9, 0.9],
        ],
        dtype=np.float64,
    )
    demands = np.array([0.0, 2.0, 2.0, 6.0], dtype=np.float64)
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=6.0,
        depot_index=0,
        instance_id=instance_id,
        n_customers=n_customers,
        seed=1,
        generator_settings={},
    )
    solution = LabeledSolution(
        routes=[[0, 1], [2]],
        cost=2.0,
        num_vehicles=2,
        feasible=True,
        solver_name="fixture",
        time_budget=None,
        seed=1,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_matrix_sources_are_symmetric_with_zero_diagonal() -> None:
    example = _example()

    matrices = [
        dense_no_mask_matrix(example),
        ground_truth_matrix(example),
        capacity_distance_predicted_matrix(example),
        random_matrix(example, positive_probability=0.3, seed=7, example_index=0),
    ]

    for matrix in matrices:
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, matrix.T)
        assert np.all(np.diag(matrix) == 0.0)


def test_capacity_distance_prediction_blocks_over_capacity_pairs() -> None:
    matrix = capacity_distance_predicted_matrix(_example())

    assert matrix[0, 1] > 0.0
    assert matrix[0, 2] == 0.0
    assert matrix[1, 2] == 0.0


def test_random_matrix_is_reproducible() -> None:
    example = _example()

    first = random_matrix(example, positive_probability=0.4, seed=11, example_index=3)
    second = random_matrix(example, positive_probability=0.4, seed=11, example_index=3)

    assert np.array_equal(first, second)


def test_split_examples_by_size_is_disjoint(tmp_path: Path) -> None:
    for index in range(4):
        save_example(_example(f"cvrp3_{index}"), tmp_path / f"cvrp3_{index}.json")

    splits = split_examples_by_size(
        tmp_path,
        sizes=[3],
        train_per_size=1,
        val_per_size=1,
        test_per_size=1,
        seed=5,
    )

    ids = {
        split_name: {example.instance.instance_id for example in examples}
        for split_name, examples in splits.items()
    }
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])


def test_ground_truth_arm_scores_perfect_matrix_f1() -> None:
    example = _example()

    row = _score_arm(
        "ground_truth_m",
        [example],
        [ground_truth_matrix(example)],
        threshold=0.5,
        adaptive_threshold=False,
        runtime_seconds=0.0,
    )

    assert row["matrix_f1"] == pytest.approx(1.0)
    assert row["route_feasible_rate"] == 1.0
    assert row["route_num_vehicles"] == 2.0


def test_positive_pair_rate_uses_off_diagonal_entries() -> None:
    assert _positive_pair_rate([_example()]) == pytest.approx(2 / 6)
