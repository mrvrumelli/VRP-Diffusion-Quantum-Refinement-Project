"""Regression tests for leakage-free P2.1/P3 ablation scoring."""

from __future__ import annotations

import numpy as np
import pytest

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.eval.matrix_ablation import (
    score_matrix_probabilities,
    validate_disjoint_examples,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost


def _example(instance_id: str) -> CVRPExample:
    instance = CVRPInstance(
        coords=np.array([[0.0, 0.0], [0.1, 0.1], [0.9, 0.9]], dtype=np.float64),
        demands=np.array([0.0, 1.0, 1.0], dtype=np.float64),
        capacity=2.0,
        depot_index=0,
        instance_id=instance_id,
        n_customers=2,
        seed=0,
        generator_settings={},
    )
    routes = [[0, 1]]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=1,
        feasible=True,
        solver_name="test",
        time_budget=None,
        seed=0,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_fixed_test_threshold_is_not_retuned() -> None:
    example = _example("test")
    prediction = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
    selected = score_matrix_probabilities([example], [prediction])
    fixed = score_matrix_probabilities(
        [example],
        [prediction],
        threshold=0.5,
        adaptive_threshold=False,
    )
    assert selected["threshold"] < 0.5
    assert selected["f1"] == 1.0
    assert fixed["threshold"] == 0.5
    assert fixed["f1"] == 0.0


def test_selection_and_test_examples_must_be_disjoint() -> None:
    selection = _example("same-id")
    test = _example("same-id")
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_examples([selection], [test])


def test_distinct_selection_and_test_examples_are_allowed() -> None:
    validate_disjoint_examples([_example("selection")], [_example("test")])
