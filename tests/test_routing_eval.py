import numpy as np
import pytest

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.eval.routing import (
    decode_matrix_to_routes,
    evaluate_decoded_matrix,
    summarize_routing_evaluations,
)
from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes


def _example() -> CVRPExample:
    instance = CVRPInstance(
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
        demands=np.array([0.0, 2.0, 2.0, 2.0, 2.0]),
        capacity=4.0,
        depot_index=0,
        instance_id="decode_toy",
        n_customers=4,
        seed=1,
        generator_settings={},
    )
    routes = [[0, 1], [2, 3]]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=2,
        feasible=True,
        solver_name="toy",
        time_budget=None,
        seed=1,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_decode_exact_membership_is_feasible() -> None:
    example = _example()
    routes, repairs = decode_matrix_to_routes(example.instance, example.constraint_matrix)
    assert validate_routes(example.instance, routes).feasible
    assert {frozenset(route) for route in routes} == {frozenset((0, 1)), frozenset((2, 3))}
    assert repairs == 0


def test_decode_repairs_over_capacity_component() -> None:
    example = _example()
    routes, repairs = decode_matrix_to_routes(example.instance, np.ones((4, 4)))
    assert validate_routes(example.instance, routes).feasible
    assert len(routes) == 2
    assert repairs == 1


def test_decode_rejects_invalid_inputs() -> None:
    example = _example()
    with pytest.raises(ValueError, match="matrix shape"):
        decode_matrix_to_routes(example.instance, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="threshold"):
        decode_matrix_to_routes(example.instance, np.zeros((4, 4)), threshold=2.0)


def test_evaluate_and_summarize_exact_matrix() -> None:
    example = _example()
    result = evaluate_decoded_matrix(example, build_constraint_matrix([[0, 1], [2, 3]], 4))
    assert result.feasible
    assert result.cost_gap_percent == pytest.approx(0.0)
    assert result.matrix_pair_accuracy == 1.0
    summary = summarize_routing_evaluations([result])
    assert summary["route_feasible_rate"] == 1.0
    assert summary["route_mean_cost_gap_percent"] == pytest.approx(0.0)
