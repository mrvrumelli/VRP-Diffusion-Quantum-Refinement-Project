import math

import numpy as np
import pytest

from vrp_diffusion_quantum.data.types import CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.utils.feasibility import (
    route_cost,
    route_loads,
    validate_labeled_solution,
    validate_routes,
)


def _instance() -> CVRPInstance:
    return CVRPInstance(
        coords=np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]]),
        demands=np.array([0.0, 3.0, 4.0, 5.0]),
        capacity=7.0,
        depot_index=0,
        instance_id="cost_toy",
        n_customers=3,
        seed=0,
        generator_settings={"kind": "hand_crafted"},
    )


def test_route_cost_includes_depot_returns() -> None:
    assert route_cost(_instance(), [[0, 1], [2]]) == 22.0


def test_route_cost_handles_non_zero_depot() -> None:
    instance = CVRPInstance(
        coords=np.array([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]]),
        demands=np.array([3.0, 4.0, 0.0]),
        capacity=7.0,
        depot_index=2,
        instance_id="offset_depot",
        n_customers=2,
        seed=0,
        generator_settings={"kind": "hand_crafted"},
    )

    assert route_cost(instance, [[0, 1]]) == 12.0


def test_route_loads_uses_customer_demands() -> None:
    assert route_loads(_instance(), [[0, 1], [2]]) == [7.0, 5.0]


def test_route_cost_rejects_out_of_range_customer() -> None:
    with pytest.raises(ValueError, match="out of range"):
        route_cost(_instance(), [[3]])


def test_validate_routes_accepts_feasible_covering_solution() -> None:
    report = validate_routes(_instance(), [[0, 1], [2]])

    assert report.feasible
    assert report.violations == ()
    assert report.route_loads == (7.0, 5.0)


def test_validate_routes_reports_missing_duplicate_out_of_range_and_capacity() -> None:
    report = validate_routes(_instance(), [[0, 0, 5], [1, 2]])

    assert not report.feasible
    assert report.missing_customers == ()
    assert report.duplicate_customers == (0,)
    assert report.out_of_range_customers == (5,)
    assert any("exceeds capacity" in violation for violation in report.violations)
    assert any("appears more than once" in violation for violation in report.violations)
    assert any("out of range" in violation for violation in report.violations)


def test_validate_routes_reports_missing_customers() -> None:
    report = validate_routes(_instance(), [[0]])

    assert not report.feasible
    assert report.missing_customers == (1, 2)


def test_validate_labeled_solution_checks_vehicle_count_and_cost() -> None:
    solution = LabeledSolution(
        routes=[[0, 1], [2]],
        cost=999.0,
        num_vehicles=1,
        feasible=True,
        solver_name="hand_checked",
        time_budget=None,
        seed=0,
        runtime_seconds=0.001,
    )

    report = validate_labeled_solution(_instance(), solution)

    assert not report.feasible
    assert any("num_vehicles" in violation for violation in report.violations)
    assert any("computed cost" in violation for violation in report.violations)


def test_validate_labeled_solution_accepts_matching_cost() -> None:
    routes = [[0, 1], [2]]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(_instance(), routes),
        num_vehicles=2,
        feasible=True,
        solver_name="hand_checked",
        time_budget=None,
        seed=0,
        runtime_seconds=0.001,
    )

    report = validate_labeled_solution(_instance(), solution)

    assert report.feasible
    assert math.isclose(solution.cost, 22.0)
