"""Tests for the two-route exchange / reassignment QUBO (task Q1.4)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from test_quantum_neighborhoods import make_instance
from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.quantum.neighborhoods import (
    Neighborhood,
    evaluate_exchange,
    extract_exchange_subproblem,
)
from vrp_diffusion_quantum.quantum.qubo import solve_exact
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    ExchangeQUBO,
    bounded_slack_coefficients,
    build_exchange_qubo,
    decode_exchange,
    repair_exchange,
)


@pytest.mark.parametrize("upper", [0, 1, 2, 5, 7, 8, 13, 50])
def test_bounded_slack_represents_exactly_zero_to_upper(upper: int) -> None:
    coefficients = bounded_slack_coefficients(upper)
    sums = {
        sum(c for c, bit in zip(coefficients, bits, strict=True) if bit)
        for bits in itertools.product((0, 1), repeat=len(coefficients))
    }
    assert sums == set(range(upper + 1))


def two_sided_instance(
    demands: list[float], capacity: float
) -> tuple[CVRPInstance, list[list[int]]]:
    # Route 0 lies to the east of the depot, route 1 to the west. Customer 4 sits in the east but
    # is currently served by the western route.
    xy = [(2.0, 0.0), (2.0, 1.0), (-2.0, 0.0), (-2.0, 1.0), (2.5, 0.5)]
    instance = make_instance(xy, demands=demands, capacity=capacity)
    return instance, [[0, 1], [2, 3, 4]]


def exchange_for(
    instance: CVRPInstance, routes: list[list[int]], movable: tuple[int, ...]
) -> ExchangeQUBO:
    neighborhood = Neighborhood("two_route_exchange", "exchange", (0, 1), movable, 1.0)
    subproblem = extract_exchange_subproblem(instance, routes, neighborhood)
    return build_exchange_qubo(instance, subproblem)


def test_misplaced_customer_moves_to_the_nearby_route_when_capacity_allows() -> None:
    instance, routes = two_sided_instance([1, 1, 1, 1, 1], capacity=10)
    exchange = exchange_for(instance, routes, (4, 1, 3))
    best = solve_exact(exchange.qubo)
    assignment = decode_exchange(exchange, best.x)
    assert assignment == (1, 1, 0)  # customer 4 joins the eastern route; 1 and 3 stay
    assert exchange.subproblem.initial_assignment == (0, 1, 0)
    before, _, _ = evaluate_exchange(instance, exchange.subproblem, (0, 1, 0))
    after, _, _ = evaluate_exchange(instance, exchange.subproblem, assignment)
    assert after < before


def test_loose_capacity_needs_no_slack_variables() -> None:
    instance, routes = two_sided_instance([1, 1, 1, 1, 1], capacity=10)
    exchange = exchange_for(instance, routes, (4, 1, 3))
    assert exchange.slack_slices == (None, None)
    assert exchange.qubo.num_variables == 3


def test_feasible_energy_equals_surrogate_and_infeasible_states_are_higher() -> None:
    # Capacity 4, fixed loads 1 and 1: customers 4 and 1 (demand 2 each) cannot both join the
    # eastern route, so the surrogate's favourite (1, 1, 0) is infeasible and the best feasible
    # assignment, worked out by hand, is (1, 0, 0).
    instance, routes = two_sided_instance([1, 2, 1, 1, 2], capacity=4)
    exchange = exchange_for(instance, routes, (4, 1, 3))
    qubo, subproblem = exchange.qubo, exchange.subproblem
    states = np.asarray(list(itertools.product((0, 1), repeat=qubo.num_variables)), dtype=float)
    energies = qubo.energies(states)
    best_feasible = np.inf
    worst_feasible = -np.inf
    lowest_infeasible = np.inf
    for state, energy in zip(states, energies, strict=True):
        assignment = decode_exchange(exchange, state.astype(int).tolist())
        if subproblem.is_feasible(assignment):
            best_feasible = min(best_feasible, energy)
            worst_feasible = max(worst_feasible, exchange.surrogate_cost(assignment))
            assert energy >= exchange.surrogate_cost(assignment) - 1e-9
        else:
            lowest_infeasible = min(lowest_infeasible, energy)
    assert lowest_infeasible > worst_feasible
    for assignment in itertools.product((0, 1), repeat=3):
        if subproblem.is_feasible(assignment):
            mask = np.all(states[:, :3] == np.asarray(assignment), axis=1)
            assert energies[mask].min() == pytest.approx(exchange.surrogate_cost(assignment))

    best = decode_exchange(exchange, solve_exact(qubo).x)
    assert subproblem.is_feasible(best)
    assert best == (1, 0, 0)


def test_repair_restores_capacity_or_reports_impossibility() -> None:
    instance, routes = two_sided_instance([1, 2, 1, 1, 2], capacity=4)
    exchange = exchange_for(instance, routes, (4, 1, 3))
    overloaded = (1, 1, 0)  # eastern route would carry 1 + 2 + 2 = 5 > 4
    assert not exchange.subproblem.is_feasible(overloaded)
    repaired = repair_exchange(exchange, overloaded)
    assert repaired is not None and exchange.subproblem.is_feasible(repaired)
    assert repair_exchange(exchange, (1, 0, 0)) == (1, 0, 0)

    tight, tight_routes = two_sided_instance([1, 2, 1, 2, 2], capacity=4)
    tight_exchange = exchange_for(tight, tight_routes, (4, 1))
    assert repair_exchange(tight_exchange, (1, 1)) is None  # 8 units cannot fit in 2 x 4


def test_invalid_subproblems_are_rejected() -> None:
    instance, routes = two_sided_instance([1.5, 1, 1, 1, 1], capacity=10)
    with pytest.raises(ValueError, match="integral"):
        exchange_for(instance, routes, (4, 1))
    heavy, heavy_routes = two_sided_instance([3, 3, 1, 1, 1], capacity=5)
    with pytest.raises(ValueError, match="fixed customers alone exceed capacity"):
        exchange_for(heavy, heavy_routes, (4,))
