"""Tests for classical local search baselines on refinement subproblems (task Q1.6)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from test_quantum_neighborhoods import make_instance
from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.local_search.baselines import refine, solve_exchange, solve_reorder
from vrp_diffusion_quantum.quantum.neighborhoods import (
    ExchangeSubproblem,
    Neighborhood,
    ReorderSubproblem,
    evaluate_exchange,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
)
from vrp_diffusion_quantum.quantum.qubo import solve_exact
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    build_exchange_qubo,
    decode_exchange,
    repair_exchange,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import build_reorder_qubo, decode_reorder
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes


def random_reorder(k: int, seed: int) -> ReorderSubproblem:
    rng = np.random.default_rng(seed)
    instance = make_instance([tuple(p) for p in rng.random((k + 2, 2))])
    route = list(range(k + 2))
    neighborhood = Neighborhood(
        "high_cost_route", "reorder", (0,), tuple(route[1 : k + 1]), 1.0, (1, k + 1)
    )
    return extract_reorder_subproblem(instance, [route], neighborhood)


@pytest.mark.parametrize("k", [1, 2, 5, 6])
@pytest.mark.parametrize("seed", [0, 1])
def test_exact_reorder_matches_brute_force(k: int, seed: int) -> None:
    subproblem = random_reorder(k, seed)
    solution = solve_reorder(subproblem, "exact")
    best = min(subproblem.path_cost(p) for p in itertools.permutations(subproblem.customers))
    assert solution.cost_after == pytest.approx(best)
    assert sorted(solution.order) == sorted(subproblem.customers)


@pytest.mark.parametrize("seed", range(5))
def test_two_opt_never_worsens_and_returns_a_permutation(seed: int) -> None:
    subproblem = random_reorder(7, seed)
    solution = solve_reorder(subproblem, "two_opt")
    assert sorted(solution.order) == sorted(subproblem.customers)
    assert solution.cost_after <= solution.cost_before + 1e-12
    assert solution.evaluations > 0


def test_two_opt_untangles_a_crossed_line() -> None:
    instance = make_instance([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    route = [0, 2, 1, 3]
    neighborhood = Neighborhood("high_cost_route", "reorder", (0,), (0, 2, 1, 3), 1.0, (0, 4))
    subproblem = extract_reorder_subproblem(instance, [route], neighborhood)
    solution = solve_reorder(subproblem, "two_opt")
    assert solution.cost_after == pytest.approx(8.0)  # out to the far end and back


def misplaced_exchange(
    capacity: float = 10.0, demands: list[float] | None = None
) -> tuple[CVRPInstance, list[list[int]], Neighborhood, ExchangeSubproblem]:
    xy = [(2.0, 0.0), (2.0, 1.0), (-2.0, 0.0), (-2.0, 1.0), (2.5, 0.5)]
    instance = make_instance(xy, demands=demands or [1, 1, 1, 1, 1], capacity=capacity)
    routes = [[0, 1], [2, 3, 4]]
    neighborhood = Neighborhood("two_route_exchange", "exchange", (0, 1), (4, 1, 3), 1.0)
    return (
        instance,
        routes,
        neighborhood,
        extract_exchange_subproblem(instance, routes, neighborhood),
    )


@pytest.mark.parametrize(("capacity", "demands"), [(10.0, [1, 1, 1, 1, 1]), (4.0, [1, 2, 1, 1, 2])])
def test_exhaustive_exchange_is_the_feasible_optimum(capacity: float, demands: list[float]) -> None:
    instance, _, _, subproblem = misplaced_exchange(capacity, demands)
    solution = solve_exchange(instance, subproblem, "exhaustive")
    feasible = [
        a for a in itertools.product((0, 1), repeat=subproblem.size) if subproblem.is_feasible(a)
    ]
    best = min(evaluate_exchange(instance, subproblem, a)[0] for a in feasible)
    assert solution.feasible
    assert solution.cost_after == pytest.approx(best)
    assert solution.evaluations == len(feasible) + int(
        subproblem.initial_assignment not in feasible
    )


def test_relocate_swap_moves_the_misplaced_customer() -> None:
    instance, _, _, subproblem = misplaced_exchange()
    solution = solve_exchange(instance, subproblem, "relocate_swap")
    assert solution.assignment == (1, 1, 0)
    assert solution.feasible and solution.improvement > 0


def test_classical_and_qubo_solvers_share_identical_subproblems() -> None:
    reorder = random_reorder(4, 3)
    reorder_qubo = build_reorder_qubo(reorder)
    qubo_order = decode_reorder(reorder_qubo, solve_exact(reorder_qubo.qubo).x)
    classical = solve_reorder(reorder, "exact")
    assert qubo_order is not None
    assert reorder.path_cost(qubo_order) == pytest.approx(classical.cost_after)

    instance, _, _, exchange = misplaced_exchange(4.0, [1, 2, 1, 1, 2])
    qubo = build_exchange_qubo(instance, exchange)
    assignment = repair_exchange(qubo, decode_exchange(qubo, solve_exact(qubo.qubo).x))
    assert assignment is not None
    qubo_cost = evaluate_exchange(instance, exchange, assignment)[0]
    exhaustive = solve_exchange(instance, exchange, "exhaustive")
    assert exhaustive.cost_after <= qubo_cost + 1e-9  # the classical exact reference bounds it


def test_refine_accepts_only_feasible_improvements() -> None:
    instance, routes, neighborhood, _ = misplaced_exchange()
    result = refine(instance, routes, neighborhood, "exhaustive")
    assert result.accepted
    assert validate_routes(instance, [r for r in result.routes if r]).feasible
    assert result.cost_after == pytest.approx(route_cost(instance, result.routes))
    assert result.accepted_improvement == pytest.approx(result.cost_before - result.cost_after)
    assert result.solver_name == "classical_exhaustive"

    optimal = [list(r) for r in result.routes]
    again = refine(
        instance,
        optimal,
        Neighborhood("two_route_exchange", "exchange", (0, 1), (4, 1, 3), 1.0),
        "exhaustive",
    )
    assert not again.accepted
    assert again.routes == optimal and again.cost_after == again.cost_before


def test_refine_reorder_and_method_validation() -> None:
    instance = make_instance([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    routes = [[0, 2, 1, 3]]
    neighborhood = Neighborhood("high_cost_route", "reorder", (0,), (0, 2, 1, 3), 1.0, (0, 4))
    result = refine(instance, routes, neighborhood, "exact")
    assert result.accepted
    assert result.cost_after == pytest.approx(8.0)  # several orders tie on a line
    assert sorted(result.routes[0]) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="not a reorder method"):
        refine(instance, routes, neighborhood, "exhaustive")
