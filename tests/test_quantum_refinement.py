"""Tests for the refinement loop and its pluggable subproblem solvers."""

from __future__ import annotations

import numpy as np
import pytest

from test_quantum_neighborhoods import make_instance
from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.local_search.baselines import solve_reorder
from vrp_diffusion_quantum.quantum import refinement
from vrp_diffusion_quantum.quantum.neighborhoods import (
    Neighborhood,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
)
from vrp_diffusion_quantum.quantum.qubo_bias import DiffusionBiasConfig
from vrp_diffusion_quantum.quantum.refinement import (
    AnnealingQUBOSolver,
    ClassicalSolver,
    ExactQUBOSolver,
    refine_solution,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes


def misplaced_instance() -> tuple[CVRPInstance, list[list[int]]]:
    # Customer 4 sits next to the eastern route but is served by the western one, and the
    # western route visits its customers in a crossed order.
    xy = [(2.0, 0.0), (2.0, 1.0), (-2.0, 0.0), (-2.0, 1.0), (2.5, 0.5), (-2.5, 0.5)]
    instance = make_instance(xy, demands=[1, 1, 1, 1, 1, 1], capacity=10)
    return instance, [[0, 1], [3, 2, 5, 4]]


@pytest.mark.parametrize(
    "solver",
    [
        ClassicalSolver("exact", "exhaustive"),
        ClassicalSolver("two_opt", "relocate_swap"),
        AnnealingQUBOSolver(num_reads=8, num_sweeps=100, seed=1),
        ExactQUBOSolver(),
    ],
)
def test_refinement_never_worsens_and_accounts_for_every_gain(solver: object) -> None:
    instance, routes = misplaced_instance()
    trace = refine_solution(instance, routes, solver, max_rounds=3)  # type: ignore[arg-type]
    assert validate_routes(instance, trace.routes).feasible
    assert trace.final_cost == pytest.approx(route_cost(instance, trace.routes))
    assert trace.final_cost <= trace.initial_cost + 1e-12
    gains = sum(step.accepted_improvement for step in trace.steps)
    assert gains == pytest.approx(trace.improvement)
    assert all(step.accepted_improvement >= 0.0 for step in trace.steps)
    assert trace.improvement > 0  # the misplaced customer is always worth moving


def test_annealing_reorder_reaches_the_exact_optimum_on_a_small_segment() -> None:
    rng = np.random.default_rng(4)
    instance = make_instance([tuple(p) for p in rng.random((6, 2))])
    route = [0, 1, 2, 3, 4, 5]
    neighborhood = Neighborhood("high_cost_route", "reorder", (0,), (1, 2, 3, 4), 1.0, (1, 5))
    subproblem = extract_reorder_subproblem(instance, [route], neighborhood)
    exact = solve_reorder(subproblem, "exact")
    outcome = AnnealingQUBOSolver(num_reads=16, num_sweeps=300, seed=0).solve_reorder(
        instance, subproblem, None
    )
    assert outcome.candidate_cost == pytest.approx(exact.cost_after)
    assert outcome.qubo_num_variables == 16
    assert outcome.num_samples == 16
    assert "permutation" in outcome.penalty_weights


def test_annealing_exchange_returns_feasible_routes_under_tight_capacity() -> None:
    xy = [(2.0, 0.0), (2.0, 1.0), (-2.0, 0.0), (-2.0, 1.0), (2.5, 0.5)]
    instance = make_instance(xy, demands=[1, 2, 1, 1, 2], capacity=4)
    routes = [[0, 1], [2, 3, 4]]
    neighborhood = Neighborhood("two_route_exchange", "exchange", (0, 1), (4, 1, 3), 1.0)
    subproblem = extract_exchange_subproblem(instance, routes, neighborhood)
    outcome = AnnealingQUBOSolver(num_reads=8, num_sweeps=100, seed=2).solve_exchange(
        instance, subproblem, None
    )
    assert outcome.ordered_routes is not None and outcome.post_repair_feasible
    first, second = outcome.ordered_routes
    loads = [sum(instance.customer_demands()[c] for c in r) for r in (first, second)]
    assert max(loads) <= instance.capacity


def test_time_budget_draws_at_least_one_read_and_bias_runs_with_a_prior() -> None:
    instance, routes = misplaced_instance()
    m_prob = np.full((6, 6), 0.5)
    np.fill_diagonal(m_prob, 0.0)
    solver = AnnealingQUBOSolver(
        num_sweeps=20,
        seed=0,
        time_budget_seconds=0.0,
        bias=DiffusionBiasConfig(enabled=True, alpha=0.5),
    )
    assert solver.name == "qubo_annealing_bias0.5"
    trace = refine_solution(instance, routes, solver, m_prob=m_prob, max_rounds=1)
    assert trace.steps and all(step.num_samples >= 1 for step in trace.steps)
    assert {step.neighborhood_type for step in trace.steps} >= {"high_cost_route"}


def test_refinement_rejects_infeasible_starting_routes() -> None:
    instance, _ = misplaced_instance()
    with pytest.raises(ValueError, match="feasible"):
        refine_solution(instance, [[0, 1, 2]], ClassicalSolver())


def test_neighborhoods_made_stale_within_a_round_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Moving customer 4 from the western route to the eastern one shortens the western route, so
    # a reorder segment over its old full length falls outside it, and an exchange that still
    # lists customer 4 as movable between the western and northern routes no longer applies.
    xy = [(2.0, 0.0), (2.0, 1.0), (-2.0, 0.0), (-2.0, 1.0), (2.5, 0.5), (-2.5, 0.5), (0.0, 3.0)]
    instance = make_instance(xy, demands=[1] * 7, capacity=10)
    routes = [[0, 1], [3, 2, 5, 4], [6]]
    chosen = [
        Neighborhood("two_route_exchange", "exchange", (0, 1), (4,), 3.0),
        Neighborhood("high_cost_route", "reorder", (1,), (3, 2, 5, 4), 2.0, (0, 4)),
        Neighborhood("two_route_exchange", "exchange", (1, 2), (4,), 1.0),
    ]
    monkeypatch.setattr(refinement, "select_neighborhoods", lambda *args, **kwargs: chosen)
    trace = refine_solution(instance, routes, ClassicalSolver("exact", "exhaustive"), max_rounds=1)
    assert trace.skipped_stale == 2
    assert [step.neighborhood_type for step in trace.steps] == ["two_route_exchange"]
    assert 4 in trace.routes[0] and trace.improvement > 0
    assert validate_routes(instance, trace.routes).feasible


def test_classical_restarts_never_do_worse_than_one_run() -> None:
    rng = np.random.default_rng(7)
    instance = make_instance([tuple(p) for p in rng.random((9, 2))], demands=[1] * 9, capacity=5)
    routes = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    reorder = extract_reorder_subproblem(
        instance,
        routes,
        Neighborhood("high_cost_route", "reorder", (0,), (0, 1, 2, 3, 4), 1.0, (0, 5)),
    )
    exchange = extract_exchange_subproblem(
        instance,
        routes,
        Neighborhood("two_route_exchange", "exchange", (0, 1), (3, 4, 5, 6), 1.0),
    )
    single, multi = ClassicalSolver(), ClassicalSolver(restarts=8, seed=3)
    assert multi.name.endswith("_x8")
    one = single.solve_reorder(instance, reorder, None)
    many = multi.solve_reorder(instance, reorder, None)
    assert many.num_samples == 8
    assert many.candidate_cost <= one.candidate_cost + 1e-12
    assert many.order is not None
    assert many.candidate_cost == pytest.approx(reorder.path_cost(many.order))
    one_x = single.solve_exchange(instance, exchange, None)
    many_x = multi.solve_exchange(instance, exchange, None)
    assert many_x.post_repair_feasible
    assert many_x.candidate_cost <= one_x.candidate_cost + 1e-12
    with pytest.raises(ValueError, match="restarts"):
        ClassicalSolver(restarts=0)
