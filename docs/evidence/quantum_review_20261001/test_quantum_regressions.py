"""Failing acceptance tests for issues found in the Q1 audit; source is unchanged.

These deliberately fail against the reviewed implementation. They are kept outside
normal tests/ discovery so the audit does not silently change the project's test gate.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from test_quantum_audit import exchange_case, instance, reorder_case

from vrp_diffusion_quantum.local_search.baselines import solve_exchange, solve_reorder
from vrp_diffusion_quantum.quantum.neighborhoods import (
    Neighborhood,
    NeighborhoodConfig,
    apply_exchange,
    apply_reorder,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
    select_low_confidence_edges,
)
from vrp_diffusion_quantum.quantum.qubo import QUBOBuilder, solve_simulated_annealing
from vrp_diffusion_quantum.quantum.qubo_bias import (
    DiffusionBiasConfig,
    build_biased_exchange_qubo,
)
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    build_exchange_qubo,
    decode_exchange,
    repair_exchange,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import (
    build_reorder_qubo,
    decode_reorder,
    repair_reorder,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost


def test_q12_apply_reorder_rejects_stale_routes() -> None:
    obj = instance([(1, 0), (2, 0), (3, 0)])
    sub = extract_reorder_subproblem(
        obj, [[0, 1], [2]], Neighborhood("high_cost_route", "reorder", (0,), (0, 1), 1, (0, 2))
    )
    # Otherwise the result is [[1, 0], [1]]: customer 1 duplicated, customer 2 lost.
    with pytest.raises(ValueError):
        apply_reorder([[0, 2], [1]], sub, (1, 0))


def test_q12_apply_exchange_rejects_moving_fixed_customers() -> None:
    obj = instance([(1, 0), (-1, 0), (0, 1)])
    routes = [[0], [1, 2]]
    sub = extract_exchange_subproblem(
        obj, routes, Neighborhood("two_route_exchange", "exchange", (0, 1), (2,), 1)
    )
    # Fixed customers 0 and 1 change routes, outside the declared neighborhood.
    with pytest.raises(ValueError):
        apply_exchange(routes, sub, [1, 2], [0])


def test_q12_low_confidence_windows_obey_documented_overlap_rule() -> None:
    obj = instance([(i, 0) for i in range(6)])
    matrix = np.full((6, 6), 0.1)
    selected = select_low_confidence_edges(
        obj, [list(range(6))], matrix, NeighborhoodConfig(max_reorder_size=3)
    )
    for first, second in itertools.combinations(selected, 2):
        assert set(first.customers).isdisjoint(second.customers)


@pytest.mark.parametrize("assignment", [(2,), (-1,), (0.5,)])
def test_q12_feasibility_rejects_nonbinary_assignments(assignment: tuple[float, ...]) -> None:
    obj = instance([(1, 0), (-1, 0), (0, 1)])
    sub = extract_exchange_subproblem(
        obj, [[0], [1, 2]], Neighborhood("two_route_exchange", "exchange", (0, 1), (2,), 1)
    )
    try:
        feasible = sub.is_feasible(assignment)
    except ValueError:
        return
    assert not feasible, f"invalid assignment {assignment} was marked feasible"


@pytest.mark.parametrize("state", [(2, -1, -1, 2), (1.9, 0, 0, 1.9)])
def test_q13_decode_reorder_rejects_nonbinary_states(state: tuple[float, ...]) -> None:
    _, _, _, sub = reorder_case(2)
    model = build_reorder_qubo(sub)
    try:
        decoded = decode_reorder(model, state)
    except ValueError:
        return
    assert decoded is None


def test_q13_repair_uses_documented_true_path_cost_tiebreak() -> None:
    obj = instance([(3, 1), (0, -2), (-2, -4), (-4, -4)])
    sub = extract_reorder_subproblem(
        obj,
        [[0, 1, 2, 3]],
        Neighborhood("high_cost_route", "reorder", (0,), (0, 1, 2, 3), 1, (0, 4)),
    )
    model = build_reorder_qubo(sub)
    # Every permutation has the same agreement with an all-zero sample.
    repaired = repair_reorder(model, [0] * 16)
    optimum = min(sub.path_cost(p) for p in itertools.permutations(sub.customers))
    assert sub.path_cost(repaired) == pytest.approx(optimum)


def test_q13_annealing_supports_valid_constant_qubo() -> None:
    builder = QUBOBuilder([])
    builder.add_constant(2)
    samples = solve_simulated_annealing(builder.build(), num_reads=2, num_sweeps=2)
    assert len(samples) == 2
    assert all(s.x == () and s.energy == 2 for s in samples)


@pytest.mark.parametrize("operation", ["decode", "repair"])
@pytest.mark.parametrize("value", [-1, 2, 0.5])
def test_q14_decode_and_repair_reject_invalid_bits(operation: str, value: float) -> None:
    obj = instance([(1, 0), (-1, 0), (0, 1)])
    sub = extract_exchange_subproblem(
        obj, [[0], [1, 2]], Neighborhood("two_route_exchange", "exchange", (0, 1), (2,), 1)
    )
    model = build_exchange_qubo(obj, sub)
    function = decode_exchange if operation == "decode" else repair_exchange
    with pytest.raises(ValueError):
        function(model, (value,))


def test_q14_exact_slack_rejects_materially_fractional_large_demands() -> None:
    # math.isclose's default relative tolerance accidentally accepts a 0.25 fraction.
    obj = instance([(1, 0), (-1, 0)], [1_000_000_000.25, 1], 2_000_000_001)
    sub = extract_exchange_subproblem(
        obj, [[0], [1]], Neighborhood("two_route_exchange", "exchange", (0, 1), (0, 1), 1)
    )
    with pytest.raises(ValueError, match="integral"):
        build_exchange_qubo(obj, sub)


@pytest.mark.parametrize("dimension", [5, 7])
def test_q15_exchange_prior_shape_matches_instance(dimension: int) -> None:
    obj, _, _, sub = exchange_case()
    with pytest.raises(ValueError, match=r"shape|size"):
        build_biased_exchange_qubo(
            obj, sub, np.full((dimension, dimension), 0.5), DiffusionBiasConfig(True, 1)
        )


@pytest.mark.parametrize("method", ["exhaustive", "relocate_swap"])
def test_q16_exchange_reports_actual_supplied_starting_cost(method: str) -> None:
    obj = instance([(1, 0), (2, 0), (3, 0), (4, 0), (-1, 0)], capacity=4)
    routes = [[0, 2, 1, 3], [4]]
    sub = extract_exchange_subproblem(
        obj, routes, Neighborhood("two_route_exchange", "exchange", (0, 1), (4,), 1)
    )
    result = solve_exchange(obj, sub, method)
    assert result.cost_before == pytest.approx(route_cost(obj, routes))


def test_q16_singleton_rejects_unknown_solver_name() -> None:
    _, _, _, sub = reorder_case(1)
    with pytest.raises(ValueError, match="unknown"):
        solve_reorder(sub, "typo")


def test_q13_bqm_retains_zero_coefficient_variables() -> None:
    pytest.importorskip("dimod")
    obj = instance([(1, 0), (-1, 0), (0, 1)], capacity=5)
    sub = extract_exchange_subproblem(
        obj, [[0], [1, 2]], Neighborhood("two_route_exchange", "exchange", (0, 1), (2,), 1)
    )
    model = build_exchange_qubo(obj, sub)
    assert model.qubo.num_variables == 1 and model.qubo.num_terms == 0
    bqm = model.qubo.to_bqm()
    assert set(bqm.variables) == set(model.qubo.labels)
