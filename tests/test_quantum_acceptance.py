"""Regression coverage for the eleven findings in the 2026-10-01 quantum audit."""

from __future__ import annotations

import itertools
from dataclasses import replace

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
    build_biased_reorder_qubo,
    exchange_bias_terms,
    reorder_bias_terms,
)
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    build_exchange_qubo,
    decode_exchange,
    repair_exchange,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import (
    MAX_EXACT_REPAIR_SIZE,
    build_reorder_qubo,
    decode_reorder,
    encode_reorder,
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


@pytest.mark.parametrize("change", ["start", "end", "shorter", "missing", "negative_index"])
def test_q12_apply_reorder_checks_boundaries_and_route_index(change: str) -> None:
    _, routes, _, sub = reorder_case(2)
    if change == "start":
        routes[0][0] = 3
    elif change == "end":
        routes[0][-1] = 0
    elif change == "shorter":
        routes[0] = routes[0][:2]
    elif change == "missing":
        routes = []
    else:
        sub = replace(sub, route_index=-1)
    with pytest.raises(ValueError, match="reselect"):
        apply_reorder(routes, sub, sub.customers)


@pytest.mark.parametrize("snapshot", [True, False])
def test_q12_apply_exchange_rejects_stale_route_members(snapshot: bool) -> None:
    _, routes, _, sub = exchange_case()
    if not snapshot:
        sub = replace(sub, initial_routes=None)
    routes[0][0], routes[1][0] = routes[1][0], routes[0][0]
    with pytest.raises(ValueError, match="no longer matches"):
        apply_exchange(routes, sub, *sub.members(sub.initial_assignment))


def test_q12_low_confidence_keeps_weakest_overlapping_window() -> None:
    obj = instance([(i, 0) for i in range(6)])
    matrix = np.full((6, 6), 0.4)
    matrix[2, 3] = matrix[3, 2] = 0.05
    selected = select_low_confidence_edges(
        obj, [list(range(6))], matrix, NeighborhoodConfig(max_reorder_size=3)
    )
    assert [n.segment for n in selected] == [(2, 5)]


@pytest.mark.parametrize("bad", [[], [0, 1], [[0]], [np.nan], [np.inf], [-np.inf]])
def test_q12_q14_binary_shape_and_nonfinite_validation(bad: list) -> None:
    obj = instance([(1, 0), (-1, 0), (0, 1)])
    sub = extract_exchange_subproblem(
        obj, [[0], [1, 2]], Neighborhood("two_route_exchange", "exchange", (0, 1), (2,), 1)
    )
    model = build_exchange_qubo(obj, sub)
    for operation in (sub.members, sub.loads, sub.is_feasible):
        with pytest.raises(ValueError):
            operation(bad)
    for operation in (decode_exchange, repair_exchange):
        with pytest.raises(ValueError):
            operation(model, bad)


def test_q14_decode_validates_slack_bits_too() -> None:
    obj, _, _, sub = exchange_case(capacity=5)
    model = build_exchange_qubo(obj, sub)
    assert model.qubo.num_variables > sub.size
    bits = [*sub.initial_assignment, *([0] * (model.qubo.num_variables - sub.size))]
    bits[-1] = -1
    with pytest.raises(ValueError, match="binary"):
        decode_exchange(model, bits)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q13_repair_full_cost_priority_over_all_three_customer_states(seed: int) -> None:
    _, _, _, sub = reorder_case(3, seed)
    model = build_reorder_qubo(sub)
    permutations = list(itertools.permutations(sub.customers))
    encoded = [encode_reorder(model, p) for p in permutations]
    for bits in itertools.product((0, 1), repeat=9):
        priorities = [
            (-int(np.dot(bits, state)), sub.path_cost(p))
            for p, state in zip(permutations, encoded, strict=True)
        ]
        expected_agreement, expected_cost = min(priorities)
        actual = repair_reorder(model, bits)
        assert -np.dot(bits, encode_reorder(model, actual)) == expected_agreement
        assert sub.path_cost(actual) == pytest.approx(expected_cost)


def test_q13_repair_matches_exact_path_at_documented_heuristic_size() -> None:
    _, _, _, sub = reorder_case(10, 2)
    model = build_reorder_qubo(sub)
    repaired = repair_reorder(model, [0] * 100)
    assert sub.path_cost(repaired) == pytest.approx(solve_reorder(sub, "exact").cost_after)


def test_q13_repair_limit_keeps_valid_larger_permutations() -> None:
    _, _, _, sub = reorder_case(MAX_EXACT_REPAIR_SIZE + 1)
    model = build_reorder_qubo(sub)
    with pytest.raises(ValueError, match="repair is limited"):
        repair_reorder(model, [0] * model.qubo.num_variables)
    assert repair_reorder(model, encode_reorder(model, sub.customers)) == sub.customers


@pytest.mark.parametrize("dimension", [4, 6])
def test_q15_reorder_prior_shape_matches_extracted_instance(dimension: int) -> None:
    _, _, _, sub = reorder_case(3)
    config = DiffusionBiasConfig(True, 1)
    matrix = np.full((dimension, dimension), 0.5)
    for operation in (
        lambda: build_biased_reorder_qubo(sub, matrix, config),
        lambda: reorder_bias_terms(build_reorder_qubo(sub), matrix, config),
    ):
        with pytest.raises(ValueError, match="shape"):
            operation()


def test_q15_hand_built_subproblems_validate_referenced_indices() -> None:
    config = DiffusionBiasConfig(True, 1)
    _, _, _, sub = reorder_case(3)
    sub = replace(sub, n_customers=None)
    with pytest.raises(ValueError, match="shape"):
        reorder_bias_terms(build_reorder_qubo(sub), np.full((4, 4), 0.5), config)
    obj, _, _, exchange_sub = exchange_case()
    exchange_sub = replace(exchange_sub, n_customers=None)
    model = build_exchange_qubo(obj, exchange_sub)
    with pytest.raises(ValueError, match="shape"):
        exchange_bias_terms(model, np.full((5, 5), 0.5), config)
    with pytest.raises(ValueError, match="shape"):
        build_biased_exchange_qubo(obj, exchange_sub, np.full((7, 7), 0.5), config)


def test_q16_exchange_snapshot_is_immutable_and_improvement_includes_polish() -> None:
    obj = instance([(1, 0), (2, 0), (3, 0), (4, 0), (-1, 0)], capacity=4)
    routes = [[0, 2, 1, 3], [4]]
    sub = extract_exchange_subproblem(
        obj, routes, Neighborhood("two_route_exchange", "exchange", (0, 1), (4,), 1)
    )
    routes[0][:] = [0, 1, 2, 3]
    result = solve_exchange(obj, sub, "exhaustive")
    assert result.cost_before == 12
    assert result.cost_after == 10 and result.improvement == 2
    # Legacy manually constructed objects have no original ordering to recover.
    legacy = solve_exchange(obj, replace(sub, initial_routes=None), "exhaustive")
    assert legacy.cost_before == 10 and legacy.improvement == 0
