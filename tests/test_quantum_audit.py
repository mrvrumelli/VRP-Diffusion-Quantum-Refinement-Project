"""Independent Q1.2-Q1.6 formula, boundary, and integration checks."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.local_search.baselines import refine, solve_exchange, solve_reorder
from vrp_diffusion_quantum.quantum.neighborhoods import (
    ExchangeSubproblem,
    Neighborhood,
    NeighborhoodConfig,
    ReorderSubproblem,
    apply_exchange,
    apply_reorder,
    evaluate_exchange,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
    select_neighborhoods,
)
from vrp_diffusion_quantum.quantum.qubo import (
    QUBO,
    QUBOBuilder,
    solve_exact,
    solve_simulated_annealing,
)
from vrp_diffusion_quantum.quantum.qubo_bias import (
    DiffusionBiasConfig,
    build_biased_exchange_qubo,
    build_biased_reorder_qubo,
    exchange_bias_terms,
    reorder_bias_terms,
)
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    bounded_slack_coefficients,
    build_exchange_qubo,
    decode_exchange,
    repair_exchange,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import (
    build_reorder_qubo,
    decode_reorder,
    encode_reorder,
    repair_reorder,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes


def instance(
    xy: list[tuple[float, float]] | np.ndarray,
    demands: list[float] | None = None,
    capacity: float = 100,
    depot_index: int = 0,
) -> CVRPInstance:
    points = list(xy)
    loads = list(demands) if demands is not None else [1] * len(points)
    points.insert(depot_index, (0.0, 0.0))
    loads.insert(depot_index, 0)
    return CVRPInstance(
        np.asarray(points, dtype=float),
        np.asarray(loads, dtype=float),
        float(capacity),
        depot_index,
        "independent-audit",
        len(xy),
        0,
        {},
    )


def states(n: int) -> np.ndarray:
    codes = np.arange(1 << n, dtype=np.int64)
    return ((codes[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(float)


def reorder_case(
    k: int,
    seed: int = 0,
    scale: float = 1.0,
) -> tuple[CVRPInstance, list[list[int]], Neighborhood, ReorderSubproblem]:
    rng = np.random.default_rng(seed)
    obj = instance(rng.random((k + 2, 2)) * scale, depot_index=seed % (k + 3))
    routes = [list(range(k + 2))]
    n = Neighborhood("high_cost_route", "reorder", (0,), tuple(range(1, k + 1)), 1, (1, k + 1))
    return obj, routes, n, extract_reorder_subproblem(obj, routes, n)


def exchange_case(
    seed: int = 0,
    capacity: float = 6,
) -> tuple[CVRPInstance, list[list[int]], Neighborhood, ExchangeSubproblem]:
    rng = np.random.default_rng(seed)
    obj = instance(rng.random((6, 2)), [1, 2, 2, 1, 2, 2], capacity, seed % 7)
    routes = [[0, 1, 2], [3, 4, 5]]
    n = Neighborhood("two_route_exchange", "exchange", (0, 1), (1, 2, 4, 5), 1)
    return obj, routes, n, extract_exchange_subproblem(obj, routes, n)


@pytest.mark.parametrize("seed", range(12))
def test_q12_select_extract_apply_invariants(seed: int) -> None:
    rng = np.random.default_rng(seed)
    obj = instance(rng.random((10, 2)), capacity=6, depot_index=seed % 11)
    routes = [[0, 1, 2, 3], [], [4, 5, 6], [7, 8, 9]]
    original = [r.copy() for r in routes]
    matrix = rng.random((10, 10))
    matrix = (matrix + matrix.T) / 2
    cfg = NeighborhoodConfig(max_reorder_size=3, max_exchange_size=4, max_per_type=3)
    selected = select_neighborhoods(obj, routes, m_prob=matrix, config=cfg)
    assert selected == select_neighborhoods(obj, routes, m_prob=matrix, config=cfg)
    for n in selected:
        assert sum(x.neighborhood_type == n.neighborhood_type for x in selected) <= 3
        if n.kind == "reorder":
            sub = extract_reorder_subproblem(obj, routes, n)
            assert 2 <= sub.size <= 3
            for order in itertools.permutations(sub.customers):
                updated = apply_reorder(routes, sub, order)
                assert validate_routes(obj, updated).feasible
                assert route_cost(obj, updated) - route_cost(obj, routes) == pytest.approx(
                    sub.path_cost(order) - sub.current_cost
                )
        else:
            sub = extract_exchange_subproblem(obj, routes, n)
            assert 1 <= sub.size <= 4
            for assignment in itertools.product((0, 1), repeat=sub.size):
                first, second = sub.members(assignment)
                expected = tuple(sum(obj.customer_demands()[r]) for r in (first, second))
                assert sub.loads(assignment) == pytest.approx(expected)
                assert sub.is_feasible(assignment) == (max(expected) <= obj.capacity)
                cost, ordered_first, ordered_second = evaluate_exchange(obj, sub, assignment)
                updated = apply_exchange(routes, sub, ordered_first, ordered_second)
                assert sorted(itertools.chain.from_iterable(updated)) == list(range(10))
                assert set(sub.fixed[0]) <= set(ordered_first)
                assert set(sub.fixed[1]) <= set(ordered_second)
                assert cost == pytest.approx(route_cost(obj, [ordered_first, ordered_second]))
    assert routes == original


@pytest.mark.parametrize("seed", range(8))
def test_q13_generic_core_against_scalar_energy(seed: int) -> None:
    rng = np.random.default_rng(seed)
    matrix = np.triu(rng.normal(size=(8, 8)))
    qubo = QUBO(matrix, 1.2, tuple(map(str, range(8))))
    all_states = states(8)
    expected = np.asarray(
        [
            1.2 + sum(matrix[i, j] * x[i] * x[j] for i in range(8) for j in range(i, 8))
            for x in all_states
        ]
    )
    np.testing.assert_allclose(qubo.energies(all_states), expected, atol=1e-12)
    exact = solve_exact(qubo)
    assert exact.energy == pytest.approx(min(expected))
    samples = solve_simulated_annealing(qubo, num_reads=4, num_sweeps=80, seed=seed)
    assert samples == solve_simulated_annealing(qubo, num_reads=4, num_sweeps=80, seed=seed)
    assert samples == sorted(samples, key=lambda sample: (sample.energy, sample.x))
    for sample in samples:
        assert set(sample.x) <= {0, 1}
        assert sample.energy == pytest.approx(qubo.energy(sample.x))


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(4))
def test_q13_reorder_full_binary_landscape(k: int, seed: int) -> None:
    _, _, _, sub = reorder_case(k, seed, [0, 0.01, 1, 1000][seed])
    model = build_reorder_qubo(sub)
    all_states = states(k * k)
    grids = all_states.reshape(-1, k, k)
    valid = (grids.sum(axis=1) == 1).all(axis=1) & (grids.sum(axis=2) == 1).all(axis=1)
    energies = model.qubo.energies(all_states)
    optimum = min(sub.path_cost(order) for order in itertools.permutations(sub.customers))
    assert energies[valid].min() == pytest.approx(optimum, abs=1e-8)
    assert energies[~valid].min() > energies[valid].min()
    for order in itertools.permutations(sub.customers):
        encoded = encode_reorder(model, order)
        assert decode_reorder(model, encoded) == order
        assert model.qubo.energy(encoded) == pytest.approx(sub.path_cost(order), abs=1e-8)
    rng = np.random.default_rng(seed)
    for x in rng.integers(0, 2, (12, k * k)):
        repaired = repair_reorder(model, x)
        assert sorted(repaired) == sorted(sub.customers)
        agreement = np.dot(x, encode_reorder(model, repaired))
        assert agreement == max(
            np.dot(x, encode_reorder(model, p)) for p in itertools.permutations(sub.customers)
        )


def test_q13_exact_limit_and_zero_variable_exact() -> None:
    assert solve_exact(QUBOBuilder([]).build()).x == ()
    with pytest.raises(ValueError, match="exhaustive limit"):
        solve_exact(QUBOBuilder([str(i) for i in range(23)]).build())


def test_q14_slack_representability_zero_through_128() -> None:
    for upper in range(129):
        coefficients = bounded_slack_coefficients(upper)
        represented = {
            sum(c * bit for c, bit in zip(coefficients, x, strict=True))
            for x in itertools.product((0, 1), repeat=len(coefficients))
        }
        assert represented == set(range(upper + 1))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("capacity", [5, 6, 10])
def test_q14_energy_equals_surrogate_plus_exact_capacity_violation(
    seed: int,
    capacity: int,
) -> None:
    obj, _, _, sub = exchange_case(seed, capacity)
    model = build_exchange_qubo(obj, sub)
    all_states = states(model.qubo.num_variables)
    energies = model.qubo.energies(all_states)
    penalty = model.qubo.penalty_weights["capacity"]
    feasible_costs, infeasible_minima = [], []
    for assignment in itertools.product((0, 1), repeat=sub.size):
        mask = (all_states[:, : sub.size] == assignment).all(axis=1)
        minimum = energies[mask].min()
        loads = sub.loads(assignment)
        expected = model.surrogate_cost(assignment) + penalty * sum(
            max(0, load - capacity) ** 2 for load in loads
        )
        assert minimum == pytest.approx(expected, abs=1e-8)
        if sub.is_feasible(assignment):
            feasible_costs.append(model.surrogate_cost(assignment))
            assert repair_exchange(model, assignment) == assignment
        else:
            infeasible_minima.append(minimum)
        repaired = repair_exchange(model, assignment)
        assert repaired is None or sub.is_feasible(repaired)
    if infeasible_minima:
        assert min(infeasible_minima) > max(feasible_costs)
    assert sub.is_feasible(decode_exchange(model, solve_exact(model.qubo).x))


@pytest.mark.parametrize(
    "case", ["zero_remaining", "first_only", "second_only", "zero_demands", "infeasible"]
)
def test_q14_capacity_boundary_variants(case: str) -> None:
    variants = {
        "zero_remaining": ([4, 1, 1, 0], 4, (1, 2)),
        "first_only": ([3, 1, 1, 0], 4, (1, 2)),
        "second_only": ([0, 1, 1, 3], 4, (1, 2)),
        "zero_demands": ([0, 0, 0, 0], 1, (1, 2)),
        "infeasible": ([0, 3, 3, 0], 2, (1, 2)),
    }
    demands, capacity, movable = variants[case]
    obj = instance([(1, 0), (2, 0), (-1, 0), (-2, 0)], demands, capacity)
    sub = extract_exchange_subproblem(
        obj, [[0, 1], [2, 3]], Neighborhood("two_route_exchange", "exchange", (0, 1), movable, 1)
    )
    model = build_exchange_qubo(obj, sub)
    for assignment in itertools.product((0, 1), repeat=2):
        fixed = [x for x in states(model.qubo.num_variables) if tuple(x[:2]) == assignment]
        expected = model.surrogate_cost(assignment) + model.qubo.penalty_weights["capacity"] * sum(
            max(0, load - capacity) ** 2 for load in sub.loads(assignment)
        )
        assert min(model.qubo.energies(fixed)) == pytest.approx(expected, abs=1e-8)
    if case == "infeasible":
        assert all(repair_exchange(model, a) is None for a in itertools.product((0, 1), repeat=2))


def prior_cost(p: float, mode: str, signed: bool = False) -> float:
    value = 1 - (2 if signed else 1) * p
    return value * (abs(2 * p - 1) if mode == "confidence" else 1)


@pytest.mark.parametrize("mode", ["probability", "confidence"])
@pytest.mark.parametrize("alpha", [0.01, 1, 100])
@pytest.mark.parametrize("seed", range(3))
def test_q15_bias_matches_independent_formulas_and_preserves_feasibility(
    mode: str,
    alpha: float,
    seed: int,
) -> None:
    obj, _, _, sub = reorder_case(3, seed)
    rng = np.random.default_rng(seed)
    p = rng.choice([0.0, 0.1, 0.5, 0.9, 1.0], size=(obj.n_customers, obj.n_customers))
    p = (p + p.T) / 2
    cfg = DiffusionBiasConfig(True, alpha, mode)
    reorder = build_biased_reorder_qubo(sub, p, cfg)
    positive = sub.distances[sub.distances > 0]
    scale = positive.mean() if positive.size else 1
    for order in itertools.permutations(sub.customers):
        path = [sub.start_node, *order, sub.end_node]
        expected = sub.path_cost(order) + alpha * scale * sum(
            prior_cost(p[a, b], mode) for a, b in itertools.pairwise(path)
        )
        assert reorder.qubo.energy(encode_reorder(reorder, order)) == pytest.approx(expected)
    assert decode_reorder(reorder, solve_exact(reorder.qubo).x) is not None

    obj, _, _, sub = exchange_case(seed, 5)
    p = rng.choice([0.0, 0.5, 1.0], size=(6, 6))
    p = (p + p.T) / 2
    exchange = build_biased_exchange_qubo(obj, sub, p, cfg)
    positive = exchange.insertion_costs[exchange.insertion_costs > 0]
    scale = positive.mean() if positive.size else 1
    all_states = states(exchange.qubo.num_variables)
    energies = exchange.qubo.energies(all_states)
    bias_only = exchange_bias_terms(exchange, p, cfg)
    feasible_costs, infeasible_minima = [], []
    for assignment in itertools.product((0, 1), repeat=sub.size):
        bias = 0.0
        for i, customer in enumerate(sub.movable):
            fixed = sub.fixed[0 if assignment[i] else 1]
            affinity = np.mean([p[customer, c] for c in fixed]) if fixed else 0.5
            bias += prior_cost(affinity, mode)
            for j in range(i + 1, sub.size):
                if assignment[i] == assignment[j]:
                    bias += prior_cost(p[customer, sub.movable[j]], mode, signed=True)
        mask = (all_states[:, : sub.size] == assignment).all(axis=1)
        expected_bias = bias * alpha * scale
        np.testing.assert_allclose(bias_only.energies(all_states[mask]), expected_bias, atol=1e-9)
        expected = exchange.surrogate_cost(assignment) + expected_bias
        if sub.is_feasible(assignment):
            assert energies[mask].min() == pytest.approx(expected, abs=1e-8)
            feasible_costs.append(expected)
        else:
            infeasible_minima.append(energies[mask].min())
    assert min(infeasible_minima) > max(feasible_costs)
    assert sub.is_feasible(decode_exchange(exchange, solve_exact(exchange.qubo).x))


@pytest.mark.parametrize("mode", ["probability", "confidence"])
def test_q15_disabled_is_identical_and_uncertain_confidence_is_zero(mode: str) -> None:
    _, _, _, sub = reorder_case(3)
    plain = build_reorder_qubo(sub)
    obj, _, _, exchange_sub = exchange_case()
    plain_exchange = build_exchange_qubo(obj, exchange_sub)
    for cfg in [DiffusionBiasConfig(False, 4, mode), DiffusionBiasConfig(True, 0, mode)]:
        off = build_biased_reorder_qubo(sub, None, cfg)
        assert off.qubo.offset == plain.qubo.offset
        np.testing.assert_array_equal(off.qubo.matrix, plain.qubo.matrix)
        off_exchange = build_biased_exchange_qubo(obj, exchange_sub, None, cfg)
        assert off_exchange.qubo.offset == plain_exchange.qubo.offset
        np.testing.assert_array_equal(off_exchange.qubo.matrix, plain_exchange.qubo.matrix)
    cfg = DiffusionBiasConfig(True, 2, "confidence")
    assert reorder_bias_terms(plain, np.full((5, 5), 0.5), cfg).num_terms == 0
    terms = exchange_bias_terms(plain_exchange, np.full((6, 6), 0.5), cfg)
    assert terms.num_terms == 0 and terms.offset == 0


@pytest.mark.parametrize("seed", range(12))
def test_q16_exact_local_and_refine_against_independent_enumeration(seed: int) -> None:
    obj, routes, n, sub = reorder_case(6, seed)
    optimum = min(sub.path_cost(p) for p in itertools.permutations(sub.customers))
    exact = solve_reorder(sub, "exact")
    local = solve_reorder(sub, "two_opt")
    assert exact.cost_after == pytest.approx(optimum)
    assert optimum - 1e-9 <= local.cost_after <= sub.current_cost + 1e-9
    for method in ["exact", "two_opt"]:
        result = refine(obj, routes, n, method)
        assert validate_routes(obj, result.routes).feasible
        assert result.cost_after == pytest.approx(route_cost(obj, result.routes))
        assert result.cost_after <= route_cost(obj, routes) + 1e-9
    obj, routes, n, sub = exchange_case(seed, 5)
    all_assignments = [a for a in itertools.product((0, 1), repeat=sub.size) if sub.is_feasible(a)]
    # Evaluate independently of solve_exchange's private cache/search implementation.
    optimum = min(evaluate_exchange(obj, sub, a)[0] for a in all_assignments)
    exact = solve_exchange(obj, sub, "exhaustive")
    local = solve_exchange(obj, sub, "relocate_swap")
    assert exact.feasible and local.feasible
    assert exact.cost_after == pytest.approx(optimum)
    assert optimum - 1e-9 <= local.cost_after <= local.cost_before + 1e-9
    for method in ["exhaustive", "relocate_swap"]:
        result = refine(obj, routes, n, method)
        assert validate_routes(obj, result.routes).feasible
        assert result.cost_after == pytest.approx(route_cost(obj, result.routes))
        assert result.cost_after <= route_cost(obj, routes) + 1e-9


def test_q16_refine_rejects_cheaper_overcapacity_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    import vrp_diffusion_quantum.local_search.baselines as baselines

    obj = instance([(1, 0), (2, 0), (3, 0)], capacity=2)
    routes = [[0, 1], [2]]
    n = Neighborhood("two_route_exchange", "exchange", (0, 1), (0, 1, 2), 1)
    sub = extract_exchange_subproblem(obj, routes, n)
    feasible_solution = solve_exchange(obj, sub, "exhaustive")
    bad_cost, first, second = evaluate_exchange(obj, sub, (1, 1, 1))
    assert bad_cost < route_cost(obj, routes)
    bad = replace(
        feasible_solution, assignment=(1, 1, 1), ordered_routes=(tuple(first), tuple(second))
    )
    monkeypatch.setattr(baselines, "solve_exchange", lambda *args: bad)
    result = refine(obj, routes, n, "exhaustive")
    assert not result.accepted and not result.post_repair_feasible
    assert result.routes == routes


def test_q16_exact_size_limits() -> None:
    _, _, _, sub = reorder_case(13)
    with pytest.raises(ValueError, match="limited"):
        solve_reorder(sub, "exact")
    obj = instance([(i, 0) for i in range(15)])
    routes = [list(range(7)), list(range(7, 15))]
    sub = extract_exchange_subproblem(
        obj, routes, Neighborhood("two_route_exchange", "exchange", (0, 1), tuple(range(15)), 1)
    )
    with pytest.raises(ValueError, match="limited"):
        solve_exchange(obj, sub, "exhaustive")


def test_q13_bqm_nonzero_energy_round_trip() -> None:
    dimod = pytest.importorskip("dimod")
    builder = QUBOBuilder(["a", "b", "c"])
    builder.add_linear(0, -2)
    builder.add_quadratic(1, 2, 3)
    builder.add_constant(1)
    qubo = builder.build()
    bqm = qubo.to_bqm()
    assert set(bqm.variables) == set(qubo.labels)
    for x in itertools.product((0, 1), repeat=3):
        assert bqm.energy(dict(zip(qubo.labels, x, strict=True))) == pytest.approx(qubo.energy(x))
    assert dimod.ExactSolver().sample(bqm).first.energy == pytest.approx(solve_exact(qubo).energy)


@pytest.mark.parametrize("seed", range(8))
def test_integration_anneal_decode_repair_and_matched_classical_reference(seed: int) -> None:
    obj, routes, _, sub = reorder_case(3, seed)
    model = build_reorder_qubo(sub)
    reference = solve_reorder(sub, "exact")
    for sample in solve_simulated_annealing(model.qubo, num_reads=3, num_sweeps=50, seed=seed):
        order = repair_reorder(model, sample.x)
        candidate = apply_reorder(routes, sub, order)
        assert validate_routes(obj, candidate).feasible
        assert sub.path_cost(order) >= reference.cost_after - 1e-9
        assert route_cost(obj, candidate) - route_cost(obj, routes) == pytest.approx(
            sub.path_cost(order) - sub.current_cost
        )
    obj, routes, _, sub = exchange_case(seed, 5)
    model = build_exchange_qubo(obj, sub)
    reference = solve_exchange(obj, sub, "exhaustive")
    for sample in solve_simulated_annealing(model.qubo, num_reads=3, num_sweeps=50, seed=seed):
        assignment = repair_exchange(model, decode_exchange(model, sample.x))
        if assignment is None:
            continue  # Failure to repair is a documented heuristic outcome.
        cost, first, second = evaluate_exchange(obj, sub, assignment)
        candidate = apply_exchange(routes, sub, first, second)
        assert validate_routes(obj, candidate).feasible
        assert cost >= reference.cost_after - 1e-9
        assert cost == pytest.approx(route_cost(obj, candidate))
