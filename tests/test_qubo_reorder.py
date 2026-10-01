"""Tests for the QUBO core and the route-reordering QUBO (task Q1.3)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from test_quantum_neighborhoods import make_instance
from vrp_diffusion_quantum.quantum.neighborhoods import (
    Neighborhood,
    ReorderSubproblem,
    extract_reorder_subproblem,
)
from vrp_diffusion_quantum.quantum.qubo import (
    QUBO,
    QUBOBuilder,
    solve_exact,
    solve_simulated_annealing,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import (
    build_reorder_qubo,
    decode_reorder,
    encode_reorder,
    repair_reorder,
)


def all_states(n: int) -> np.ndarray:
    return np.asarray(list(itertools.product((0, 1), repeat=n)), dtype=np.float64)


def test_squared_penalty_expansion_matches_direct_evaluation() -> None:
    builder = QUBOBuilder(["a", "b", "c", "d"])
    coefficients = {0: 2.0, 1: -1.0, 3: 3.5}
    builder.add_squared_penalty(coefficients, 2.5, 1.7)
    builder.add_linear(2, -0.4)
    qubo = builder.build()
    for state in all_states(4):
        lhs = sum(a * state[i] for i, a in coefficients.items())
        expected = 1.7 * (lhs - 2.5) ** 2 - 0.4 * state[2]
        assert qubo.energy(state) == pytest.approx(expected)


def test_qubo_rejects_lower_triangle_and_mismatched_addition() -> None:
    with pytest.raises(ValueError, match="upper triangular"):
        QUBO(np.array([[0.0, 0.0], [1.0, 0.0]]), 0.0, ("a", "b"))
    first = QUBO(np.zeros((1, 1)), 0.0, ("a",))
    with pytest.raises(ValueError, match="identical variable labels"):
        _ = first + QUBO(np.zeros((1, 1)), 0.0, ("b",))


def test_exact_and_annealing_solvers_find_the_brute_force_minimum() -> None:
    rng = np.random.default_rng(1)
    matrix = np.triu(rng.normal(size=(10, 10)))
    qubo = QUBO(matrix, 0.3, tuple(f"v{i}" for i in range(10)))
    energies = qubo.energies(all_states(10))
    exact = solve_exact(qubo)
    assert exact.energy == pytest.approx(energies.min())
    best = solve_simulated_annealing(qubo, num_reads=8, num_sweeps=300, seed=7)[0]
    assert best.energy == pytest.approx(energies.min())
    assert solve_simulated_annealing(qubo, num_reads=2, num_sweeps=50, seed=3) == (
        solve_simulated_annealing(qubo, num_reads=2, num_sweeps=50, seed=3)
    )


def test_to_dicts_round_trip() -> None:
    builder = QUBOBuilder(["a", "b"])
    builder.add_linear(0, 1.5)
    builder.add_quadratic(1, 0, -2.0)
    builder.add_constant(0.25)
    linear, quadratic, offset = builder.build().to_dicts()
    assert linear == {"a": 1.5}
    assert quadratic == {("a", "b"): -2.0}
    assert offset == 0.25


def line_subproblem(order: tuple[int, ...]) -> ReorderSubproblem:
    # Depot at x = 0; customers on a line at x = 1..4. Optimal open order is increasing x.
    instance = make_instance([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    routes = [list(order)]
    neighborhood = Neighborhood("high_cost_route", "reorder", (0,), order[:3], 1.0, (0, 3))
    return extract_reorder_subproblem(instance, routes, neighborhood)


def test_permutation_energy_equals_path_cost_for_every_order() -> None:
    subproblem = line_subproblem((2, 0, 1, 3))
    reorder = build_reorder_qubo(subproblem)
    assert reorder.qubo.num_variables == 9
    for order in itertools.permutations(reorder.subproblem.customers):
        x = encode_reorder(reorder, order)
        assert decode_reorder(reorder, x) == order
        assert reorder.qubo.energy(x) == pytest.approx(reorder.subproblem.path_cost(order))


def test_line_example_ground_state_is_the_sorted_order() -> None:
    # Segment (2, 0, 1) between the depot and customer 3: best is 0, 1, 2 (x = 1, 2, 3).
    subproblem = line_subproblem((2, 0, 1, 3))
    reorder = build_reorder_qubo(subproblem)
    best = solve_exact(reorder.qubo)
    assert decode_reorder(reorder, best.x) == (0, 1, 2)
    assert best.energy == pytest.approx(4.0)  # depot -> 1 -> 2 -> 3 -> customer 3 at x = 4


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ground_state_is_a_valid_optimal_permutation(k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    instance = make_instance([tuple(p) for p in rng.random((k + 2, 2))])
    route = list(range(k + 2))
    neighborhood = Neighborhood(
        "high_cost_route", "reorder", (0,), tuple(route[1 : k + 1]), 1.0, (1, k + 1)
    )
    subproblem = extract_reorder_subproblem(instance, [route], neighborhood)
    reorder = build_reorder_qubo(subproblem)
    best = solve_exact(reorder.qubo)
    order = decode_reorder(reorder, best.x)
    assert order is not None
    optimum = min(subproblem.path_cost(p) for p in itertools.permutations(subproblem.customers))
    assert subproblem.path_cost(order) == pytest.approx(optimum)
    assert best.energy == pytest.approx(optimum)
    # Every non-permutation state is strictly above the optimum.
    energies = reorder.qubo.energies(all_states(k * k))
    invalid = [
        energy
        for state, energy in zip(all_states(k * k), energies, strict=True)
        if decode_reorder(reorder, state.astype(int).tolist()) is None
    ]
    assert min(invalid) > optimum + 1e-9


def test_repair_returns_a_permutation_and_keeps_valid_states() -> None:
    subproblem = line_subproblem((2, 0, 1, 3))
    reorder = build_reorder_qubo(subproblem)
    valid = encode_reorder(reorder, (1, 0, 2))
    assert repair_reorder(reorder, valid) == (1, 0, 2)
    broken = [0] * 9
    broken[reorder.variable(0, 0)] = 1
    broken[reorder.variable(1, 0)] = 1  # two customers claim position 0
    repaired = repair_reorder(reorder, broken)
    assert sorted(repaired) == sorted(reorder.subproblem.customers)
    assert repaired[0] in (2, 0)


def test_penalty_weight_is_recorded_and_validated() -> None:
    subproblem = line_subproblem((2, 0, 1, 3))
    reorder = build_reorder_qubo(subproblem, penalty_weight=10.0)
    assert reorder.qubo.penalty_weights == {"permutation": 10.0}
    with pytest.raises(ValueError, match="penalty_weight"):
        build_reorder_qubo(subproblem, penalty_weight=0.0)
