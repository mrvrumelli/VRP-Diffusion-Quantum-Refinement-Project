"""Tests for the QAOA statevector solver and its random-sampling control."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from test_quantum_neighborhoods import make_instance
from test_quantum_refinement import misplaced_instance
from vrp_diffusion_quantum.local_search.baselines import solve_reorder
from vrp_diffusion_quantum.quantum.neighborhoods import Neighborhood, extract_reorder_subproblem
from vrp_diffusion_quantum.quantum.qaoa_solver import (
    QAOASolver,
    RandomQUBOSampler,
    interpolate_angles,
    optimise_qaoa,
    qaoa_probabilities,
    qubo_energies,
)
from vrp_diffusion_quantum.quantum.qubo import QUBO, QUBOBuilder
from vrp_diffusion_quantum.quantum.refinement import ClassicalSolver, refine_solution
from vrp_diffusion_quantum.utils.feasibility import validate_routes


def random_qubo(n: int, seed: int) -> QUBO:
    rng = np.random.default_rng(seed)
    builder = QUBOBuilder([f"x{i}" for i in range(n)])
    for i in range(n):
        builder.add_linear(i, float(rng.normal()))
        for j in range(i + 1, n):
            builder.add_quadratic(i, j, float(rng.normal()))
    return builder.build()


def test_energies_match_direct_evaluation_in_qiskit_bit_order() -> None:
    qubo = random_qubo(4, 0)
    energies = qubo_energies(qubo)
    for code in range(16):
        x = tuple((code >> i) & 1 for i in range(4))
        assert energies[code] == pytest.approx(qubo.energy(x))


def test_simulator_matches_dense_matrix_evolution() -> None:
    n, reps = 3, 2
    qubo = random_qubo(n, 1)
    energies = qubo_energies(qubo)
    theta = np.array([0.3, 1.1, 0.7, -0.4])
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]])
    mixer = np.zeros((1 << n, 1 << n))
    for qubit in range(n):
        ops = [np.eye(2)] * n
        ops[qubit] = x_gate
        term = ops[n - 1]
        for op in reversed(ops[: n - 1]):
            term = np.kron(term, op)
        mixer += term
    state = np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=complex)
    for beta, gamma in zip(theta[:reps], theta[reps:], strict=True):
        state = np.exp(-1j * gamma * energies) * state
        state = expm(-1j * beta * mixer) @ state
    assert qaoa_probabilities(theta, energies, n, reps) == pytest.approx(np.abs(state) ** 2)


def test_zero_angles_give_the_uniform_distribution() -> None:
    energies = qubo_energies(random_qubo(3, 2))
    assert qaoa_probabilities(np.zeros(2), energies, 3, 1) == pytest.approx(np.full(8, 1 / 8))


def test_interpolation_extends_a_schedule_by_one_layer() -> None:
    assert interpolate_angles([1.0]) == pytest.approx([1.0, 1.0])
    assert interpolate_angles([1.0, 2.0]) == pytest.approx([1.0, 1.5, 2.0])


def test_optimised_state_puts_more_weight_on_the_optimum_than_uniform() -> None:
    qubo = random_qubo(5, 3)
    energies = qubo_energies(qubo)
    result = optimise_qaoa(energies, 5, 2, restarts=3, rng=np.random.default_rng(0))
    probabilities = qaoa_probabilities(result.theta, energies, 5, 2)
    optimal = np.isclose(energies, energies.min())
    assert probabilities[optimal].sum() > optimal.mean()
    assert result.expectation < energies.mean()
    assert result.evaluations > 0


def test_qaoa_solver_finds_the_optimal_short_segment() -> None:
    rng = np.random.default_rng(4)
    instance = make_instance([tuple(p) for p in rng.random((6, 2))])
    neighborhood = Neighborhood("high_cost_route", "reorder", (0,), (1, 2, 3), 1.0, (1, 4))
    subproblem = extract_reorder_subproblem(instance, [[0, 1, 2, 3, 4, 5]], neighborhood)
    outcome = QAOASolver(reps=1, seed=0).solve_reorder(instance, subproblem, None)
    assert outcome.candidate_cost == pytest.approx(solve_reorder(subproblem, "exact").cost_after)
    assert outcome.qubo_num_variables == 9
    assert 1 <= outcome.num_samples <= 50
    assert outcome.solver_name == "qaoa_p1"


def test_subproblems_above_the_qubit_limit_are_skipped_unchanged() -> None:
    rng = np.random.default_rng(5)
    instance = make_instance([tuple(p) for p in rng.random((6, 2))])
    neighborhood = Neighborhood("high_cost_route", "reorder", (0,), (0, 1, 2, 3, 4), 1.0, (0, 5))
    subproblem = extract_reorder_subproblem(instance, [[0, 1, 2, 3, 4, 5]], neighborhood)
    for solver in (QAOASolver(max_qubits=16), RandomQUBOSampler(max_qubits=16)):
        outcome = solver.solve_reorder(instance, subproblem, None)
        assert outcome.solver_name.endswith("_skipped")
        assert outcome.order == subproblem.customers
        assert outcome.candidate_cost == pytest.approx(subproblem.current_cost)
        assert outcome.num_samples == 0


@pytest.mark.parametrize("solver", [QAOASolver(reps=1, restarts=2), RandomQUBOSampler()])
def test_sampling_solvers_refine_without_worsening(solver: object) -> None:
    instance, routes = misplaced_instance()
    trace = refine_solution(instance, routes, solver, max_rounds=2)  # type: ignore[arg-type]
    assert validate_routes(instance, trace.routes).feasible
    assert trace.final_cost <= trace.initial_cost + 1e-12
    baseline = refine_solution(
        instance, routes, ClassicalSolver("exact", "exhaustive"), max_rounds=2
    )
    assert trace.final_cost >= baseline.final_cost - 1e-9 or trace.improvement > 0


def test_random_sampler_is_deterministic_and_keeps_distinct_low_energy_states() -> None:
    qubo = random_qubo(6, 6)
    sampler = RandomQUBOSampler(shots=200, keep=10, seed=3)
    first, second = sampler._sample(qubo), sampler._sample(qubo)
    assert first == second
    assert len({s.x for s in first}) == len(first) <= 10
    assert [s.energy for s in first] == sorted(s.energy for s in first)
    all_states = sorted(qubo.energy(x) for x in itertools.product((0, 1), repeat=6))
    assert first[0].energy >= all_states[0] - 1e-12
