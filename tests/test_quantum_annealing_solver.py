"""Tests for simulated quantum annealing on QUBOs and as a refinement solver."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from test_quantum_qaoa_solver import random_qubo
from test_quantum_refinement import misplaced_instance
from vrp_diffusion_quantum.quantum.annealing_solver import (
    SimulatedQuantumAnnealingSolver,
    solve_simulated_quantum_annealing,
)
from vrp_diffusion_quantum.quantum.qubo_bias import DiffusionBiasConfig
from vrp_diffusion_quantum.quantum.refinement import refine_solution
from vrp_diffusion_quantum.utils.feasibility import validate_routes


def test_sqa_finds_the_brute_force_minimum_of_small_qubos() -> None:
    for seed in range(3):
        qubo = random_qubo(8, 10 + seed)
        best = min(qubo.energy(x) for x in itertools.product((0, 1), repeat=8))
        samples = solve_simulated_quantum_annealing(qubo, num_reads=8, num_sweeps=60, seed=seed)
        assert samples[0].energy == pytest.approx(best)
        assert [s.energy for s in samples] == sorted(s.energy for s in samples)
        assert all(s.energy == pytest.approx(qubo.energy(s.x)) for s in samples)


def test_sqa_is_deterministic_for_a_seed_and_validates_arguments() -> None:
    qubo = random_qubo(5, 20)
    first = solve_simulated_quantum_annealing(qubo, num_reads=3, num_sweeps=20, seed=7)
    assert first == solve_simulated_quantum_annealing(qubo, num_reads=3, num_sweeps=20, seed=7)
    with pytest.raises(ValueError):
        solve_simulated_quantum_annealing(qubo, trotter_slices=1)
    with pytest.raises(ValueError):
        solve_simulated_quantum_annealing(qubo, gamma_start=0.1, gamma_end=1.0)


def test_sqa_solver_refines_without_worsening_with_and_without_bias() -> None:
    instance, routes = misplaced_instance()
    m_prob = np.full((6, 6), 0.5)
    np.fill_diagonal(m_prob, 0.0)
    for bias in (DiffusionBiasConfig(), DiffusionBiasConfig(enabled=True, alpha=0.5)):
        solver = SimulatedQuantumAnnealingSolver(num_reads=4, num_sweeps=40, seed=1, bias=bias)
        trace = refine_solution(instance, routes, solver, m_prob=m_prob, max_rounds=2)
        assert validate_routes(instance, trace.routes).feasible
        assert trace.improvement > 0
        assert all(step.solver_name.startswith("qubo_sqa") for step in trace.steps)


def test_time_budget_runs_at_least_one_read() -> None:
    solver = SimulatedQuantumAnnealingSolver(num_sweeps=5, time_budget_seconds=0.0)
    assert len(solver._sample(random_qubo(4, 30))) == 1
