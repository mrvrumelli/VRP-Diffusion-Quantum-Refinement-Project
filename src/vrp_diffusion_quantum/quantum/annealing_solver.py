"""Simulated annealing for a QUBO. This is the fallback when no quantum device is available."""

from __future__ import annotations

import math

import numpy as np

from vrp_diffusion_quantum.quantum.qubo import (
    AnnealingSettings,
    Qubo,
    QuboSolveResult,
    make_result,
)

SOLVER_NAME = "simulated_annealing"


def solve_annealing(qubo: Qubo, settings: AnnealingSettings) -> QuboSolveResult:
    """Return one annealed sample per restart, keeping the lowest-energy bitstring."""
    rng = np.random.default_rng(settings.seed)
    n = qubo.num_variables
    samples = np.empty((settings.num_samples, n), dtype=np.int8)
    sample_energies = np.empty(settings.num_samples, dtype=np.float64)
    for sample_index in range(settings.num_samples):
        bitstring = rng.integers(0, 2, size=n, dtype=np.int8)
        energy = qubo.energy(bitstring)
        best = bitstring.copy()
        best_energy = energy
        for sweep in range(settings.num_sweeps):
            temperature = _temperature(settings, sweep)
            for qubit in rng.permutation(n):
                bitstring[qubit] = np.int8(1 - int(bitstring[qubit]))
                proposed = qubo.energy(bitstring)
                delta = (proposed - energy) / qubo.energy_scale
                if delta <= 0.0 or rng.random() < _accept_probability(delta, temperature):
                    energy = proposed
                    if energy < best_energy:
                        best_energy = energy
                        best = bitstring.copy()
                else:
                    bitstring[qubit] = np.int8(1 - int(bitstring[qubit]))
        samples[sample_index] = best
        sample_energies[sample_index] = best_energy
    return make_result(
        solver_name=SOLVER_NAME,
        qubo=qubo,
        samples=samples,
        sample_energies=sample_energies,
        seed=settings.seed,
    )


def _temperature(settings: AnnealingSettings, sweep: int) -> float:
    if settings.num_sweeps == 1:
        return settings.final_temperature
    fraction = sweep / (settings.num_sweeps - 1)
    ratio = settings.final_temperature / settings.initial_temperature
    return float(settings.initial_temperature * (ratio**fraction))


def _accept_probability(delta: float, temperature: float) -> float:
    if delta / temperature > 700.0:
        return 0.0
    return math.exp(-delta / temperature)
