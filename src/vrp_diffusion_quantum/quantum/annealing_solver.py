"""Annealing-style QUBO solvers for refinement subproblems (task Q2.2).

Two classical stand-ins for quantum annealing hardware, both plugged into the shared decode path
of :class:`~vrp_diffusion_quantum.quantum.refinement.QUBOSamplingSolver`:

* :class:`~vrp_diffusion_quantum.quantum.refinement.AnnealingQUBOSolver` — simulated annealing
  (thermal fluctuations only), re-exported here.
* :class:`SimulatedQuantumAnnealingSolver` — path-integral Monte Carlo simulation of a
  transverse-field quantum annealer: ``trotter_slices`` coupled replicas of the classical state at
  a fixed temperature, with the transverse field ``Gamma`` lowered from ``gamma_start`` to
  ``gamma_end`` over the sweeps (Martonak, Santoro and Tosatti, PRB 66, 2002).

Both return one sample per read: SQA keeps the lowest-energy replica of each read. No external
annealer is required; ``dwave-neal`` is not installed in this environment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from vrp_diffusion_quantum.quantum.qubo import QUBO, QUBOSample
from vrp_diffusion_quantum.quantum.qubo_bias import DiffusionBiasConfig
from vrp_diffusion_quantum.quantum.refinement import AnnealingQUBOSolver, QUBOSamplingSolver

logger = logging.getLogger(__name__)

__all__ = [
    "AnnealingQUBOSolver",
    "SimulatedQuantumAnnealingSolver",
    "solve_simulated_quantum_annealing",
]


def solve_simulated_quantum_annealing(
    qubo: QUBO,
    *,
    num_reads: int = 16,
    num_sweeps: int = 100,
    trotter_slices: int = 16,
    slice_temperature: float = 1.0,
    gamma_start: float = 3.0,
    gamma_end: float = 1e-3,
    seed: int = 0,
) -> list[QUBOSample]:
    """Path-integral Monte Carlo quantum annealing; one sample (best replica) per read.

    Energies are measured in units of the QUBO's largest coefficient magnitude. With ``P`` replicas
    and ``P T = slice_temperature``, the Suzuki-Trotter action is
    ``sum_k E(x_k) / (P T) - K sum_k sum_i s_ki s_(k+1)i`` with spins ``s = 2x - 1`` and the
    ferromagnetic replica coupling ``K = -(1/2) ln tanh(Gamma / (P T))``. Each sweep visits every
    variable once in random order and proposes its flip in all replicas at once (Metropolis on the
    action). ``Gamma`` falls geometrically from ``gamma_start`` to ``gamma_end``. Returns the reads
    sorted by energy.
    """
    if num_reads < 1 or num_sweeps < 1 or trotter_slices < 2:
        raise ValueError("num_reads, num_sweeps >= 1 and trotter_slices >= 2 are required")
    if slice_temperature <= 0.0 or not 0.0 < gamma_end < gamma_start:
        raise ValueError("need slice_temperature > 0 and 0 < gamma_end < gamma_start")
    n = qubo.num_variables
    if n == 0:
        return [QUBOSample(x=(), energy=qubo.offset) for _ in range(num_reads)]
    rng = np.random.default_rng(seed)
    scale = float(np.abs(qubo.matrix).max()) or 1.0
    linear = np.diag(qubo.matrix) / scale
    coupling = (qubo.matrix + qubo.matrix.T) / scale
    np.fill_diagonal(coupling, 0.0)
    slices = trotter_slices
    pt = slice_temperature
    gammas = np.geomspace(gamma_start, gamma_end, num_sweeps)
    # Dimensionless replica coupling K per sweep (spin units).
    couplings = -0.5 * np.log(np.tanh(gammas / pt))
    samples: list[QUBOSample] = []
    for _ in range(num_reads):
        x = rng.integers(0, 2, size=(slices, n)).astype(np.float64)
        fields = linear[None, :] + x @ coupling  # [P, n]: dE/dx_i per replica
        for k_coupling in couplings:
            for i in rng.permutation(n):
                change = 1.0 - 2.0 * x[:, i]  # +1 to set, -1 to clear
                delta_classical = change * fields[:, i]  # [P]
                spin = 2.0 * x[:, i] - 1.0
                neighbours = np.roll(spin, 1) + np.roll(spin, -1)
                delta_quantum = 2.0 * k_coupling * spin * neighbours
                delta = delta_classical / pt + delta_quantum
                accept = (delta <= 0.0) | (rng.random(slices) < np.exp(-np.clip(delta, 0, 700)))
                if accept.any():
                    step = np.where(accept, change, 0.0)
                    x[:, i] += step
                    fields += step[:, None] * coupling[i][None, :]
        energies = [qubo.energy(tuple(int(v) for v in row)) for row in x]
        best = int(np.argmin(energies))
        samples.append(QUBOSample(x=tuple(int(v) for v in x[best]), energy=float(energies[best])))
    samples.sort(key=lambda sample: (sample.energy, sample.x))
    logger.debug("SQA best energy %.6f over %d reads", samples[0].energy, num_reads)
    return samples


@dataclass(frozen=True)
class SimulatedQuantumAnnealingSolver(QUBOSamplingSolver):
    """Simulated quantum annealing on the reorder/exchange QUBOs.

    With ``time_budget_seconds`` set, reads run one at a time until the budget is spent (at least
    one read); otherwise exactly ``num_reads`` reads run. Read ``r`` uses seed ``seed + r``.
    """

    num_reads: int = 16
    num_sweeps: int = 100
    trotter_slices: int = 16
    slice_temperature: float = 1.0
    gamma_start: float = 3.0
    gamma_end: float = 1e-3
    seed: int = 0
    bias: DiffusionBiasConfig = field(default_factory=DiffusionBiasConfig)
    time_budget_seconds: float | None = None

    @property
    def name(self) -> str:
        tag = f"_bias{self.bias.alpha:g}" if self.bias.active else ""
        return f"qubo_sqa{tag}"

    def _read(self, qubo: QUBO, reads: int, seed: int) -> list[QUBOSample]:
        return solve_simulated_quantum_annealing(
            qubo,
            num_reads=reads,
            num_sweeps=self.num_sweeps,
            trotter_slices=self.trotter_slices,
            slice_temperature=self.slice_temperature,
            gamma_start=self.gamma_start,
            gamma_end=self.gamma_end,
            seed=seed,
        )

    def _sample(self, qubo: QUBO) -> list[QUBOSample]:
        if self.time_budget_seconds is None:
            return self._read(qubo, self.num_reads, self.seed)
        samples: list[QUBOSample] = []
        started = time.perf_counter()
        read = 0
        while not samples or time.perf_counter() - started < self.time_budget_seconds:
            samples += self._read(qubo, 1, self.seed + read)
            read += 1
        return samples
