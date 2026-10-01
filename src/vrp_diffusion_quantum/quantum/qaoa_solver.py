"""QAOA on small refinement QUBOs by exact statevector simulation (task Q2.1).

The simulator is plain NumPy, so the solver runs in the main environment without Qiskit. It
applies ``|+>^n``, then per layer the cost phase ``exp(-i gamma E(x))`` and ``RX(2 beta)`` on every
qubit, which is the ``QAOAAnsatz`` circuit up to a global phase; the screening script
``scripts/run_qaoa_screening.py`` checks the same simulator against Qiskit's ``Statevector``.

Angles are optimised with COBYLA on the exact expectation from ``restarts`` random starts. With
``warm_start`` the depth is grown one layer at a time and each depth also starts from the previous
depth's best angles, interpolated to one more layer (Zhou et al., PRX 10, 021067, 2020). Shots are
then drawn from the optimised state; the ``keep`` lowest-energy distinct shots are decoded by the
shared rule of :class:`~vrp_diffusion_quantum.quantum.refinement.QUBOSamplingSolver`. QUBOs with
more than ``max_qubits`` variables are skipped: the subproblem is left unchanged.

:class:`RandomQUBOSampler` is the matched control: uniform random bitstrings, the same shot count,
the same ``keep`` rule and the same qubit limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from vrp_diffusion_quantum.quantum.qubo import QUBO, QUBOSample
from vrp_diffusion_quantum.quantum.qubo_bias import DiffusionBiasConfig
from vrp_diffusion_quantum.quantum.refinement import QUBOSamplingSolver

logger = logging.getLogger(__name__)

__all__ = [
    "QAOAResult",
    "QAOASolver",
    "RandomQUBOSampler",
    "interpolate_angles",
    "optimise_qaoa",
    "qaoa_probabilities",
    "qubo_energies",
]

MAX_SIMULATED_QUBITS = 22


def qubo_energies(qubo: QUBO) -> npt.NDArray[np.float64]:
    """Energy of every basis state; qubit ``i`` is bit ``i`` of the state index (Qiskit order)."""
    n = qubo.num_variables
    if n > MAX_SIMULATED_QUBITS:
        raise ValueError(f"statevector simulation is limited to {MAX_SIMULATED_QUBITS} qubits")
    codes = np.arange(1 << n, dtype=np.int64)
    bits = ((codes[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    energies = np.einsum("si,ij,sj->s", bits, qubo.matrix, bits) + qubo.offset
    return np.asarray(energies, dtype=np.float64)


def qaoa_probabilities(
    theta: npt.ArrayLike, energies: npt.NDArray[np.float64], num_qubits: int, reps: int
) -> npt.NDArray[np.float64]:
    """Basis-state probabilities of the QAOA state; ``theta`` = all betas, then all gammas."""
    angles = np.asarray(theta, dtype=np.float64)
    if angles.shape != (2 * reps,):
        raise ValueError(f"theta must have {2 * reps} angles")
    state = np.full(1 << num_qubits, 1.0 / np.sqrt(1 << num_qubits), dtype=np.complex128)
    for beta, gamma in zip(angles[:reps], angles[reps:], strict=True):
        state = state * np.exp(-1j * gamma * energies)
        cos, sin = np.cos(beta), -1j * np.sin(beta)
        for qubit in range(num_qubits):
            view = state.reshape(1 << (num_qubits - qubit - 1), 2, 1 << qubit)
            zero, one = view[:, 0, :].copy(), view[:, 1, :].copy()
            view[:, 0, :] = cos * zero + sin * one
            view[:, 1, :] = sin * zero + cos * one
    probabilities = np.abs(state) ** 2
    return np.asarray(probabilities, dtype=np.float64)


def interpolate_angles(angles: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """INTERP initialisation: depth-``p`` angles (one schedule) to depth ``p + 1``."""
    values = np.asarray(angles, dtype=np.float64)
    p = values.size
    padded = np.concatenate([[0.0], values, [0.0]])
    out = np.empty(p + 1)
    for i in range(1, p + 2):
        out[i - 1] = (i - 1) / p * padded[i - 1] + (p - i + 1) / p * padded[i]
    return out


@dataclass(frozen=True)
class QAOAResult:
    """Optimised angles, their expectation and the number of expectation evaluations."""

    theta: npt.NDArray[np.float64]
    expectation: float
    evaluations: int
    reps: int


def optimise_qaoa(
    energies: npt.NDArray[np.float64],
    num_qubits: int,
    reps: int,
    *,
    restarts: int = 4,
    maxiter: int = 300,
    warm_start: bool = True,
    rng: np.random.Generator,
) -> QAOAResult:
    """Minimise the exact energy expectation over the ``2 * reps`` QAOA angles with COBYLA."""
    if reps < 1 or restarts < 1:
        raise ValueError("reps and restarts must be >= 1")
    scale = float(np.abs(energies - energies.mean()).max()) or 1.0

    def random_start(depth: int) -> npt.NDArray[np.float64]:
        return np.concatenate(
            [rng.uniform(0.0, np.pi, depth), rng.uniform(0.0, np.pi / scale, depth)]
        )

    best: QAOAResult | None = None
    evaluations = 0
    depths = range(1, reps + 1) if warm_start else range(reps, reps + 1)
    for depth in depths:

        def expectation(theta: npt.NDArray[np.float64], depth: int = depth) -> float:
            return float(qaoa_probabilities(theta, energies, num_qubits, depth) @ energies)

        starts = [random_start(depth) for _ in range(restarts)]
        if best is not None:
            previous = best.theta
            starts[0] = np.concatenate(
                [
                    interpolate_angles(previous[: depth - 1]),
                    interpolate_angles(previous[depth - 1 :]),
                ]
            )
        best = None
        for start in starts:
            result = minimize(expectation, start, method="COBYLA", options={"maxiter": maxiter})
            evaluations += int(result.nfev)
            if best is None or float(result.fun) < best.expectation:
                best = QAOAResult(np.asarray(result.x), float(result.fun), 0, depth)
    assert best is not None
    return QAOAResult(best.theta, best.expectation, evaluations, reps)


def _keep_lowest(
    codes: npt.NDArray[np.int64],
    energies: npt.NDArray[np.float64],
    num_qubits: int,
    keep: int,
) -> list[QUBOSample]:
    unique = np.unique(codes)
    order = np.argsort(energies[unique], kind="stable")[:keep]
    return [
        QUBOSample(
            x=tuple(int((int(code) >> i) & 1) for i in range(num_qubits)),
            energy=float(energies[code]),
        )
        for code in unique[order]
    ]


@dataclass(frozen=True)
class QAOASolver(QUBOSamplingSolver):
    """QAOA in exact simulation as a refinement subproblem solver."""

    reps: int = 1
    restarts: int = 4
    maxiter: int = 300
    shots: int = 1024
    keep: int = 50
    max_qubits: int = 16
    warm_start: bool = True
    seed: int = 0
    bias: DiffusionBiasConfig = field(default_factory=DiffusionBiasConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.max_qubits <= MAX_SIMULATED_QUBITS:
            raise ValueError(f"max_qubits must be in [1, {MAX_SIMULATED_QUBITS}]")
        if self.shots < 1 or self.keep < 1:
            raise ValueError("shots and keep must be >= 1")

    @property
    def name(self) -> str:
        tag = f"_bias{self.bias.alpha:g}" if self.bias.active else ""
        return f"qaoa_p{self.reps}{tag}"

    def _sample(self, qubo: QUBO) -> list[QUBOSample]:
        n = qubo.num_variables
        if n > self.max_qubits:
            return []
        if n == 0:
            return [QUBOSample(x=(), energy=qubo.offset)]
        rng = np.random.default_rng(self.seed)
        energies = qubo_energies(qubo)
        result = optimise_qaoa(
            energies,
            n,
            self.reps,
            restarts=self.restarts,
            maxiter=self.maxiter,
            warm_start=self.warm_start,
            rng=rng,
        )
        probabilities = qaoa_probabilities(result.theta, energies, n, self.reps)
        codes = rng.choice(
            probabilities.size, size=self.shots, p=probabilities / probabilities.sum()
        )
        logger.debug(
            "QAOA n=%d p=%d expectation %.4f in %d evaluations",
            n,
            self.reps,
            result.expectation,
            result.evaluations,
        )
        return _keep_lowest(codes.astype(np.int64), energies, n, self.keep)


@dataclass(frozen=True)
class RandomQUBOSampler(QUBOSamplingSolver):
    """Matched control for QAOA: uniform random shots, same keep rule and qubit limit."""

    shots: int = 1024
    keep: int = 50
    max_qubits: int = 16
    seed: int = 0
    bias: DiffusionBiasConfig = field(default_factory=DiffusionBiasConfig)

    @property
    def name(self) -> str:
        return "random_qubo"

    def _sample(self, qubo: QUBO) -> list[QUBOSample]:
        n = qubo.num_variables
        if n > self.max_qubits:
            return []
        if n == 0:
            return [QUBOSample(x=(), energy=qubo.offset)]
        rng = np.random.default_rng(self.seed)
        energies = qubo_energies(qubo)
        codes = rng.integers(0, 1 << n, size=self.shots, dtype=np.int64)
        return _keep_lowest(codes, energies, n, self.keep)
