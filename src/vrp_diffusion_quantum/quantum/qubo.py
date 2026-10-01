"""QUBO representation, construction helpers and reference solvers.

A :class:`QUBO` stores an upper-triangular matrix ``Q`` and a constant ``offset``; the energy of a
binary vector ``x`` is ``xᵀ Q x + offset``. Linear terms sit on the diagonal (``x_i² = x_i``).
Everything here is plain NumPy so formulations can be checked without optional quantum packages;
:meth:`QUBO.to_bqm` converts to a ``dimod`` model when the ``quantum`` extra is installed.

The solvers are references for small subproblems: exhaustive enumeration (exact, for tests and
hand checks) and single-flip simulated annealing (the quantum-inspired control in
docs/quantum_scope.md).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = [
    "QUBO",
    "QUBOBuilder",
    "QUBOSample",
    "solve_exact",
    "solve_simulated_annealing",
]

MAX_EXACT_VARIABLES = 22


@dataclass(frozen=True)
class QUBO:
    """Upper-triangular QUBO with variable labels and the penalty weights used to build it."""

    matrix: npt.NDArray[np.float64]
    offset: float
    labels: tuple[str, ...]
    penalty_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.labels)
        if self.matrix.shape != (n, n):
            raise ValueError(f"matrix shape {self.matrix.shape} does not match {n} labels")
        if np.any(np.tril(self.matrix, k=-1) != 0.0):
            raise ValueError("matrix must be upper triangular")
        if not np.all(np.isfinite(self.matrix)) or not np.isfinite(self.offset):
            raise ValueError("QUBO coefficients must be finite")

    @property
    def num_variables(self) -> int:
        return len(self.labels)

    @property
    def num_terms(self) -> int:
        """Number of nonzero linear and quadratic coefficients (the logged ``qubo_num_terms``)."""
        return int(np.count_nonzero(self.matrix))

    def energy(self, x: npt.ArrayLike) -> float:
        vector = self._binary(x)
        return float(vector @ self.matrix @ vector + self.offset)

    def energies(self, states: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Energies of a ``[num_states, num_variables]`` batch of binary states."""
        batch = np.asarray(states, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.num_variables:
            raise ValueError("states must have shape [num_states, num_variables]")
        energies: npt.NDArray[np.float64] = (
            np.einsum("si,ij,sj->s", batch, self.matrix, batch) + self.offset
        )
        return energies

    def __add__(self, other: QUBO) -> QUBO:
        """Sum of two QUBOs over the same variables (used to add optional bias terms)."""
        if other.labels != self.labels:
            raise ValueError("can only add QUBOs with identical variable labels")
        weights = dict(self.penalty_weights)
        weights.update(other.penalty_weights)
        return QUBO(self.matrix + other.matrix, self.offset + other.offset, self.labels, weights)

    def to_dicts(self) -> tuple[dict[str, float], dict[tuple[str, str], float], float]:
        """``(linear, quadratic, offset)`` keyed by labels, omitting zero coefficients."""
        linear = {
            self.labels[i]: float(self.matrix[i, i])
            for i in range(self.num_variables)
            if self.matrix[i, i] != 0.0
        }
        rows, cols = np.nonzero(np.triu(self.matrix, k=1))
        quadratic = {
            (self.labels[i], self.labels[j]): float(self.matrix[i, j])
            for i, j in zip(rows.tolist(), cols.tolist(), strict=True)
        }
        return linear, quadratic, self.offset

    def to_bqm(self) -> object:
        """Convert to a ``dimod.BinaryQuadraticModel`` (requires the ``quantum`` extra)."""
        try:
            dimod = importlib.import_module("dimod")
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise ImportError("to_bqm requires `pip install -e .[quantum]`") from error
        linear, quadratic, offset = self.to_dicts()
        bqm: object = dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)
        return bqm

    def _binary(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        vector = np.asarray(x, dtype=np.float64)
        if vector.shape != (self.num_variables,):
            raise ValueError(f"x must have shape ({self.num_variables},)")
        if not np.all((vector == 0.0) | (vector == 1.0)):
            raise ValueError("x must be binary")
        return vector


class QUBOBuilder:
    """Accumulate linear, quadratic and constant terms by variable label."""

    def __init__(self, labels: Sequence[str]) -> None:
        if len(set(labels)) != len(labels):
            raise ValueError("variable labels must be unique")
        self.labels = tuple(labels)
        self._index = {label: index for index, label in enumerate(self.labels)}
        self._matrix = np.zeros((len(self.labels), len(self.labels)), dtype=np.float64)
        self._offset = 0.0
        self._penalty_weights: dict[str, float] = {}

    def index(self, label: str) -> int:
        return self._index[label]

    def add_linear(self, i: int, value: float) -> None:
        self._matrix[i, i] += value

    def add_quadratic(self, i: int, j: int, value: float) -> None:
        if i == j:
            self._matrix[i, i] += value  # x_i * x_i = x_i
        else:
            self._matrix[min(i, j), max(i, j)] += value

    def add_constant(self, value: float) -> None:
        self._offset += value

    def add_squared_penalty(
        self, coefficients: Mapping[int, float], rhs: float, weight: float
    ) -> None:
        """Add ``weight * (sum_i a_i x_i - rhs)²``, expanded with ``x_i² = x_i``."""
        if weight < 0.0:
            raise ValueError("penalty weight must be non-negative")
        items = sorted(coefficients.items())
        for position, (i, a_i) in enumerate(items):
            self.add_linear(i, weight * (a_i * a_i - 2.0 * rhs * a_i))
            for j, a_j in items[position + 1 :]:
                self.add_quadratic(i, j, weight * 2.0 * a_i * a_j)
        self.add_constant(weight * rhs * rhs)

    def record_penalty(self, name: str, weight: float) -> None:
        self._penalty_weights[name] = float(weight)

    def build(self) -> QUBO:
        return QUBO(self._matrix.copy(), self._offset, self.labels, dict(self._penalty_weights))


@dataclass(frozen=True)
class QUBOSample:
    """One binary state and its energy."""

    x: tuple[int, ...]
    energy: float


def solve_exact(qubo: QUBO, *, max_variables: int = MAX_EXACT_VARIABLES) -> QUBOSample:
    """Exhaustively enumerate all states; ties go to the lexicographically smallest state."""
    n = qubo.num_variables
    if n > max_variables:
        raise ValueError(f"{n} variables exceed the exhaustive limit of {max_variables}")
    best_energy = np.inf
    best_state = 0
    chunk = 1 << min(n, 16)
    bit_values = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
    for start in range(0, 1 << n, chunk):
        codes = np.arange(start, min(start + chunk, 1 << n), dtype=np.int64)
        states = ((codes[:, None] & bit_values[None, :]) > 0).astype(np.float64)
        energies = qubo.energies(states)
        index = int(np.argmin(energies))
        if energies[index] < best_energy - 1e-12:
            best_energy, best_state = float(energies[index]), int(codes[index])
    x = tuple(int(bool(best_state & int(bit))) for bit in bit_values)
    return QUBOSample(x=x, energy=float(best_energy))


def solve_simulated_annealing(
    qubo: QUBO,
    *,
    num_reads: int = 16,
    num_sweeps: int = 500,
    beta_range: tuple[float, float] | None = None,
    seed: int = 0,
) -> list[QUBOSample]:
    """Single-flip Metropolis annealing with a geometric inverse-temperature schedule.

    Returns one sample per read, sorted by energy. ``beta_range`` defaults to values scaled by the
    QUBO's coefficient magnitudes so the default works across penalty scales.
    """
    if num_reads < 1 or num_sweeps < 1:
        raise ValueError("num_reads and num_sweeps must be >= 1")
    n = qubo.num_variables
    rng = np.random.default_rng(seed)
    linear = np.diag(qubo.matrix).copy()
    coupling = qubo.matrix + qubo.matrix.T
    np.fill_diagonal(coupling, 0.0)
    if beta_range is None:
        scale = float(np.abs(qubo.matrix).max()) or 1.0
        beta_range = (0.1 / scale, 10.0 / scale)
    betas = np.geomspace(beta_range[0], beta_range[1], num_sweeps)
    samples: list[QUBOSample] = []
    for _ in range(num_reads):
        x = rng.integers(0, 2, size=n).astype(np.float64)
        field_values = linear + coupling @ x
        for beta in betas:
            for i in rng.permutation(n):
                delta = (1.0 - 2.0 * x[i]) * field_values[i]
                if delta <= 0.0 or rng.random() < np.exp(-beta * delta):
                    change = 1.0 - 2.0 * x[i]
                    x[i] += change
                    field_values += coupling[:, i] * change
        state = tuple(int(value) for value in x)
        samples.append(QUBOSample(x=state, energy=qubo.energy(state)))
    samples.sort(key=lambda sample: (sample.energy, sample.x))
    logger.debug("annealing best energy %.6f over %d reads", samples[0].energy, num_reads)
    return samples
