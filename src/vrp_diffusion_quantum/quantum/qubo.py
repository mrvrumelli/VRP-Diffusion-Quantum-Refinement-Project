"""QUBO energy and the settings shared by the solvers."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qubo:
    """Binary quadratic energy: offset + linear·x + sum_{i<j} coeff_ij x_i x_j."""

    linear: np.ndarray
    quadratic: tuple[tuple[int, int, float], ...]
    offset: float = 0.0
    energy_scale: float = 1.0

    def __post_init__(self) -> None:
        linear = np.asarray(self.linear, dtype=np.float64)
        if linear.ndim != 1 or linear.shape[0] < 1:
            raise ValueError("linear must be a 1-dimensional array with at least one variable")
        if not np.isfinite(linear).all() or not math.isfinite(self.offset):
            raise ValueError("QUBO coefficients must be finite")
        if not math.isfinite(self.energy_scale) or self.energy_scale <= 0.0:
            raise ValueError("energy_scale must be positive")
        n = int(linear.shape[0])
        seen: set[tuple[int, int]] = set()
        for i, j, coeff in self.quadratic:
            if not (0 <= i < j < n):
                raise ValueError(f"quadratic index {(i, j)} is outside 0..{n - 1} with i < j")
            if (i, j) in seen:
                raise ValueError(f"duplicate quadratic term {(i, j)}")
            if not math.isfinite(coeff):
                raise ValueError("QUBO coefficients must be finite")
            seen.add((i, j))
        object.__setattr__(self, "linear", linear)

    @property
    def num_variables(self) -> int:
        return int(self.linear.shape[0])

    @property
    def num_terms(self) -> int:
        linear_terms = int(np.count_nonzero(self.linear))
        quadratic_terms = sum(1 for _i, _j, coeff in self.quadratic if coeff != 0.0)
        constant = 1 if self.offset != 0.0 else 0
        return linear_terms + quadratic_terms + constant

    def energy(self, bitstring: np.ndarray | Sequence[int]) -> float:
        bits = _bit_vector(bitstring, self.num_variables)
        total = self.offset + float(self.linear @ bits)
        for i, j, coeff in self.quadratic:
            total += coeff * float(bits[i] * bits[j])
        return total

    def energies(self) -> np.ndarray:
        """Energy of every bitstring. Bit q of index i is `(i >> q) & 1`."""
        n = self.num_variables
        if n > 16:
            raise ValueError("enumerating a QUBO supports at most 16 variables")
        index = np.arange(1 << n, dtype=np.int64)
        bits = ((index[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)
        total = self.offset + bits @ self.linear
        for i, j, coeff in self.quadratic:
            total += coeff * bits[:, i] * bits[:, j]
        return cast(np.ndarray, np.asarray(total, dtype=np.float64))


@dataclass(frozen=True)
class QuboSolveResult:
    """Best bitstring among the returned samples, plus those samples and their energies."""

    solver_name: str
    bitstring: np.ndarray
    energy: float
    samples: np.ndarray
    sample_energies: np.ndarray
    num_variables: int
    num_terms: int
    num_samples: int
    seed: int
    qaoa_depth: int | None = None
    expectation: float | None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.int8)
        sample_energies = np.asarray(self.sample_energies, dtype=np.float64)
        bitstring = np.asarray(self.bitstring, dtype=np.int8)
        if samples.ndim != 2 or samples.shape != (self.num_samples, self.num_variables):
            raise ValueError("samples must have shape (num_samples, num_variables)")
        if sample_energies.shape != (self.num_samples,):
            raise ValueError("sample_energies must have one entry per sample")
        if bitstring.shape != (self.num_variables,):
            raise ValueError("bitstring length must equal num_variables")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_energies", sample_energies)
        object.__setattr__(self, "bitstring", bitstring)


@dataclass(frozen=True)
class QaoaSettings:
    depth: int
    shots: int
    maxiter: int
    restarts: int
    seed: int
    noise: str = "noiseless"
    noise_probability: float = 0.0

    def __post_init__(self) -> None:
        _require_positive_int(self.depth, "qaoa.depth")
        _require_positive_int(self.shots, "qaoa.shots")
        _require_positive_int(self.maxiter, "qaoa.maxiter")
        _require_positive_int(self.restarts, "qaoa.restarts")
        _require_seed(self.seed)
        if self.noise not in {"noiseless", "depolarizing"}:
            raise ValueError("qaoa.noise must be noiseless or depolarizing")
        if (
            isinstance(self.noise_probability, bool)
            or not isinstance(self.noise_probability, int | float)
            or not math.isfinite(self.noise_probability)
            or not 0.0 <= self.noise_probability <= 1.0
        ):
            raise ValueError("qaoa.noise_probability must be between 0 and 1")
        if self.noise == "noiseless" and self.noise_probability != 0.0:
            raise ValueError("qaoa.noise_probability must be 0 when noise is noiseless")
        if self.noise == "depolarizing" and self.noise_probability <= 0.0:
            raise ValueError("qaoa.noise_probability must be positive for depolarizing noise")


@dataclass(frozen=True)
class AnnealingSettings:
    num_samples: int
    num_sweeps: int
    initial_temperature: float
    final_temperature: float
    seed: int

    def __post_init__(self) -> None:
        _require_positive_int(self.num_samples, "annealing.num_samples")
        _require_positive_int(self.num_sweeps, "annealing.num_sweeps")
        if self.initial_temperature <= 0.0 or self.final_temperature <= 0.0:
            raise ValueError("annealing temperatures must be positive")
        if self.initial_temperature < self.final_temperature:
            raise ValueError("annealing initial_temperature must be >= final_temperature")
        _require_seed(self.seed)


def decode_reorder(
    bitstring: np.ndarray | Sequence[int],
    segment: Sequence[int],
) -> tuple[int, ...] | None:
    """Return the customer order, or None when the bits are not a permutation."""
    customers = [int(node) for node in segment]
    k = len(customers)
    bits = _bit_vector(bitstring, k * k).reshape(k, k)
    if not np.all(bits.sum(axis=1) == 1.0) or not np.all(bits.sum(axis=0) == 1.0):
        return None
    order: list[int] = []
    for slot in range(k):
        row = int(np.flatnonzero(bits[:, slot] == 1.0)[0])
        order.append(customers[row])
    return tuple(order)


def assignment_bits(bitstring: np.ndarray | Sequence[int], n_customers: int) -> tuple[int, ...]:
    """Route bit for each selected customer. 1 is route A and 0 is route B."""
    if n_customers < 1:
        raise ValueError("n_customers must be positive")
    bits = np.asarray(bitstring)
    if bits.shape[0] < n_customers:
        raise ValueError("bitstring is shorter than the selected customers")
    return tuple(int(bit) for bit in bits[:n_customers])


def make_result(
    *,
    solver_name: str,
    qubo: Qubo,
    samples: np.ndarray,
    sample_energies: np.ndarray,
    seed: int,
    qaoa_depth: int | None = None,
    expectation: float | None = None,
) -> QuboSolveResult:
    best = int(np.argmin(sample_energies))
    result = QuboSolveResult(
        solver_name=solver_name,
        bitstring=samples[best],
        energy=float(sample_energies[best]),
        samples=samples,
        sample_energies=sample_energies,
        num_variables=qubo.num_variables,
        num_terms=qubo.num_terms,
        num_samples=int(samples.shape[0]),
        seed=seed,
        qaoa_depth=qaoa_depth,
        expectation=expectation,
    )
    log_solution(result)
    return result


def log_solution(result: QuboSolveResult) -> None:
    sample = "".join(str(int(bit)) for bit in result.bitstring)
    shown = [round(float(value), 6) for value in result.sample_energies[:8]]
    logger.info(
        "solver_name=%s qubo_num_variables=%d qubo_num_terms=%d num_samples=%d "
        "raw_energy=%.6f seed=%d qaoa_depth=%s sample=%s",
        result.solver_name,
        result.num_variables,
        result.num_terms,
        result.num_samples,
        result.energy,
        result.seed,
        "-" if result.qaoa_depth is None else result.qaoa_depth,
        sample,
    )
    logger.info("sample_energies=%s", shown)


def _bit_vector(bitstring: np.ndarray | Sequence[int], n: int) -> np.ndarray:
    bits = np.asarray(bitstring, dtype=np.float64)
    if bits.shape != (n,):
        raise ValueError(f"bitstring must have shape ({n},)")
    if not np.isin(bits, (0.0, 1.0)).all():
        raise ValueError("bitstring entries must be 0 or 1")
    return bits


def _require_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an integer >= 1")


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be an integer >= 0")
