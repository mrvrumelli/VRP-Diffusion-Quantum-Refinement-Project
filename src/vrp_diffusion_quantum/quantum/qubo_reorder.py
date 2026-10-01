"""Route-reordering QUBO for one route or route segment (task Q1.3).

Given a :class:`~vrp_diffusion_quantum.quantum.neighborhoods.ReorderSubproblem` with ``k`` free
customers between a fixed start node and a fixed end node, binary variable ``x[i, p]`` is 1 when
customer ``i`` (in ``subproblem.customers`` order) is visited at position ``p``. The QUBO is

    E(x) =   sum_i D[s, i] x[i, 0]
           + sum_{p < k-1} sum_{i != j} D[i, j] x[i, p] x[j, p+1]
           + sum_i D[i, e] x[i, k-1]
           + A * sum_i (sum_p x[i, p] - 1)²
           + A * sum_p (sum_i x[i, p] - 1)²

so every permutation has energy equal to its path cost, and every non-permutation pays at least
``A``. The default ``A = penalty_factor * max(D)`` with ``penalty_factor = 2`` keeps permutations
lowest; tests verify this exhaustively for small ``k``. ``k²`` variables are used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from vrp_diffusion_quantum.quantum.neighborhoods import ReorderSubproblem
from vrp_diffusion_quantum.quantum.qubo import QUBO, QUBOBuilder

__all__ = [
    "ReorderQUBO",
    "build_reorder_qubo",
    "decode_reorder",
    "encode_reorder",
    "repair_reorder",
]

DEFAULT_PENALTY_FACTOR = 2.0


@dataclass(frozen=True)
class ReorderQUBO:
    """A reorder QUBO together with the subproblem it encodes."""

    qubo: QUBO
    subproblem: ReorderSubproblem

    @property
    def size(self) -> int:
        return self.subproblem.size

    def variable(self, customer_index: int, position: int) -> int:
        """Index of ``x[customer_index, position]`` in the QUBO vector."""
        return customer_index * self.size + position


def _label(customer: int, position: int) -> str:
    return f"x[c={customer},p={position}]"


def build_reorder_qubo(
    subproblem: ReorderSubproblem,
    *,
    penalty_weight: float | None = None,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
    extra_edge_weight: float = 0.0,
) -> ReorderQUBO:
    """Build the one-hot position-encoding QUBO for ``subproblem``.

    ``extra_edge_weight`` is the largest additional cost any single transition may receive from
    terms added later (for example diffusion bias); the default penalty grows to cover it.
    """
    k = subproblem.size
    if k < 1:
        raise ValueError("reorder subproblem must contain at least one customer")
    distances = subproblem.distances
    if penalty_weight is None:
        if penalty_factor <= 0.0:
            raise ValueError("penalty_factor must be positive")
        if extra_edge_weight < 0.0:
            raise ValueError("extra_edge_weight must be non-negative")
        penalty_weight = penalty_factor * max(float(distances.max()) + extra_edge_weight, 1e-12)
    if penalty_weight <= 0.0:
        raise ValueError("penalty_weight must be positive")

    labels = [_label(c, p) for c in subproblem.customers for p in range(k)]
    builder = QUBOBuilder(labels)

    def var(i: int, p: int) -> int:
        return i * k + p

    start, end = 0, k + 1
    for i in range(k):
        builder.add_linear(var(i, 0), float(distances[start, i + 1]))
        builder.add_linear(var(i, k - 1), float(distances[i + 1, end]))
    for p in range(k - 1):
        for i in range(k):
            for j in range(k):
                if i != j:
                    builder.add_quadratic(var(i, p), var(j, p + 1), float(distances[i + 1, j + 1]))
    for i in range(k):
        builder.add_squared_penalty({var(i, p): 1.0 for p in range(k)}, 1.0, penalty_weight)
    for p in range(k):
        builder.add_squared_penalty({var(i, p): 1.0 for i in range(k)}, 1.0, penalty_weight)
    builder.record_penalty("permutation", penalty_weight)
    return ReorderQUBO(qubo=builder.build(), subproblem=subproblem)


def encode_reorder(reorder: ReorderQUBO, order: Sequence[int]) -> tuple[int, ...]:
    """Binary state for visiting ``order`` (a permutation of the subproblem customers)."""
    customers = reorder.subproblem.customers
    if sorted(order) != sorted(customers):
        raise ValueError("order must be a permutation of the subproblem customers")
    index = {customer: i for i, customer in enumerate(customers)}
    x = [0] * reorder.qubo.num_variables
    for position, customer in enumerate(order):
        x[reorder.variable(index[customer], position)] = 1
    return tuple(x)


def decode_reorder(reorder: ReorderQUBO, x: Sequence[int]) -> tuple[int, ...] | None:
    """Visiting order encoded by ``x``, or ``None`` if ``x`` is not a permutation matrix."""
    k = reorder.size
    grid = np.asarray(x, dtype=np.int64).reshape(k, k)
    if not (np.all(grid.sum(axis=0) == 1) and np.all(grid.sum(axis=1) == 1)):
        return None
    positions = grid.argmax(axis=1)
    customers = reorder.subproblem.customers
    return tuple(customers[i] for i in np.argsort(positions, kind="stable"))


def repair_reorder(reorder: ReorderQUBO, x: Sequence[int]) -> tuple[int, ...]:
    """Nearest valid permutation to ``x``: maximise agreement, break ties by path cost.

    Solves an assignment problem whose score is the number of agreeing bits, with a small
    distance-based tie-breaker so that equally consistent permutations prefer cheaper positions.
    """
    decoded = decode_reorder(reorder, x)
    if decoded is not None:
        return decoded
    k = reorder.size
    grid = np.asarray(x, dtype=np.float64).reshape(k, k)
    distances = reorder.subproblem.distances
    scale = max(float(distances.max()), 1e-12)
    endpoint_cost = np.zeros((k, k))
    endpoint_cost[:, 0] += distances[0, 1 : k + 1]
    endpoint_cost[:, k - 1] += distances[1 : k + 1, k + 1]
    cost: npt.NDArray[np.float64] = -grid + 1e-3 * endpoint_cost / scale
    rows, cols = linear_sum_assignment(cost)
    position_of = dict(zip(rows.tolist(), cols.tolist(), strict=True))
    customers = reorder.subproblem.customers
    return tuple(customers[i] for i in sorted(range(k), key=lambda i: position_of[i]))
