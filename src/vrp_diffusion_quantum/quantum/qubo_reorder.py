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

from vrp_diffusion_quantum.quantum._validation import binary_vector
from vrp_diffusion_quantum.quantum.neighborhoods import ReorderSubproblem
from vrp_diffusion_quantum.quantum.qubo import QUBO, QUBOBuilder

__all__ = [
    "MAX_EXACT_REPAIR_SIZE",
    "ReorderQUBO",
    "build_reorder_qubo",
    "decode_reorder",
    "encode_reorder",
    "repair_reorder",
]

DEFAULT_PENALTY_FACTOR = 2.0
MAX_EXACT_REPAIR_SIZE = 12


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
    """Decode a binary permutation matrix, or return ``None`` for invalid one-hot constraints.

    Malformed shapes and nonbinary states raise ``ValueError`` before any integer conversion.
    """
    k = reorder.size
    grid = binary_vector(x, k * k).reshape(k, k)
    if not (np.all(grid.sum(axis=0) == 1) and np.all(grid.sum(axis=1) == 1)):
        return None
    positions = grid.argmax(axis=1)
    customers = reorder.subproblem.customers
    return tuple(customers[i] for i in np.argsort(positions, kind="stable"))


def repair_reorder(reorder: ReorderQUBO, x: Sequence[int]) -> tuple[int, ...]:
    """Nearest valid permutation to ``x``: maximise agreement, break ties by path cost.

    A subset dynamic program first maximises the number of selected one-bits, then minimises
    the full path cost, including both endpoints. For a binary input, this also minimises Hamming
    distance to the sample. Invalid states are limited to ``MAX_EXACT_REPAIR_SIZE`` customers
    (time O(k² 2^k), memory O(k 2^k)); valid permutations are returned directly at any size.
    """
    decoded = decode_reorder(reorder, x)
    if decoded is not None:
        return decoded
    k = reorder.size
    if k > MAX_EXACT_REPAIR_SIZE:
        raise ValueError(f"exact reorder repair is limited to {MAX_EXACT_REPAIR_SIZE} customers")
    grid = binary_vector(x, k * k).reshape(k, k)
    distances = reorder.subproblem.distances
    full = (1 << k) - 1
    agreements = np.full((1 << k, k), -1, dtype=np.int64)
    costs = np.full((1 << k, k), np.inf)
    parents = np.full((1 << k, k), -1, dtype=np.int64)
    for last in range(k):
        agreements[1 << last, last] = int(grid[last, 0])
        costs[1 << last, last] = distances[0, last + 1]
    for mask in range(1, full):
        position = mask.bit_count()
        for last in range(k):
            if agreements[mask, last] < 0:
                continue
            for nxt in range(k):
                if mask & (1 << nxt):
                    continue
                target = mask | (1 << nxt)
                agreement = agreements[mask, last] + int(grid[nxt, position])
                cost = costs[mask, last] + distances[last + 1, nxt + 1]
                if agreement > agreements[target, nxt] or (
                    agreement == agreements[target, nxt] and cost < costs[target, nxt]
                ):
                    agreements[target, nxt] = agreement
                    costs[target, nxt] = cost
                    parents[target, nxt] = last
    last = min(
        range(k),
        key=lambda i: (-agreements[full, i], costs[full, i] + distances[i + 1, k + 1], i),
    )
    indices: list[int] = []
    mask = full
    while last >= 0:
        indices.append(last)
        previous = int(parents[mask, last])
        mask ^= 1 << last
        last = previous
    customers = reorder.subproblem.customers
    return tuple(customers[i] for i in reversed(indices))
