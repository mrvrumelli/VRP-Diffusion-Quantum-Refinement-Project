"""Two-route customer exchange / reassignment QUBO with capacity penalties (task Q1.4).

For an :class:`~vrp_diffusion_quantum.quantum.neighborhoods.ExchangeSubproblem` with ``m`` movable
customers, binary ``y_i = 1`` assigns movable customer ``i`` to the first route and ``y_i = 0`` to
the second. Fixed customers stay where they are.

True route cost depends on visiting order, so the QUBO minimises a surrogate:

    cost(y) = sum_i [ y_i * ins_A(i) + (1 - y_i) * ins_B(i) ]
              + (gamma / (m - 1)) * sum_{i<j} d_ij * [y_i == y_j]

``ins_R(i)`` is the cheapest insertion cost of customer ``i`` into route ``R``'s fixed customers
(kept in their current order, depot at both ends). The pairwise term discourages putting far-apart
movable customers in the same route. Decoded assignments are always re-evaluated with true route
cost before acceptance (see docs/quantum_scope.md).

Capacity: ``F_A + sum_i d_i y_i <= Q`` and ``F_B + sum_i d_i (1 - y_i) <= Q`` become equalities
with binary-encoded slack and quadratic penalties. A constraint that can never bind gets no slack.
Demands and capacity must be integral so the slack represents every feasible load exactly. The
default penalty weight exceeds the surrogate's whole range, so every capacity-violating state is
more expensive than every feasible one.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.quantum.neighborhoods import ExchangeSubproblem
from vrp_diffusion_quantum.quantum.qubo import QUBO, QUBOBuilder

__all__ = [
    "ExchangeQUBO",
    "bounded_slack_coefficients",
    "build_exchange_qubo",
    "decode_exchange",
    "repair_exchange",
]

DEFAULT_PENALTY_FACTOR = 1.5
DEFAULT_DISPERSION_WEIGHT = 1.0


@dataclass(frozen=True)
class ExchangeQUBO:
    """An exchange QUBO, its subproblem, and the surrogate's ingredients."""

    qubo: QUBO
    subproblem: ExchangeSubproblem
    insertion_costs: npt.NDArray[np.float64]  # [m, 2]: column 0 = first route, 1 = second
    pair_weights: npt.NDArray[np.float64]  # [m, m]: weight of [y_i == y_j], upper triangle used
    slack_slices: tuple[slice | None, slice | None]

    @property
    def size(self) -> int:
        return self.subproblem.size

    def surrogate_cost(self, assignment: Sequence[int]) -> float:
        """Surrogate objective of ``assignment`` (no capacity penalty)."""
        y = np.asarray(assignment, dtype=np.float64)
        linear = float(y @ self.insertion_costs[:, 0] + (1.0 - y) @ self.insertion_costs[:, 1])
        same = np.equal.outer(y, y).astype(np.float64)
        return linear + float(np.triu(self.pair_weights * same, k=1).sum())


def bounded_slack_coefficients(upper: int) -> list[int]:
    """Binary coefficients whose subset sums are exactly ``0..upper`` (capped last bit)."""
    if upper < 0:
        raise ValueError("slack upper bound must be non-negative")
    if upper == 0:
        return []
    bits = math.floor(math.log2(upper)) + 1
    coefficients = [1 << j for j in range(bits - 1)]
    coefficients.append(upper - (sum(coefficients)))
    return coefficients


def _integral(value: float, name: str) -> int:
    if not math.isclose(value, round(value), abs_tol=1e-9):
        raise ValueError(f"{name} must be integral for exact slack encoding, got {value}")
    return round(value)


def _insertion_cost(
    xy: dict[int | None, npt.NDArray[np.float64]], route: Sequence[int], customer: int
) -> float:
    nodes: list[int | None] = [None, *route, None]
    point = xy[customer]
    best = math.inf
    for before, after in itertools.pairwise(nodes):
        added = (
            float(np.linalg.norm(xy[before] - point))
            + float(np.linalg.norm(point - xy[after]))
            - float(np.linalg.norm(xy[before] - xy[after]))
        )
        best = min(best, added)
    return best


def build_exchange_qubo(
    instance: CVRPInstance,
    subproblem: ExchangeSubproblem,
    *,
    dispersion_weight: float = DEFAULT_DISPERSION_WEIGHT,
    penalty_weight: float | None = None,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
) -> ExchangeQUBO:
    """Build the assignment QUBO with slack-encoded capacity penalties."""
    m = subproblem.size
    if m < 1:
        raise ValueError("exchange subproblem must contain at least one movable customer")
    if dispersion_weight < 0.0:
        raise ValueError("dispersion_weight must be non-negative")
    capacity = _integral(subproblem.capacity, "capacity")
    demands = [_integral(float(d), "demand") for d in subproblem.demands]
    fixed_first = _integral(subproblem.fixed_loads[0], "fixed load")
    fixed_second = _integral(subproblem.fixed_loads[1], "fixed load")
    if fixed_first > capacity or fixed_second > capacity:
        raise ValueError("fixed customers alone exceed capacity; the subproblem is infeasible")

    nodes = instance.customer_node_indices()
    involved = [*subproblem.fixed[0], *subproblem.fixed[1], *subproblem.movable]
    xy: dict[int | None, npt.NDArray[np.float64]] = {
        c: np.asarray(instance.coords[nodes[c]], dtype=float) for c in involved
    }
    xy[None] = np.asarray(instance.coords[instance.depot_index], dtype=float)
    insertion = np.asarray(
        [
            [
                _insertion_cost(xy, subproblem.fixed[0], c),
                _insertion_cost(xy, subproblem.fixed[1], c),
            ]
            for c in subproblem.movable
        ],
        dtype=np.float64,
    )
    movable_xy = np.stack([xy[c] for c in subproblem.movable])
    distance = np.sqrt(((movable_xy[:, None, :] - movable_xy[None, :, :]) ** 2).sum(axis=-1))
    pair_weights = np.triu(distance, k=1) * (dispersion_weight / max(m - 1, 1))

    total_demand = sum(demands)
    upper_first = capacity - fixed_first  # sum d_i y_i <= upper_first
    lower_second = fixed_second + total_demand - capacity  # sum d_i y_i >= lower_second
    slack_first = bounded_slack_coefficients(upper_first) if upper_first < total_demand else None
    slack_second = (
        bounded_slack_coefficients(total_demand - lower_second) if lower_second > 0 else None
    )

    labels = [f"y[c={c}]" for c in subproblem.movable]
    slices: list[slice | None] = []
    for name, coefficients in (("first", slack_first), ("second", slack_second)):
        if coefficients is None:
            slices.append(None)
            continue
        start = len(labels)
        labels += [f"s_{name}[{j}]" for j in range(len(coefficients))]
        slices.append(slice(start, len(labels)))
    builder = QUBOBuilder(labels)

    # Surrogate objective.
    for i in range(m):
        builder.add_linear(i, float(insertion[i, 0] - insertion[i, 1]))
        builder.add_constant(float(insertion[i, 1]))
    for i in range(m):
        for j in range(i + 1, m):
            w = float(pair_weights[i, j])
            if w == 0.0:
                continue
            builder.add_constant(w)  # [y_i == y_j] = 1 - y_i - y_j + 2 y_i y_j
            builder.add_linear(i, -w)
            builder.add_linear(j, -w)
            builder.add_quadratic(i, j, 2.0 * w)

    if penalty_weight is None:
        if penalty_factor <= 1.0:
            raise ValueError("penalty_factor must exceed 1 to keep infeasible states above")
        span = float(np.abs(insertion[:, 0] - insertion[:, 1]).sum() + pair_weights.sum())
        penalty_weight = penalty_factor * max(span, 1e-9)
    if penalty_weight <= 0.0:
        raise ValueError("penalty_weight must be positive")

    if slack_first is not None and slices[0] is not None:
        first_terms: dict[int, float] = {i: float(demands[i]) for i in range(m)}
        for offset, c in zip(range(slices[0].start, slices[0].stop), slack_first, strict=True):
            first_terms[offset] = float(c)
        builder.add_squared_penalty(first_terms, float(upper_first), penalty_weight)
    if slack_second is not None and slices[1] is not None:
        second_terms: dict[int, float] = {i: float(demands[i]) for i in range(m)}
        for offset, c in zip(range(slices[1].start, slices[1].stop), slack_second, strict=True):
            second_terms[offset] = -float(c)
        builder.add_squared_penalty(second_terms, float(lower_second), penalty_weight)
    builder.record_penalty("capacity", penalty_weight)
    return ExchangeQUBO(
        qubo=builder.build(),
        subproblem=subproblem,
        insertion_costs=insertion,
        pair_weights=pair_weights,
        slack_slices=(slices[0], slices[1]),
    )


def decode_exchange(exchange: ExchangeQUBO, x: Sequence[int]) -> tuple[int, ...]:
    """Assignment part of a QUBO state (slack bits are dropped)."""
    if len(x) != exchange.qubo.num_variables:
        raise ValueError("state length does not match the QUBO")
    return tuple(int(value) for value in x[: exchange.size])


def repair_exchange(exchange: ExchangeQUBO, assignment: Sequence[int]) -> tuple[int, ...] | None:
    """Make ``assignment`` capacity-feasible by greedy moves; ``None`` if no move sequence fits.

    While a route is overloaded, move the movable customer of that route whose move to the other
    route increases the surrogate least and still fits the other route.
    """
    subproblem = exchange.subproblem
    current = [int(value) for value in assignment]
    for _ in range(exchange.size):
        if subproblem.is_feasible(current):
            return tuple(current)
        loads = subproblem.loads(current)
        overloaded = 0 if loads[0] > subproblem.capacity else 1
        source_value = 1 if overloaded == 0 else 0
        candidates: list[tuple[float, int]] = []
        for i, value in enumerate(current):
            if value != source_value:
                continue
            trial = current.copy()
            trial[i] = 1 - value
            target_load = subproblem.loads(trial)[1 - overloaded]
            if target_load <= subproblem.capacity + 1e-9:
                candidates.append((exchange.surrogate_cost(trial), i))
        if not candidates:
            return None
        _, chosen = min(candidates)
        current[chosen] = 1 - current[chosen]
    return tuple(current) if subproblem.is_feasible(current) else None
