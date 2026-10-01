"""Classical local search baselines on the refinement subproblems (task Q1.6).

Every method here takes exactly the subproblem objects that the QUBO builders take
(:class:`~vrp_diffusion_quantum.quantum.neighborhoods.ReorderSubproblem` and
:class:`~vrp_diffusion_quantum.quantum.neighborhoods.ExchangeSubproblem`), so quantum and classical
results are compared on identical inputs.

Reorder methods:

* ``exact`` — Held-Karp dynamic programming for a path with fixed endpoints (optimal).
* ``two_opt`` — first-improvement 2-opt from the current order.

Exchange methods (true route cost, each route ordered by nearest neighbour + 2-opt):

* ``exhaustive`` — every capacity-feasible assignment (optimal for this evaluation).
* ``relocate_swap`` — first-improvement local search over single relocations and pair swaps.

:func:`refine` runs one method on one neighborhood of a full solution and accepts the result only
if the whole solution stays feasible and its true cost strictly decreases.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.quantum.neighborhoods import (
    ExchangeSubproblem,
    Neighborhood,
    ReorderSubproblem,
    apply_exchange,
    apply_reorder,
    evaluate_exchange,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_EXACT_REORDER",
    "MAX_EXHAUSTIVE_EXCHANGE",
    "ExchangeMethod",
    "ExchangeSolution",
    "RefinementResult",
    "ReorderMethod",
    "ReorderSolution",
    "refine",
    "solve_exchange",
    "solve_reorder",
]

ReorderMethod = Literal["exact", "two_opt"]
ExchangeMethod = Literal["exhaustive", "relocate_swap"]

MAX_EXACT_REORDER = 12
MAX_EXHAUSTIVE_EXCHANGE = 14
_EPS = 1e-12


@dataclass(frozen=True)
class ReorderSolution:
    """Result of a classical reorder method on one subproblem."""

    method: ReorderMethod
    order: tuple[int, ...]
    cost_before: float
    cost_after: float
    evaluations: int
    runtime_seconds: float

    @property
    def improvement(self) -> float:
        return self.cost_before - self.cost_after


@dataclass(frozen=True)
class ExchangeSolution:
    """Result of a classical exchange method on one subproblem."""

    method: ExchangeMethod
    assignment: tuple[int, ...]
    ordered_routes: tuple[tuple[int, ...], tuple[int, ...]]
    cost_before: float
    cost_after: float
    feasible: bool
    evaluations: int
    runtime_seconds: float

    @property
    def improvement(self) -> float:
        return self.cost_before - self.cost_after


# --------------------------------------------------------------------------------------------
# Reorder


def _held_karp(subproblem: ReorderSubproblem) -> tuple[tuple[int, ...], int]:
    """Optimal path from start to end through all customers; returns (order, evaluations)."""
    k = subproblem.size
    distances = subproblem.distances
    full = (1 << k) - 1
    cost = np.full((1 << k, k), np.inf)
    parent = np.full((1 << k, k), -1, dtype=np.int64)
    for j in range(k):
        cost[1 << j, j] = distances[0, j + 1]
    evaluations = 0
    for mask in range(1, full + 1):
        for last in range(k):
            if not mask & (1 << last) or not np.isfinite(cost[mask, last]):
                continue
            for nxt in range(k):
                if mask & (1 << nxt):
                    continue
                candidate = cost[mask, last] + distances[last + 1, nxt + 1]
                evaluations += 1
                target = mask | (1 << nxt)
                if candidate < cost[target, nxt] - _EPS:
                    cost[target, nxt] = candidate
                    parent[target, nxt] = last
    closing = cost[full] + distances[1 : k + 1, k + 1]
    last = int(np.argmin(closing))
    order_indices: list[int] = []
    mask = full
    while last >= 0:
        order_indices.append(last)
        previous = int(parent[mask, last])
        mask ^= 1 << last
        last = previous
    order_indices.reverse()
    return tuple(subproblem.customers[i] for i in order_indices), evaluations


def _two_opt(subproblem: ReorderSubproblem, *, max_passes: int) -> tuple[tuple[int, ...], int]:
    """First-improvement 2-opt on the free customers with both endpoints fixed."""
    k = subproblem.size
    distances = subproblem.distances
    nodes = list(range(1, k + 1))  # distance-matrix indices of the free customers, current order
    evaluations = 0
    for _ in range(max_passes):
        improved = False
        path = [0, *nodes, k + 1]
        for i in range(1, k):
            for j in range(i + 1, k + 1):
                a, b, c, d = path[i - 1], path[i], path[j], path[j + 1]
                delta = distances[a, c] + distances[b, d] - distances[a, b] - distances[c, d]
                evaluations += 1
                if delta < -_EPS:
                    path[i : j + 1] = reversed(path[i : j + 1])
                    improved = True
        nodes = path[1:-1]
        if not improved:
            break
    return tuple(subproblem.customers[index - 1] for index in nodes), evaluations


def solve_reorder(
    subproblem: ReorderSubproblem, method: ReorderMethod = "two_opt", *, max_passes: int = 100
) -> ReorderSolution:
    """Reorder the subproblem's customers with a classical method."""
    started = time.perf_counter()
    if method not in ("exact", "two_opt"):
        raise ValueError(f"unknown reorder method {method!r}")
    if subproblem.size <= 1:
        order, evaluations = subproblem.customers, 0
    elif method == "exact":
        if subproblem.size > MAX_EXACT_REORDER:
            raise ValueError(f"exact reorder is limited to {MAX_EXACT_REORDER} customers")
        order, evaluations = _held_karp(subproblem)
    else:
        order, evaluations = _two_opt(subproblem, max_passes=max_passes)
    return ReorderSolution(
        method=method,
        order=tuple(order),
        cost_before=subproblem.current_cost,
        cost_after=subproblem.path_cost(order),
        evaluations=evaluations,
        runtime_seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------------------------
# Exchange


@dataclass
class _ExchangeEvaluator:
    """Caches true-cost evaluations so each assignment is ordered and costed once."""

    instance: CVRPInstance
    subproblem: ExchangeSubproblem
    cache: dict[tuple[int, ...], tuple[float, list[int], list[int]]] = field(default_factory=dict)

    def __call__(self, assignment: tuple[int, ...]) -> tuple[float, list[int], list[int]]:
        if assignment not in self.cache:
            self.cache[assignment] = evaluate_exchange(self.instance, self.subproblem, assignment)
        return self.cache[assignment]


def _exhaustive(evaluate: _ExchangeEvaluator) -> tuple[int, ...] | None:
    subproblem = evaluate.subproblem
    best: tuple[float, tuple[int, ...]] | None = None
    for assignment in itertools.product((0, 1), repeat=subproblem.size):
        if not subproblem.is_feasible(assignment):
            continue
        cost = evaluate(assignment)[0]
        if best is None or cost < best[0] - _EPS:
            best = (cost, assignment)
    return None if best is None else best[1]


def _relocate_swap(evaluate: _ExchangeEvaluator, *, max_passes: int) -> tuple[int, ...]:
    subproblem = evaluate.subproblem
    current = subproblem.initial_assignment
    current_cost = evaluate(current)[0]
    m = subproblem.size
    for _ in range(max_passes):
        improved = False
        moves: list[tuple[int, ...]] = [(i,) for i in range(m)]
        moves += [(i, j) for i in range(m) for j in range(i + 1, m) if current[i] != current[j]]
        for move in moves:
            trial = list(current)
            for index in move:
                trial[index] = 1 - trial[index]
            candidate = tuple(trial)
            if not subproblem.is_feasible(candidate):
                continue
            cost = evaluate(candidate)[0]
            if cost < current_cost - _EPS:
                current, current_cost, improved = candidate, cost, True
                break
        if not improved:
            break
    return current


def solve_exchange(
    instance: CVRPInstance,
    subproblem: ExchangeSubproblem,
    method: ExchangeMethod = "relocate_swap",
    *,
    max_passes: int = 100,
) -> ExchangeSolution:
    """Reassign the subproblem's movable customers with a classical method.

    If no capacity-feasible assignment exists (or the method finds none), the initial assignment
    is returned with ``feasible`` reflecting whether it satisfies capacity.

    ``cost_before`` measures the two supplied routes before any reordering. For manually built
    subproblems without ``initial_routes``, it falls back to the reconstructed initial assignment.
    Search still uses the same nearest-neighbour/2-opt assignment objective as QUBO evaluation.
    """
    started = time.perf_counter()
    evaluate = _ExchangeEvaluator(instance, subproblem)
    cost_before = evaluate(subproblem.initial_assignment)[0]
    if subproblem.initial_routes is not None:
        cost_before = route_cost(instance, [list(route) for route in subproblem.initial_routes])
    if method == "exhaustive":
        if subproblem.size > MAX_EXHAUSTIVE_EXCHANGE:
            raise ValueError(f"exhaustive exchange is limited to {MAX_EXHAUSTIVE_EXCHANGE}")
        found = _exhaustive(evaluate)
        assignment = found if found is not None else subproblem.initial_assignment
    elif method == "relocate_swap":
        assignment = _relocate_swap(evaluate, max_passes=max_passes)
    else:
        raise ValueError(f"unknown exchange method {method!r}")
    cost_after, first, second = evaluate(assignment)
    return ExchangeSolution(
        method=method,
        assignment=assignment,
        ordered_routes=(tuple(first), tuple(second)),
        cost_before=cost_before,
        cost_after=cost_after,
        feasible=subproblem.is_feasible(assignment),
        evaluations=len(evaluate.cache),
        runtime_seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------------------------
# Full-solution refinement


@dataclass(frozen=True)
class RefinementResult:
    """One refinement attempt on one neighborhood of a full solution (logging fields included)."""

    neighborhood_type: str
    neighborhood_size: int
    solver_name: str
    routes: list[list[int]]
    cost_before: float
    cost_after: float
    accepted_improvement: float
    post_repair_feasible: bool
    evaluations: int
    runtime_seconds: float

    @property
    def accepted(self) -> bool:
        return self.accepted_improvement > 0.0


def refine(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    neighborhood: Neighborhood,
    method: ReorderMethod | ExchangeMethod,
) -> RefinementResult:
    """Solve one neighborhood classically; keep the change only if it is feasible and cheaper."""
    original = [list(route) for route in routes]
    cost_before = route_cost(instance, [route for route in original if route])
    if neighborhood.kind == "reorder":
        if method not in ("exact", "two_opt"):
            raise ValueError(f"{method!r} is not a reorder method")
        reorder = extract_reorder_subproblem(instance, original, neighborhood)
        solved = solve_reorder(reorder, method)
        candidate = apply_reorder(original, reorder, solved.order)
        evaluations, runtime = solved.evaluations, solved.runtime_seconds
    else:
        if method not in ("exhaustive", "relocate_swap"):
            raise ValueError(f"{method!r} is not an exchange method")
        exchange = extract_exchange_subproblem(instance, original, neighborhood)
        solution = solve_exchange(instance, exchange, method)
        candidate = apply_exchange(original, exchange, *solution.ordered_routes)
        evaluations, runtime = solution.evaluations, solution.runtime_seconds
    feasible = validate_routes(instance, [route for route in candidate if route]).feasible
    cost_after = route_cost(instance, [route for route in candidate if route])
    accept = feasible and cost_after < cost_before - 1e-9
    logger.debug(
        "refine %s size=%d method=%s before=%.6f after=%.6f accepted=%s",
        neighborhood.neighborhood_type,
        neighborhood.size,
        method,
        cost_before,
        cost_after,
        accept,
    )
    return RefinementResult(
        neighborhood_type=neighborhood.neighborhood_type,
        neighborhood_size=neighborhood.size,
        solver_name=f"classical_{method}",
        routes=candidate if accept else original,
        cost_before=cost_before,
        cost_after=cost_after if accept else cost_before,
        accepted_improvement=cost_before - cost_after if accept else 0.0,
        post_repair_feasible=feasible,
        evaluations=evaluations,
        runtime_seconds=runtime,
    )
