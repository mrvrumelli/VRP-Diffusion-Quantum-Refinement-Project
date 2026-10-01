"""Refinement loop with pluggable subproblem solvers (Phase 6 experiments).

A solver turns one extracted subproblem into a candidate: a visiting order for a
:class:`~vrp_diffusion_quantum.quantum.neighborhoods.ReorderSubproblem`, or two ordered routes for
an :class:`~vrp_diffusion_quantum.quantum.neighborhoods.ExchangeSubproblem`. Classical solvers wrap
``local_search.baselines``; QUBO solvers build the reorder or exchange QUBO (optionally with
diffusion bias), sample it, decode and repair every sample, and keep the candidate with the lowest
*true* cost. Every candidate is judged on true route cost, never on QUBO energy.

:func:`refine_solution` repeatedly selects neighborhoods of the current solution, solves them, and
accepts a change only if the full solution stays feasible and its true cost strictly falls, so it
can never make a solution worse. Each attempt is logged with the fields required by
docs/coding_standards.md for quantum refinement runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.local_search.baselines import (
    ExchangeMethod,
    ReorderMethod,
    solve_exchange,
    solve_reorder,
)
from vrp_diffusion_quantum.quantum.neighborhoods import (
    ExchangeSubproblem,
    Neighborhood,
    NeighborhoodConfig,
    NeighborhoodType,
    ReorderSubproblem,
    apply_exchange,
    apply_reorder,
    evaluate_exchange,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
    select_neighborhoods,
)
from vrp_diffusion_quantum.quantum.qubo import (
    QUBO,
    QUBOSample,
    solve_exact,
    solve_simulated_annealing,
)
from vrp_diffusion_quantum.quantum.qubo_bias import (
    DiffusionBiasConfig,
    build_biased_exchange_qubo,
    build_biased_reorder_qubo,
)
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    ExchangeQUBO,
    build_exchange_qubo,
    decode_exchange,
    repair_exchange,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import (
    ReorderQUBO,
    build_reorder_qubo,
    decode_reorder,
    repair_reorder,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

logger = logging.getLogger(__name__)

__all__ = [
    "AnnealingQUBOSolver",
    "ClassicalSolver",
    "ExactQUBOSolver",
    "RefinementStep",
    "RefinementTrace",
    "SolverOutcome",
    "SubproblemSolver",
    "refine_solution",
]


@dataclass(frozen=True)
class SolverOutcome:
    """A solver's candidate for one subproblem plus the logged solver fields."""

    solver_name: str
    candidate_cost: float
    order: tuple[int, ...] | None = None
    ordered_routes: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    runtime_seconds: float = 0.0
    evaluations: int = 0
    num_samples: int = 0
    raw_energy: float | None = None
    post_repair_feasible: bool = True
    qubo_num_variables: int | None = None
    qubo_num_terms: int | None = None
    penalty_weights: dict[str, float] = field(default_factory=dict)


class SubproblemSolver(Protocol):
    """Anything that can solve both subproblem kinds."""

    @property
    def name(self) -> str: ...

    def solve_reorder(
        self,
        instance: CVRPInstance,
        subproblem: ReorderSubproblem,
        m_prob: npt.NDArray[np.float64] | None,
    ) -> SolverOutcome: ...

    def solve_exchange(
        self,
        instance: CVRPInstance,
        subproblem: ExchangeSubproblem,
        m_prob: npt.NDArray[np.float64] | None,
    ) -> SolverOutcome: ...


# --------------------------------------------------------------------------------------------
# Solvers


@dataclass(frozen=True)
class ClassicalSolver:
    """Classical baselines from ``local_search.baselines`` on the identical subproblems.

    With ``restarts > 1``, the local search methods also run from ``restarts - 1`` random starting
    points drawn with ``seed``: random segment orders for 2-opt, random route assignments for
    relocate/swap. The best feasible result is kept. The first run always starts from the current
    solution, so restarts never do worse than a single run. Exact methods ignore ``restarts``.
    """

    reorder_method: ReorderMethod = "two_opt"
    exchange_method: ExchangeMethod = "relocate_swap"
    restarts: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.restarts < 1:
            raise ValueError("restarts must be >= 1")

    @property
    def name(self) -> str:
        base = f"classical_{self.reorder_method}+{self.exchange_method}"
        return base if self.restarts == 1 else f"{base}_x{self.restarts}"

    @staticmethod
    def _tag(method: str, runs: int) -> str:
        return f"classical_{method}" if runs == 1 else f"classical_{method}_x{runs}"

    def solve_reorder(
        self,
        instance: CVRPInstance,
        subproblem: ReorderSubproblem,
        m_prob: npt.NDArray[np.float64] | None,
    ) -> SolverOutcome:
        started = time.perf_counter()
        runs = self.restarts if self.reorder_method == "two_opt" else 1
        rng = np.random.default_rng(self.seed)
        k = subproblem.size
        best_cost, best_order, evaluations = np.inf, subproblem.customers, 0
        for run in range(runs):
            start = subproblem
            if run > 0:
                perm = rng.permutation(k)
                index = [0, *(int(i) + 1 for i in perm), k + 1]
                start = replace(
                    subproblem,
                    customers=tuple(subproblem.customers[int(i)] for i in perm),
                    distances=subproblem.distances[np.ix_(index, index)],
                )
            solution = solve_reorder(start, self.reorder_method)
            evaluations += solution.evaluations
            cost = subproblem.path_cost(solution.order)
            if cost < best_cost - 1e-12:
                best_cost, best_order = cost, solution.order
        return SolverOutcome(
            solver_name=self._tag(self.reorder_method, runs),
            candidate_cost=float(best_cost),
            order=best_order,
            runtime_seconds=time.perf_counter() - started,
            evaluations=evaluations,
            num_samples=runs,
        )

    def solve_exchange(
        self,
        instance: CVRPInstance,
        subproblem: ExchangeSubproblem,
        m_prob: npt.NDArray[np.float64] | None,
    ) -> SolverOutcome:
        started = time.perf_counter()
        runs = self.restarts if self.exchange_method == "relocate_swap" else 1
        rng = np.random.default_rng(self.seed)
        best = None
        evaluations = 0
        for run in range(runs):
            start = subproblem
            if run > 0:
                bits = rng.integers(0, 2, subproblem.size)
                start = replace(subproblem, initial_assignment=tuple(int(b) for b in bits))
            solution = solve_exchange(instance, start, self.exchange_method)
            evaluations += solution.evaluations
            if best is None or (solution.feasible, -solution.cost_after) > (
                best.feasible,
                -best.cost_after,
            ):
                best = solution
        assert best is not None
        return SolverOutcome(
            solver_name=self._tag(self.exchange_method, runs),
            candidate_cost=best.cost_after,
            ordered_routes=best.ordered_routes,
            runtime_seconds=time.perf_counter() - started,
            evaluations=evaluations,
            num_samples=runs,
            post_repair_feasible=best.feasible,
        )


def _reorder_qubo(
    subproblem: ReorderSubproblem,
    m_prob: npt.NDArray[np.float64] | None,
    bias: DiffusionBiasConfig,
) -> ReorderQUBO:
    if bias.active and m_prob is not None:
        return build_biased_reorder_qubo(subproblem, m_prob, bias)
    return build_reorder_qubo(subproblem)


def _exchange_qubo(
    instance: CVRPInstance,
    subproblem: ExchangeSubproblem,
    m_prob: npt.NDArray[np.float64] | None,
    bias: DiffusionBiasConfig,
) -> ExchangeQUBO:
    if bias.active and m_prob is not None:
        return build_biased_exchange_qubo(instance, subproblem, m_prob, bias)
    return build_exchange_qubo(instance, subproblem)


def _best_reorder(
    reorder: ReorderQUBO, samples: Sequence[QUBOSample]
) -> tuple[tuple[int, ...], float, float, bool]:
    """Best true-cost order among decoded or repaired samples: (order, cost, energy, valid)."""
    best: tuple[float, float, tuple[int, ...], bool] | None = None
    for sample in samples:
        decoded = decode_reorder(reorder, sample.x)
        order = decoded if decoded is not None else repair_reorder(reorder, sample.x)
        cost = reorder.subproblem.path_cost(order)
        key = (cost, sample.energy, order, decoded is not None)
        if best is None or key[:2] < best[:2]:
            best = key
    assert best is not None
    cost, energy, order, valid = best
    return order, cost, energy, valid


def _best_exchange(
    instance: CVRPInstance, exchange: ExchangeQUBO, samples: Sequence[QUBOSample]
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], float, float, bool, int]:
    """Best feasible true-cost routes among decoded samples (repairing infeasible ones).

    Returns ``(routes, cost, energy, any_feasible, evaluations)``. When no sample can be made
    feasible, the original routes are returned unchanged with ``any_feasible = False``.
    """
    subproblem = exchange.subproblem
    seen: dict[tuple[int, ...], float] = {}
    best: tuple[float, float, tuple[tuple[int, ...], tuple[int, ...]]] | None = None
    for sample in samples:
        assignment = repair_exchange(exchange, decode_exchange(exchange, sample.x))
        if assignment is None:
            continue
        if assignment in seen:
            continue
        cost, first, second = evaluate_exchange(instance, subproblem, assignment)
        seen[assignment] = cost
        if best is None or (cost, sample.energy) < best[:2]:
            best = (cost, sample.energy, (tuple(first), tuple(second)))
    if best is None:
        assert subproblem.initial_routes is not None
        original = subproblem.initial_routes
        cost = route_cost(instance, [list(route) for route in original if route])
        return original, cost, float("nan"), False, len(seen)
    return best[2], best[0], best[1], True, len(seen)


@dataclass(frozen=True)
class AnnealingQUBOSolver:
    """Simulated annealing on the reorder/exchange QUBOs (the quantum-inspired control).

    With ``time_budget_seconds`` set, reads run one at a time until the budget is spent (at least
    one read); otherwise exactly ``num_reads`` reads run. Read ``r`` uses seed ``seed + r``.
    """

    num_reads: int = 16
    num_sweeps: int = 200
    seed: int = 0
    bias: DiffusionBiasConfig = field(default_factory=DiffusionBiasConfig)
    time_budget_seconds: float | None = None

    @property
    def name(self) -> str:
        tag = f"_bias{self.bias.alpha:g}" if self.bias.active else ""
        return f"qubo_annealing{tag}"

    def _sample(self, qubo: QUBO) -> list[QUBOSample]:
        if self.time_budget_seconds is None:
            return solve_simulated_annealing(
                qubo, num_reads=self.num_reads, num_sweeps=self.num_sweeps, seed=self.seed
            )
        samples: list[QUBOSample] = []
        started = time.perf_counter()
        read = 0
        while not samples or time.perf_counter() - started < self.time_budget_seconds:
            samples += solve_simulated_annealing(
                qubo, num_reads=1, num_sweeps=self.num_sweeps, seed=self.seed + read
            )
            read += 1
        return samples

    def solve_reorder(
        self,
        instance: CVRPInstance,
        subproblem: ReorderSubproblem,
        m_prob: npt.NDArray[np.float64] | None,
    ) -> SolverOutcome:
        started = time.perf_counter()
        reorder = _reorder_qubo(subproblem, m_prob, self.bias)
        samples = self._sample(reorder.qubo)
        order, cost, energy, _decoded_valid = _best_reorder(reorder, samples)
        return SolverOutcome(
            solver_name=self.name,
            candidate_cost=cost,
            order=order,
            runtime_seconds=time.perf_counter() - started,
            evaluations=len(samples),
            num_samples=len(samples),
            raw_energy=energy,
            post_repair_feasible=True,
            qubo_num_variables=reorder.qubo.num_variables,
            qubo_num_terms=reorder.qubo.num_terms,
            penalty_weights=dict(reorder.qubo.penalty_weights),
        )

    def solve_exchange(
        self,
        instance: CVRPInstance,
        subproblem: ExchangeSubproblem,
        m_prob: npt.NDArray[np.float64] | None,
    ) -> SolverOutcome:
        started = time.perf_counter()
        exchange = _exchange_qubo(instance, subproblem, m_prob, self.bias)
        samples = self._sample(exchange.qubo)
        routes, cost, energy, feasible, evaluations = _best_exchange(instance, exchange, samples)
        return SolverOutcome(
            solver_name=self.name,
            candidate_cost=cost,
            ordered_routes=routes,
            runtime_seconds=time.perf_counter() - started,
            evaluations=evaluations,
            num_samples=len(samples),
            raw_energy=energy,
            post_repair_feasible=feasible,
            qubo_num_variables=exchange.qubo.num_variables,
            qubo_num_terms=exchange.qubo.num_terms,
            penalty_weights=dict(exchange.qubo.penalty_weights),
        )


@dataclass(frozen=True)
class ExactQUBOSolver:
    """Exhaustive QUBO minimisation: shows what the formulation itself prefers (small only)."""

    bias: DiffusionBiasConfig = field(default_factory=DiffusionBiasConfig)

    @property
    def name(self) -> str:
        tag = f"_bias{self.bias.alpha:g}" if self.bias.active else ""
        return f"qubo_exact{tag}"

    def solve_reorder(
        self,
        instance: CVRPInstance,
        subproblem: ReorderSubproblem,
        m_prob: npt.NDArray[np.float64] | None,
    ) -> SolverOutcome:
        started = time.perf_counter()
        reorder = _reorder_qubo(subproblem, m_prob, self.bias)
        sample = solve_exact(reorder.qubo)
        order, cost, energy, valid = _best_reorder(reorder, [sample])
        return SolverOutcome(
            solver_name=self.name,
            candidate_cost=cost,
            order=order,
            runtime_seconds=time.perf_counter() - started,
            evaluations=1,
            num_samples=1,
            raw_energy=energy,
            post_repair_feasible=valid,
            qubo_num_variables=reorder.qubo.num_variables,
            qubo_num_terms=reorder.qubo.num_terms,
            penalty_weights=dict(reorder.qubo.penalty_weights),
        )

    def solve_exchange(
        self,
        instance: CVRPInstance,
        subproblem: ExchangeSubproblem,
        m_prob: npt.NDArray[np.float64] | None,
    ) -> SolverOutcome:
        started = time.perf_counter()
        exchange = _exchange_qubo(instance, subproblem, m_prob, self.bias)
        sample = solve_exact(exchange.qubo)
        routes, cost, energy, feasible, evaluations = _best_exchange(instance, exchange, [sample])
        return SolverOutcome(
            solver_name=self.name,
            candidate_cost=cost,
            ordered_routes=routes,
            runtime_seconds=time.perf_counter() - started,
            evaluations=evaluations,
            num_samples=1,
            raw_energy=energy,
            post_repair_feasible=feasible,
            qubo_num_variables=exchange.qubo.num_variables,
            qubo_num_terms=exchange.qubo.num_terms,
            penalty_weights=dict(exchange.qubo.penalty_weights),
        )


# --------------------------------------------------------------------------------------------
# Loop


@dataclass(frozen=True)
class RefinementStep:
    """One attempted neighborhood with the coding-standard refinement logging fields."""

    round_index: int
    neighborhood_type: str
    neighborhood_size: int
    route_indices: tuple[int, ...]
    solver_name: str
    qubo_num_variables: int | None
    qubo_num_terms: int | None
    penalty_weights: dict[str, float]
    num_samples: int
    raw_energy: float | None
    cost_before: float
    cost_after: float
    accepted_improvement: float
    post_repair_feasible: bool
    runtime_seconds: float
    evaluations: int


@dataclass
class RefinementTrace:
    """Initial and final solution of one refinement run plus every attempted step."""

    initial_cost: float
    final_cost: float
    routes: list[list[int]]
    steps: list[RefinementStep]
    skipped_stale: int = 0

    @property
    def improvement(self) -> float:
        return self.initial_cost - self.final_cost

    @property
    def accepted_steps(self) -> int:
        return sum(step.accepted_improvement > 0.0 for step in self.steps)


Kind = Literal["reorder", "exchange"]


def _solution_cost(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> float:
    return route_cost(instance, [list(route) for route in routes if route])


def _still_applies(neighborhood: Neighborhood, routes: Sequence[Sequence[int]]) -> bool:
    """Whether a neighborhood selected at the start of a round still fits the current routes.

    An accepted step can shorten or rewrite routes that later neighborhoods of the same round
    point into: a reorder segment can then fall outside its route or cover other customers, and
    an exchange can list a movable customer that has left both of its routes.
    """
    if any(index >= len(routes) for index in neighborhood.route_indices):
        return False
    if neighborhood.kind == "reorder":
        if neighborhood.segment is None:
            return False
        (route_index,) = neighborhood.route_indices
        start, end = neighborhood.segment
        route = routes[route_index]
        return 0 <= start < end <= len(route) and tuple(route[start:end]) == neighborhood.customers
    first, second = neighborhood.route_indices
    members = set(routes[first]) | set(routes[second])
    return all(customer in members for customer in neighborhood.customers)


def refine_solution(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    solver: SubproblemSolver,
    *,
    m_prob: npt.ArrayLike | None = None,
    config: NeighborhoodConfig | None = None,
    types: Sequence[NeighborhoodType] | None = None,
    max_rounds: int = 3,
) -> RefinementTrace:
    """Select, solve and accept improving neighborhoods until a round makes no change.

    ``types`` defaults to all four selectors when ``m_prob`` is given, otherwise to the two that
    need no prediction. Neighborhoods made stale by an earlier acceptance in the same round are
    skipped (and counted); empty routes are dropped between rounds.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    matrix = None if m_prob is None else np.asarray(m_prob, dtype=np.float64)
    chosen_types: Sequence[NeighborhoodType] = (
        types
        if types is not None
        else (
            ("high_cost_route", "uncertain_m", "low_confidence_edges", "two_route_exchange")
            if matrix is not None
            else ("high_cost_route", "two_route_exchange")
        )
    )
    current = [list(route) for route in routes if route]
    if not validate_routes(instance, current).feasible:
        raise ValueError("initial routes must be feasible")
    current_cost = _solution_cost(instance, current)
    initial_cost = current_cost
    steps: list[RefinementStep] = []
    skipped = 0
    for round_index in range(max_rounds):
        neighborhoods = select_neighborhoods(
            instance, current, m_prob=matrix, config=config, types=chosen_types
        )
        improved = False
        for neighborhood in neighborhoods:
            if not _still_applies(neighborhood, current):
                skipped += 1
                continue
            if neighborhood.kind == "reorder":
                reorder = extract_reorder_subproblem(instance, current, neighborhood)
                outcome = solver.solve_reorder(instance, reorder, matrix)
                assert outcome.order is not None
                candidate = apply_reorder(current, reorder, outcome.order)
            else:
                exchange = extract_exchange_subproblem(instance, current, neighborhood)
                outcome = solver.solve_exchange(instance, exchange, matrix)
                assert outcome.ordered_routes is not None
                candidate = apply_exchange(current, exchange, *outcome.ordered_routes)
            feasible = validate_routes(instance, [r for r in candidate if r]).feasible
            candidate_cost = _solution_cost(instance, candidate)
            accept = feasible and candidate_cost < current_cost - 1e-9
            steps.append(
                RefinementStep(
                    round_index=round_index,
                    neighborhood_type=neighborhood.neighborhood_type,
                    neighborhood_size=neighborhood.size,
                    route_indices=neighborhood.route_indices,
                    solver_name=outcome.solver_name,
                    qubo_num_variables=outcome.qubo_num_variables,
                    qubo_num_terms=outcome.qubo_num_terms,
                    penalty_weights=outcome.penalty_weights,
                    num_samples=outcome.num_samples,
                    raw_energy=outcome.raw_energy,
                    cost_before=current_cost,
                    cost_after=candidate_cost if accept else current_cost,
                    accepted_improvement=current_cost - candidate_cost if accept else 0.0,
                    post_repair_feasible=feasible and outcome.post_repair_feasible,
                    runtime_seconds=outcome.runtime_seconds,
                    evaluations=outcome.evaluations,
                )
            )
            if accept:
                current, current_cost, improved = candidate, candidate_cost, True
        current = [route for route in current if route]
        if not improved:
            break
    logger.debug(
        "refinement %s: %.6f -> %.6f in %d steps",
        solver.name,
        initial_cost,
        current_cost,
        len(steps),
    )
    return RefinementTrace(
        initial_cost=initial_cost,
        final_cost=current_cost,
        routes=current,
        steps=steps,
        skipped_stale=skipped,
    )
