"""Classical OR labeling for CVRP instances (P1.1)."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pyvrp.stop import StoppingCriterion

from vrp_diffusion_quantum.data.generate_cvrp import CVRPDataset, CVRPInstance

logger = logging.getLogger(__name__)

__all__ = [
    "COORD_SCALE",
    "LOAD_SCALE",
    "CVRPSolution",
    "FleetMode",
    "is_feasible_solution",
    "route_cost",
    "save_labels",
    "solve_dataset",
    "solve_instance",
]

SolverName = Literal["pyvrp", "ortools"]
FleetMode = Literal["unlimited", "up_to", "exact"]

# Scale [0, 1] coordinates to integer distances for discrete OR solvers.
COORD_SCALE: int = 1000
# Scale non-integer demands/capacity to integers while preserving ratios.
LOAD_SCALE: int = 1_000_000
# Safety wall-clock cap when the caller asks for "unlimited" time.
_UNLIMITED_SAFETY_SECONDS: float = 24 * 60 * 60


@dataclass(frozen=True)
class CVRPSolution:
    """A labeled CVRP solution for one instance.

    Attributes:
        instance_id: index of the instance inside its source dataset.
        n_customers: number of customers (excluding the depot).
        routes: list of routes; each route is a list of customer node indices
            (depot is implicit at the start and end of every route).
        cost: total Euclidean tour length in the instance's coordinate space.
        num_vehicles: number of non-empty routes used in the solution.
        feasible: whether the solution respects capacity and covers every customer.
        solver_name: ``\"pyvrp\"`` or ``\"ortools\"``.
        runtime_seconds: wall-clock time spent solving this instance.
        seed: seed passed to the stochastic solver.
        fleet_mode: fleet constraint used while solving (``unlimited`` / ``up_to`` / ``exact``).
        fleet_size: configured fleet cap / exact size; ``None`` when unlimited.
    """

    instance_id: int
    n_customers: int
    routes: list[list[int]]
    cost: float
    num_vehicles: int
    feasible: bool
    solver_name: str
    runtime_seconds: float
    seed: int
    fleet_mode: FleetMode = "unlimited"
    fleet_size: int | None = None


class _NoImprovementRuntime:
    """Stop when the best cost has not improved for ``max_seconds`` wall-clock time."""

    def __init__(self, max_seconds: float) -> None:
        if max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}")
        self.max_seconds = float(max_seconds)
        self._best: float | None = None
        self._best_at: float | None = None

    def __call__(self, best_cost: float) -> bool:
        now = time.perf_counter()
        if self._best is None or best_cost < self._best:
            self._best = float(best_cost)
            self._best_at = now
            return False
        assert self._best_at is not None
        return (now - self._best_at) >= self.max_seconds


def route_cost(
    routes: list[list[int]],
    coords: np.ndarray,
    *,
    depot_index: int = 0,
) -> float:
    """Return the total Euclidean length of ``routes`` over ``coords``."""
    total = 0.0
    depot = coords[depot_index]
    for route in routes:
        if not route:
            continue
        prev = depot
        for node_id in route:
            curr = coords[int(node_id)]
            total += float(np.linalg.norm(curr - prev))
            prev = curr
        total += float(np.linalg.norm(depot - prev))
    return total


def is_feasible_solution(instance: CVRPInstance, routes: list[list[int]]) -> bool:
    """Return True iff ``routes`` cover every customer exactly once within capacity."""
    n = instance.n_nodes
    depot = instance.depot_index
    seen: set[int] = set()
    for route in routes:
        load = 0.0
        for node_id in route:
            node = int(node_id)
            if node == depot or node < 0 or node >= n:
                return False
            if node in seen:
                return False
            seen.add(node)
            load += float(instance.demands[node])
            if load > instance.capacity + 1e-9:
                return False
    expected = set(range(n)) - {depot}
    return seen == expected


def _integer_loads(instance: CVRPInstance) -> tuple[np.ndarray, int]:
    """Return integer (demands, capacity) suitable for discrete OR solvers."""
    demands = instance.demands.astype(np.float64)
    capacity = float(instance.capacity)

    if np.allclose(demands, np.round(demands)) and abs(capacity - round(capacity)) < 1e-9:
        demand_int = np.round(demands).astype(np.int64)
        capacity_int = round(capacity)
    else:
        demand_int = np.maximum(np.rint(demands * LOAD_SCALE).astype(np.int64), 0)
        capacity_int = max(round(capacity * LOAD_SCALE), 1)
        # Rounding must not make a single demand exceed capacity.
        demand_int = np.minimum(demand_int, capacity_int)

    demand_int[instance.depot_index] = 0
    if capacity_int <= 0:
        raise ValueError(f"capacity must be positive after scaling, got {capacity_int}")
    return demand_int, capacity_int


def _min_vehicles(demands: np.ndarray, capacity: int, depot_index: int) -> int:
    """Lower-bound the fleet size by total demand; always at least one vehicle."""
    total = int(demands.sum()) - int(demands[depot_index])
    return max(1, math.ceil(total / capacity))


def _resolve_num_vehicles(
    instance: CVRPInstance,
    demand_int: np.ndarray,
    capacity_int: int,
    *,
    fleet_mode: FleetMode,
    fleet_size: int | None,
) -> int:
    """Return how many vehicles the solver may use."""
    min_v = _min_vehicles(demand_int, capacity_int, instance.depot_index)
    if fleet_mode == "unlimited":
        return max(min_v, instance.n_customers)
    if fleet_size is None:
        raise ValueError(f"fleet_size is required when fleet_mode={fleet_mode!r}")
    size = int(fleet_size)
    if size < min_v:
        raise ValueError(
            f"fleet_size={size} is below the demand lower bound ({min_v}) "
            f"for n_customers={instance.n_customers}"
        )
    if fleet_mode in ("up_to", "exact"):
        return size
    raise ValueError(f"unknown fleet_mode: {fleet_mode!r}")


def _build_pyvrp_stop(
    *,
    time_limit: float | None,
    no_improvement_seconds: float | None,
) -> StoppingCriterion:
    from pyvrp.stop import MaxRuntime, MultipleCriteria

    criteria: list[StoppingCriterion] = []
    if time_limit is not None:
        criteria.append(MaxRuntime(float(time_limit)))
    if no_improvement_seconds is not None:
        criteria.append(_NoImprovementRuntime(float(no_improvement_seconds)))
    if not criteria:
        # Safety: never run forever if the caller forgets both knobs.
        criteria.append(MaxRuntime(_UNLIMITED_SAFETY_SECONDS))
    if len(criteria) == 1:
        return criteria[0]
    return MultipleCriteria(criteria)


def _solve_pyvrp(
    instance: CVRPInstance,
    *,
    time_limit: float | None,
    no_improvement_seconds: float | None,
    seed: int,
    fleet_mode: FleetMode,
    fleet_size: int | None,
) -> list[list[int]]:
    from pyvrp import Client, Depot, Model

    demand_int, capacity_int = _integer_loads(instance)
    coords = instance.coords
    n = instance.n_nodes
    depot_index = instance.depot_index

    # PyVRP requires depots before clients. Build locations as [depot, *customers] and
    # keep a map from PyVRP location index -> our node id.
    customer_nodes = [i for i in range(n) if i != depot_index]
    pyvrp_to_node = [depot_index, *customer_nodes]

    model = Model()
    locations: list[Depot | Client] = [
        model.add_depot(x=float(coords[depot_index, 0]), y=float(coords[depot_index, 1]))
    ]
    for node_id in customer_nodes:
        locations.append(
            model.add_client(
                x=float(coords[node_id, 0]),
                y=float(coords[node_id, 1]),
                delivery=int(demand_int[node_id]),
            )
        )

    num_vehicles = _resolve_num_vehicles(
        instance,
        demand_int,
        capacity_int,
        fleet_mode=fleet_mode,
        fleet_size=fleet_size,
    )
    if fleet_mode == "exact":
        # PyVRP rejects negative fixed_cost; without that lever we can only cap the
        # fleet (same as up_to). Prefer OR-Tools when you need "use exactly X".
        logger.info(
            "PyVRP exact fleet requested; using at-most %d vehicles "
            "(PyVRP fixed_cost must be >= 0)",
            num_vehicles,
        )
    model.add_vehicle_type(num_available=num_vehicles, capacity=capacity_int)

    for i, frm in enumerate(locations):
        for j, to in enumerate(locations):
            if i == j:
                continue
            node_i = pyvrp_to_node[i]
            node_j = pyvrp_to_node[j]
            dist = round(float(np.linalg.norm(coords[node_i] - coords[node_j])) * COORD_SCALE)
            model.add_edge(frm, to, distance=dist)

    stop = _build_pyvrp_stop(
        time_limit=time_limit,
        no_improvement_seconds=no_improvement_seconds,
    )
    result = model.solve(stop=stop, seed=seed, display=False)
    if not result.is_feasible():
        raise RuntimeError("PyVRP returned no feasible solution")

    routes: list[list[int]] = []
    for route in result.best.routes():
        visits = [pyvrp_to_node[int(v)] for v in route.visits()]
        routes.append(visits)
    return routes


def _solve_ortools(
    instance: CVRPInstance,
    *,
    time_limit: float | None,
    no_improvement_seconds: float | None,
    seed: int,
    fleet_mode: FleetMode,
    fleet_size: int | None,
) -> list[list[int]]:
    from ortools.constraint_solver import (  # type: ignore[import-untyped]
        pywrapcp,
        routing_enums_pb2,
    )

    demand_int, capacity_int = _integer_loads(instance)
    coords = instance.coords
    n = instance.n_nodes
    depot_index = instance.depot_index

    distance_matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            distance_matrix[i, j] = round(
                float(np.linalg.norm(coords[i] - coords[j])) * COORD_SCALE
            )

    num_vehicles = _resolve_num_vehicles(
        instance,
        demand_int,
        capacity_int,
        fleet_mode=fleet_mode,
        fleet_size=fleet_size,
    )
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index: int) -> int:
        return int(demand_int[manager.IndexToNode(from_index)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,
        [capacity_int] * num_vehicles,
        True,
        "Capacity",
    )

    if fleet_mode == "exact":
        # Prefer using every vehicle in the fixed fleet via a negative fixed cost.
        # Bound it to just beyond the largest achievable total distance so full-fleet
        # usage dominates while distance stays the tiebreaker. A larger magnitude
        # (e.g. -1e9) would skew Guided Local Search; OR-Tools also dislikes very
        # negative costs. The reported cost is recomputed from route geometry, so this
        # only steers the search, never the labeled cost.
        exact_bonus = -(int(distance_matrix.sum()) + 1)
        for vehicle_id in range(num_vehicles):
            routing.SetFixedCostOfVehicle(exact_bonus, vehicle_id)

    # OR-Tools only exposes a hard time limit; map no-improvement onto that budget
    # when the caller did not set an explicit wall-clock limit.
    if time_limit is not None:
        ortools_limit = float(time_limit)
    elif no_improvement_seconds is not None:
        ortools_limit = float(no_improvement_seconds)
    else:
        ortools_limit = _UNLIMITED_SAFETY_SECONDS

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromMilliseconds(max(1, round(ortools_limit * 1000)))
    # ``seed`` is accepted for API symmetry with PyVRP; this OR-Tools build has no
    # RoutingSearchParameters.random_seed field, so it cannot be threaded through.
    _ = seed

    assignment = routing.SolveWithParameters(search_params)
    if assignment is None:
        raise RuntimeError("OR-Tools returned no feasible solution")

    routes: list[list[int]] = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot_index:
                route.append(int(node))
            index = assignment.Value(routing.NextVar(index))
        if route:
            routes.append(route)
    return routes


def solve_instance(
    instance: CVRPInstance,
    *,
    solver: SolverName = "pyvrp",
    time_limit: float | None = 1.0,
    no_improvement_seconds: float | None = None,
    fleet_mode: FleetMode = "unlimited",
    fleet_size: int | None = None,
    seed: int = 0,
    instance_id: int = 0,
) -> CVRPSolution:
    """Solve one CVRP instance and return a labeled :class:`CVRPSolution`.

    Args:
        time_limit: wall-clock seconds; ``None`` means no hard time limit.
        no_improvement_seconds: stop after this many seconds without improving the
            best cost (PyVRP). OR-Tools maps this to its time budget when
            ``time_limit`` is ``None``.
        fleet_mode: ``unlimited``, ``up_to`` (at most ``fleet_size``), or
            ``exact`` (prefer using exactly ``fleet_size`` vehicles).
        fleet_size: required when ``fleet_mode`` is ``up_to`` or ``exact``.
    """
    instance.validate()
    if time_limit is not None and time_limit <= 0:
        raise ValueError(f"time_limit must be positive or None, got {time_limit}")
    if no_improvement_seconds is not None and no_improvement_seconds <= 0:
        raise ValueError(
            f"no_improvement_seconds must be positive or None, got {no_improvement_seconds}"
        )
    if time_limit is None and no_improvement_seconds is None:
        raise ValueError("set time_limit and/or no_improvement_seconds")

    started = time.perf_counter()
    if solver == "pyvrp":
        routes = _solve_pyvrp(
            instance,
            time_limit=time_limit,
            no_improvement_seconds=no_improvement_seconds,
            seed=seed,
            fleet_mode=fleet_mode,
            fleet_size=fleet_size,
        )
    elif solver == "ortools":
        routes = _solve_ortools(
            instance,
            time_limit=time_limit,
            no_improvement_seconds=no_improvement_seconds,
            seed=seed,
            fleet_mode=fleet_mode,
            fleet_size=fleet_size,
        )
    else:
        raise ValueError(f"unknown solver: {solver!r}")
    runtime = time.perf_counter() - started

    cost = route_cost(routes, instance.coords, depot_index=instance.depot_index)
    feasible = is_feasible_solution(instance, routes)
    solution = CVRPSolution(
        instance_id=instance_id,
        n_customers=instance.n_customers,
        routes=routes,
        cost=float(cost),
        num_vehicles=len(routes),
        feasible=feasible,
        solver_name=solver,
        runtime_seconds=float(runtime),
        seed=seed,
        fleet_mode=fleet_mode,
        fleet_size=None if fleet_mode == "unlimited" else fleet_size,
    )
    if not feasible:
        logger.warning(
            "solver=%s instance_id=%d produced an infeasible labeling", solver, instance_id
        )
    return solution


def solve_dataset(
    dataset: CVRPDataset,
    *,
    solver: SolverName = "pyvrp",
    time_limit: float | None = 1.0,
    no_improvement_seconds: float | None = None,
    fleet_mode: FleetMode = "unlimited",
    fleet_size: int | None = None,
    seed: int = 0,
) -> list[CVRPSolution]:
    """Label every instance in ``dataset`` with feasible routes and costs."""
    solutions: list[CVRPSolution] = []
    for instance_id, instance in enumerate(dataset):
        # Derive a per-instance seed so the batch is reproducible but uncorrelated.
        instance_seed = int(np.random.SeedSequence([seed, instance_id]).generate_state(1)[0])
        solution = solve_instance(
            instance,
            solver=solver,
            time_limit=time_limit,
            no_improvement_seconds=no_improvement_seconds,
            fleet_mode=fleet_mode,
            fleet_size=fleet_size,
            seed=instance_seed,
            instance_id=instance_id,
        )
        solutions.append(solution)
        logger.info(
            "labeled instance_id=%d n_customers=%d cost=%.4f vehicles=%d feasible=%s "
            "runtime=%.3fs solver=%s",
            solution.instance_id,
            solution.n_customers,
            solution.cost,
            solution.num_vehicles,
            solution.feasible,
            solution.runtime_seconds,
            solution.solver_name,
        )
    return solutions


def save_labels(solutions: list[CVRPSolution], path: str | Path) -> Path:
    """Write a batch of labeled solutions to ``path`` as JSON and return that path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = [asdict(solution) for solution in solutions]
    target.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("saved %d labels to %s", len(solutions), target)
    return target
