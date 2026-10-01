"""Neighborhood selection and subproblem extraction for local refinement (task Q1.2).

Selectors rank small parts of a feasible CVRP solution that are worth re-optimizing and return
solver-agnostic :class:`Neighborhood` objects. :func:`extract_reorder_subproblem` and
:func:`extract_exchange_subproblem` turn a neighborhood into a self-contained subproblem, and
:func:`apply_reorder` / :func:`apply_exchange` splice a solved subproblem back into the routes.
QUBO builders (``quantum/``) and classical baselines (``local_search/``) consume exactly the same
subproblem objects, which is what makes their comparison matched.

Routes use 0-based customer ids with the depot omitted, as everywhere in the project.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.eval.routing import order_route_nearest_neighbor_two_opt
from vrp_diffusion_quantum.quantum._validation import binary_vector
from vrp_diffusion_quantum.utils.feasibility import route_cost

logger = logging.getLogger(__name__)

__all__ = [
    "ExchangeSubproblem",
    "Neighborhood",
    "NeighborhoodConfig",
    "NeighborhoodType",
    "ReorderSubproblem",
    "apply_exchange",
    "apply_reorder",
    "evaluate_exchange",
    "extract_exchange_subproblem",
    "extract_reorder_subproblem",
    "route_affinities",
    "select_high_cost_routes",
    "select_low_confidence_edges",
    "select_neighborhoods",
    "select_two_route_exchanges",
    "select_uncertain_customers",
]

NeighborhoodType = Literal[
    "high_cost_route", "uncertain_m", "low_confidence_edges", "two_route_exchange"
]
SubproblemKind = Literal["reorder", "exchange"]

_ALL_TYPES: tuple[NeighborhoodType, ...] = (
    "high_cost_route",
    "uncertain_m",
    "low_confidence_edges",
    "two_route_exchange",
)


@dataclass(frozen=True)
class Neighborhood:
    """A small part of a solution selected for refinement.

    For ``kind == "reorder"``, ``route_indices`` holds one route, ``segment`` the half-open
    position range ``[start, end)`` inside it, and ``customers`` that segment in current order.
    For ``kind == "exchange"``, ``route_indices`` holds two routes and ``customers`` the movable
    customers; every other customer of the two routes stays where it is.
    """

    neighborhood_type: NeighborhoodType
    kind: SubproblemKind
    route_indices: tuple[int, ...]
    customers: tuple[int, ...]
    score: float
    segment: tuple[int, int] | None = None

    @property
    def size(self) -> int:
        """Number of free customers (the logged ``neighborhood_size``)."""
        return len(self.customers)


@dataclass(frozen=True)
class NeighborhoodConfig:
    """Size limits and thresholds for the selectors (defaults follow docs/quantum_scope.md)."""

    max_reorder_size: int = 5
    max_exchange_size: int = 8
    max_per_type: int = 10
    uncertainty_margin: float = 0.2
    low_confidence_threshold: float = 0.5
    exchange_route_pairs_per_route: int = 2

    def __post_init__(self) -> None:
        if self.max_reorder_size < 2:
            raise ValueError("max_reorder_size must be >= 2")
        if self.max_exchange_size < 1:
            raise ValueError("max_exchange_size must be >= 1")
        if self.max_per_type < 1:
            raise ValueError("max_per_type must be >= 1")
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be in [0, 1]")
        if self.exchange_route_pairs_per_route < 1:
            raise ValueError("exchange_route_pairs_per_route must be >= 1")


@dataclass(frozen=True)
class ReorderSubproblem:
    """Order ``customers`` on a path from ``start_node`` to ``end_node`` (``None`` = depot).

    ``distances`` has shape ``[k + 2, k + 2]``: index 0 is the start node, ``1..k`` the free
    customers in the order of ``customers``, and ``k + 1`` the end node.
    """

    route_index: int
    segment: tuple[int, int]
    customers: tuple[int, ...]
    start_node: int | None
    end_node: int | None
    distances: npt.NDArray[np.float64]
    n_customers: int | None = None

    @property
    def size(self) -> int:
        return len(self.customers)

    def path_cost(self, order: Sequence[int]) -> float:
        """Cost of visiting ``order`` (a permutation of ``customers``) between the endpoints."""
        if sorted(order) != sorted(self.customers):
            raise ValueError("order must be a permutation of the subproblem customers")
        position = {customer: index + 1 for index, customer in enumerate(self.customers)}
        nodes = [0, *(position[customer] for customer in order), self.size + 1]
        return float(sum(self.distances[a, b] for a, b in itertools.pairwise(nodes)))

    @property
    def current_cost(self) -> float:
        return self.path_cost(self.customers)


@dataclass(frozen=True)
class ExchangeSubproblem:
    """Decide which of two routes serves each movable customer.

    ``assignment[i] == 1`` places ``movable[i]`` in ``route_indices[0]`` and ``0`` in
    ``route_indices[1]``. Fixed customers keep their route. True costs are evaluated by
    :func:`evaluate_exchange`, which orders each resulting route by nearest neighbour plus 2-opt.
    Extracted subproblems retain ``initial_routes`` for the actual starting cost and stale-input
    checks, and ``n_customers`` for validating full-instance probability matrices.
    """

    route_indices: tuple[int, int]
    movable: tuple[int, ...]
    fixed: tuple[tuple[int, ...], tuple[int, ...]]
    initial_assignment: tuple[int, ...]
    demands: npt.NDArray[np.float64]
    fixed_loads: tuple[float, float]
    capacity: float
    initial_routes: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    n_customers: int | None = None

    @property
    def size(self) -> int:
        return len(self.movable)

    def members(self, assignment: Sequence[int]) -> tuple[list[int], list[int]]:
        """Unordered customers of both routes under ``assignment``."""
        chosen = binary_vector(assignment, self.size, name="assignment")
        first = list(self.fixed[0]) + [c for c, a in zip(self.movable, chosen, strict=True) if a]
        second = list(self.fixed[1]) + [
            c for c, a in zip(self.movable, chosen, strict=True) if not a
        ]
        return first, second

    def loads(self, assignment: Sequence[int]) -> tuple[float, float]:
        chosen = binary_vector(assignment, self.size, name="assignment")
        moved = float(self.demands @ chosen)
        total = float(self.demands.sum())
        return self.fixed_loads[0] + moved, self.fixed_loads[1] + (total - moved)

    def is_feasible(self, assignment: Sequence[int], *, tolerance: float = 1e-9) -> bool:
        first, second = self.loads(assignment)
        return first <= self.capacity + tolerance and second <= self.capacity + tolerance


# --------------------------------------------------------------------------------------------
# Geometry helpers


def _customer_xy(instance: CVRPInstance, customer: int) -> npt.NDArray[np.float64]:
    return np.asarray(instance.coords[instance.customer_node_indices()[customer]], dtype=float)


def _node_xy(instance: CVRPInstance, customer: int | None) -> npt.NDArray[np.float64]:
    if customer is None:
        return np.asarray(instance.coords[instance.depot_index], dtype=float)
    return _customer_xy(instance, customer)


def _distance(instance: CVRPInstance, first: int | None, second: int | None) -> float:
    return float(np.linalg.norm(_node_xy(instance, first) - _node_xy(instance, second)))


def _segment_window(route_length: int, edge: int, size: int) -> tuple[int, int]:
    """Window of ``size`` positions around route edge ``edge`` (edge e joins positions e-1, e)."""
    size = min(size, route_length)
    start = max(0, min(edge - size // 2, route_length - size))
    return start, start + size


def _validate_routes(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> None:
    seen: set[int] = set()
    for route in routes:
        for customer in route:
            if not 0 <= customer < instance.n_customers:
                raise ValueError(f"customer {customer} out of range")
            if customer in seen:
                raise ValueError(f"customer {customer} appears more than once")
            seen.add(customer)


def _validate_m_prob(instance: CVRPInstance, m_prob: npt.ArrayLike) -> npt.NDArray[np.float64]:
    matrix = np.asarray(m_prob, dtype=np.float64)
    n = instance.n_customers
    if matrix.shape != (n, n):
        raise ValueError(f"m_prob must have shape ({n}, {n}), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("m_prob must contain finite probabilities in [0, 1]")
    return matrix


# --------------------------------------------------------------------------------------------
# Selectors


def select_high_cost_routes(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    config: NeighborhoodConfig | None = None,
) -> list[Neighborhood]:
    """Routes with the highest cost per customer, cut around their most expensive edge."""
    cfg = config or NeighborhoodConfig()
    _validate_routes(instance, routes)
    selected: list[Neighborhood] = []
    for index, route in enumerate(routes):
        if len(route) < 2:
            continue
        cost = route_cost(instance, [list(route)])
        nodes: list[int | None] = [None, *route, None]
        edge_costs = [_distance(instance, a, b) for a, b in itertools.pairwise(nodes)]
        worst_edge = int(np.argmax(edge_costs))
        start, end = _segment_window(len(route), worst_edge, cfg.max_reorder_size)
        selected.append(
            Neighborhood(
                neighborhood_type="high_cost_route",
                kind="reorder",
                route_indices=(index,),
                customers=tuple(route[start:end]),
                score=cost / len(route),
                segment=(start, end),
            )
        )
    selected.sort(key=lambda item: (-item.score, item.route_indices))
    return selected[: cfg.max_per_type]


def route_affinities(
    m_prob: npt.ArrayLike, routes: Sequence[Sequence[int]], customer: int
) -> list[float]:
    """Mean predicted same-route probability of ``customer`` to each route's other members."""
    matrix = np.asarray(m_prob, dtype=np.float64)
    affinities: list[float] = []
    for route in routes:
        others = [member for member in route if member != customer]
        affinities.append(float(matrix[customer, others].mean()) if others else 0.0)
    return affinities


def select_uncertain_customers(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    m_prob: npt.ArrayLike,
    config: NeighborhoodConfig | None = None,
) -> list[Neighborhood]:
    """Two-route exchanges around customers whose own-route affinity barely wins (or loses).

    A customer is uncertain when its mean ``m_prob`` to its own route exceeds that to the best
    other route by less than ``uncertainty_margin``. Uncertain customers are grouped by the pair
    (own route, best other route); each pair becomes one exchange neighborhood.
    """
    cfg = config or NeighborhoodConfig()
    _validate_routes(instance, routes)
    matrix = _validate_m_prob(instance, m_prob)
    if len(routes) < 2:
        return []
    groups: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for own_index, route in enumerate(routes):
        for customer in route:
            affinities = route_affinities(matrix, routes, customer)
            own = affinities[own_index]
            others = [
                (value, index) for index, value in enumerate(affinities) if index != own_index
            ]
            best_value, best_index = max(others, key=lambda item: (item[0], -item[1]))
            margin = own - best_value
            if margin < cfg.uncertainty_margin:
                pair = (min(own_index, best_index), max(own_index, best_index))
                groups.setdefault(pair, []).append((margin, customer))
    selected: list[Neighborhood] = []
    for pair, members in groups.items():
        members.sort()
        chosen = members[: cfg.max_exchange_size]
        score = float(sum(cfg.uncertainty_margin - margin for margin, _ in chosen))
        selected.append(
            Neighborhood(
                neighborhood_type="uncertain_m",
                kind="exchange",
                route_indices=pair,
                customers=tuple(customer for _, customer in chosen),
                score=score,
            )
        )
    selected.sort(key=lambda item: (-item.score, item.route_indices))
    return selected[: cfg.max_per_type]


def select_low_confidence_edges(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    edge_prob: npt.ArrayLike,
    config: NeighborhoodConfig | None = None,
) -> list[Neighborhood]:
    """Route segments around consecutive customers whose predicted pair probability is low.

    ``edge_prob`` is any symmetric pair-probability matrix; the predicted constraint matrix
    ``m_prob`` is the default source. Each edge below ``low_confidence_threshold`` yields one
    reorder neighborhood centred on it; overlapping windows in a route keep the lowest edge.
    """
    cfg = config or NeighborhoodConfig()
    _validate_routes(instance, routes)
    matrix = _validate_m_prob(instance, edge_prob)
    selected: list[Neighborhood] = []
    for index, route in enumerate(routes):
        if len(route) < 2:
            continue
        taken: set[int] = set()
        edges = sorted(
            (float(matrix[route[pos - 1], route[pos]]), pos) for pos in range(1, len(route))
        )
        for confidence, pos in edges:
            if confidence >= cfg.low_confidence_threshold:
                break
            window = _segment_window(len(route), pos, cfg.max_reorder_size)
            start, end = window
            if any(position in taken for position in range(start, end)):
                continue
            taken.update(range(start, end))
            selected.append(
                Neighborhood(
                    neighborhood_type="low_confidence_edges",
                    kind="reorder",
                    route_indices=(index,),
                    customers=tuple(route[start:end]),
                    score=cfg.low_confidence_threshold - confidence,
                    segment=window,
                )
            )
    selected.sort(key=lambda item: (-item.score, item.route_indices, item.segment))
    return selected[: cfg.max_per_type]


def _nearest_first(
    instance: CVRPInstance, customers: Sequence[int], target: npt.NDArray[np.float64]
) -> list[int]:
    """Customers sorted by distance to ``target`` (ties by customer id)."""

    def distance(customer: int) -> tuple[float, int]:
        return float(np.linalg.norm(_customer_xy(instance, customer) - target)), customer

    return sorted(customers, key=distance)


def select_two_route_exchanges(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    config: NeighborhoodConfig | None = None,
) -> list[Neighborhood]:
    """Spatially adjacent route pairs; movable customers are those nearest the other route.

    Each route is paired with its ``exchange_route_pairs_per_route`` nearest routes (by the
    closest pair of customers). Movable customers alternate between the two routes, taking each
    route's customers in order of distance to the other route's centroid.
    """
    cfg = config or NeighborhoodConfig()
    _validate_routes(instance, routes)
    active = [index for index, route in enumerate(routes) if route]
    if len(active) < 2:
        return []
    xy = {i: np.stack([_customer_xy(instance, c) for c in routes[i]]) for i in active}
    gaps: dict[tuple[int, int], float] = {}
    for position, first in enumerate(active):
        for second in active[position + 1 :]:
            delta = xy[first][:, None, :] - xy[second][None, :, :]
            gaps[(first, second)] = float(np.sqrt((delta**2).sum(axis=-1)).min())
    pairs: set[tuple[int, int]] = set()
    for route_index in active:
        nearest = sorted((gap, pair) for pair, gap in gaps.items() if route_index in pair)[
            : cfg.exchange_route_pairs_per_route
        ]
        pairs.update(pair for _, pair in nearest)
    selected: list[Neighborhood] = []
    for first, second in sorted(pairs):
        centroid = {first: xy[second].mean(axis=0), second: xy[first].mean(axis=0)}
        ranked = {
            route_index: _nearest_first(instance, routes[route_index], centroid[route_index])
            for route_index in (first, second)
        }
        movable: list[int] = []
        for position in range(max(len(ranked[first]), len(ranked[second]))):
            for route_index in (first, second):
                if position < len(ranked[route_index]) and len(movable) < cfg.max_exchange_size:
                    movable.append(ranked[route_index][position])
        selected.append(
            Neighborhood(
                neighborhood_type="two_route_exchange",
                kind="exchange",
                route_indices=(first, second),
                customers=tuple(movable),
                score=-gaps[(first, second)],
            )
        )
    selected.sort(key=lambda item: (-item.score, item.route_indices))
    return selected[: cfg.max_per_type]


def select_neighborhoods(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    *,
    m_prob: npt.ArrayLike | None = None,
    config: NeighborhoodConfig | None = None,
    types: Sequence[NeighborhoodType] = _ALL_TYPES,
) -> list[Neighborhood]:
    """Run the requested selectors in a fixed order. Matrix-based types need ``m_prob``."""
    unknown = set(types) - set(_ALL_TYPES)
    if unknown:
        raise ValueError(f"unknown neighborhood types: {sorted(unknown)}")
    needs_matrix = {"uncertain_m", "low_confidence_edges"} & set(types)
    if needs_matrix and m_prob is None:
        raise ValueError(f"{sorted(needs_matrix)} require m_prob")
    selected: list[Neighborhood] = []
    for neighborhood_type in _ALL_TYPES:
        if neighborhood_type not in types:
            continue
        if neighborhood_type == "high_cost_route":
            selected += select_high_cost_routes(instance, routes, config)
        elif neighborhood_type == "uncertain_m":
            assert m_prob is not None
            selected += select_uncertain_customers(instance, routes, m_prob, config)
        elif neighborhood_type == "low_confidence_edges":
            assert m_prob is not None
            selected += select_low_confidence_edges(instance, routes, m_prob, config)
        else:
            selected += select_two_route_exchanges(instance, routes, config)
    logger.debug("selected %d neighborhoods", len(selected))
    return selected


# --------------------------------------------------------------------------------------------
# Subproblems


def extract_reorder_subproblem(
    instance: CVRPInstance, routes: Sequence[Sequence[int]], neighborhood: Neighborhood
) -> ReorderSubproblem:
    """Fix the nodes adjacent to the segment and build its distance matrix."""
    if neighborhood.kind != "reorder" or neighborhood.segment is None:
        raise ValueError("neighborhood is not a reorder neighborhood")
    (route_index,) = neighborhood.route_indices
    route = list(routes[route_index])
    start, end = neighborhood.segment
    if not 0 <= start < end <= len(route):
        raise ValueError("segment is outside the route")
    if tuple(route[start:end]) != neighborhood.customers:
        raise ValueError("neighborhood no longer matches the route; reselect it")
    start_node = route[start - 1] if start > 0 else None
    end_node = route[end] if end < len(route) else None
    nodes: list[int | None] = [start_node, *neighborhood.customers, end_node]
    xy = np.stack([_node_xy(instance, node) for node in nodes])
    distances = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    return ReorderSubproblem(
        route_index=route_index,
        segment=(start, end),
        customers=neighborhood.customers,
        start_node=start_node,
        end_node=end_node,
        distances=distances,
        n_customers=instance.n_customers,
    )


def apply_reorder(
    routes: Sequence[Sequence[int]], subproblem: ReorderSubproblem, order: Sequence[int]
) -> list[list[int]]:
    """Replace the segment, rejecting changes to its original customers or fixed endpoints."""
    if sorted(order) != sorted(subproblem.customers):
        raise ValueError("order must be a permutation of the subproblem customers")
    updated = [list(route) for route in routes]
    start, end = subproblem.segment
    if not 0 <= subproblem.route_index < len(updated):
        raise ValueError("subproblem route no longer exists; reselect it")
    route = updated[subproblem.route_index]
    if not 0 <= start < end <= len(route) or tuple(route[start:end]) != subproblem.customers:
        raise ValueError("subproblem no longer matches the route; reselect it")
    start_node = route[start - 1] if start > 0 else None
    end_node = route[end] if end < len(route) else None
    if (start_node, end_node) != (subproblem.start_node, subproblem.end_node):
        raise ValueError("subproblem endpoints have changed; reselect it")
    updated[subproblem.route_index] = route[:start] + list(order) + route[end:]
    return updated


def extract_exchange_subproblem(
    instance: CVRPInstance, routes: Sequence[Sequence[int]], neighborhood: Neighborhood
) -> ExchangeSubproblem:
    """Split two routes into fixed customers and movable customers with their demands."""
    if neighborhood.kind != "exchange" or len(neighborhood.route_indices) != 2:
        raise ValueError("neighborhood is not a two-route exchange neighborhood")
    first, second = neighborhood.route_indices
    if first == second:
        raise ValueError("exchange routes must differ")
    members = (list(routes[first]), list(routes[second]))
    movable = neighborhood.customers
    in_first = set(members[0])
    in_either = in_first | set(members[1])
    if any(customer not in in_either for customer in movable):
        raise ValueError("movable customers must belong to the two routes")
    if len(set(movable)) != len(movable):
        raise ValueError("movable customers must be unique")
    movable_set = set(movable)
    fixed = (
        tuple(c for c in members[0] if c not in movable_set),
        tuple(c for c in members[1] if c not in movable_set),
    )
    customer_demands = instance.customer_demands()
    fixed_loads = (
        float(sum(customer_demands[c] for c in fixed[0])),
        float(sum(customer_demands[c] for c in fixed[1])),
    )
    return ExchangeSubproblem(
        route_indices=(first, second),
        movable=movable,
        fixed=fixed,
        initial_assignment=tuple(int(c in in_first) for c in movable),
        demands=np.asarray([customer_demands[c] for c in movable], dtype=np.float64),
        fixed_loads=fixed_loads,
        capacity=float(instance.capacity),
        initial_routes=(tuple(members[0]), tuple(members[1])),
        n_customers=instance.n_customers,
    )


def evaluate_exchange(
    instance: CVRPInstance, subproblem: ExchangeSubproblem, assignment: Sequence[int]
) -> tuple[float, list[int], list[int]]:
    """True cost of both routes under ``assignment``, each ordered by nearest neighbour + 2-opt.

    Returns ``(cost, ordered_first, ordered_second)``. Capacity is not checked here; use
    :meth:`ExchangeSubproblem.is_feasible`.
    """
    first, second = subproblem.members(assignment)
    ordered_first = order_route_nearest_neighbor_two_opt(instance, sorted(first))
    ordered_second = order_route_nearest_neighbor_two_opt(instance, sorted(second))
    routes = [route for route in (ordered_first, ordered_second) if route]
    return route_cost(instance, routes), ordered_first, ordered_second


def apply_exchange(
    routes: Sequence[Sequence[int]],
    subproblem: ExchangeSubproblem,
    ordered_first: Sequence[int],
    ordered_second: Sequence[int],
) -> list[list[int]]:
    """Replace both routes, preserving fixed memberships and rejecting stale input routes.

    An emptied route is kept empty. Capacity is checked by the refinement acceptance step.
    """
    expected = sorted([*subproblem.fixed[0], *subproblem.fixed[1], *subproblem.movable])
    if sorted([*ordered_first, *ordered_second]) != expected:
        raise ValueError("new routes must contain exactly the subproblem's customers")
    if not set(subproblem.fixed[0]).issubset(ordered_first) or not set(
        subproblem.fixed[1]
    ).issubset(ordered_second):
        raise ValueError("fixed customers must stay in their original routes")
    updated = [list(route) for route in routes]
    first, second = subproblem.route_indices
    if first == second or any(index < 0 or index >= len(updated) for index in (first, second)):
        raise ValueError("subproblem routes no longer exist or are not distinct; reselect it")
    if subproblem.initial_routes is not None:
        if (tuple(updated[first]), tuple(updated[second])) != subproblem.initial_routes:
            raise ValueError("subproblem no longer matches the routes; reselect it")
    else:
        initial_first, initial_second = subproblem.members(subproblem.initial_assignment)
        if sorted(updated[first]) != sorted(initial_first) or sorted(updated[second]) != sorted(
            initial_second
        ):
            raise ValueError("subproblem no longer matches the routes; reselect it")
    updated[first] = list(ordered_first)
    updated[second] = list(ordered_second)
    return updated
