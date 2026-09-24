"""Decode route-membership predictions and evaluate their routing utility."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance
from vrp_diffusion_quantum.eval.confidence import bootstrap_mean_interval
from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

__all__ = [
    "RoutingEvaluation",
    "decode_matrix_to_routes",
    "evaluate_decoded_matrix",
    "order_route_nearest_neighbor_two_opt",
    "summarize_routing_evaluations",
]


@dataclass(frozen=True)
class RoutingEvaluation:
    """Routing metrics for one decoded matrix prediction."""

    instance_id: str
    n_customers: int
    feasible: bool
    violation_count: int
    capacity_violation_count: int
    num_vehicles: int
    reference_num_vehicles: int
    vehicle_delta: int
    vehicle_inflation_ratio: float
    decoded_cost: float
    reference_cost: float
    cost_gap: float
    cost_gap_percent: float
    min_route_size: int
    max_route_size: int
    mean_route_size: float
    singleton_route_count: int
    singleton_route_fraction: float
    route_size_histogram: dict[int, int]
    decode_runtime_seconds: float
    repair_count: int
    matrix_pair_accuracy: float
    matrix_true_positive_pairs: int
    matrix_false_positive_pairs: int
    matrix_false_negative_pairs: int
    matrix_positive_edge_precision: float
    matrix_positive_edge_recall: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_scores(matrix: npt.ArrayLike, n_customers: int) -> npt.NDArray[np.float64]:
    scores = np.asarray(matrix, dtype=np.float64)
    if scores.shape != (n_customers, n_customers):
        raise ValueError(
            f"matrix shape {scores.shape} does not match ({n_customers}, {n_customers})"
        )
    if not np.isfinite(scores).all():
        raise ValueError("matrix must contain only finite values")
    scores = np.clip((scores + scores.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(scores, 0.0)
    return scores


def order_route_nearest_neighbor_two_opt(instance: CVRPInstance, customers: list[int]) -> list[int]:
    """Order one route by deterministic nearest neighbour followed by 2-opt."""
    if len(customers) < 2:
        return customers.copy()
    nodes = instance.customer_node_indices()
    depot = instance.coords[instance.depot_index]
    remaining = set(customers)
    first = min(
        remaining,
        key=lambda c: (float(np.linalg.norm(instance.coords[nodes[c]] - depot)), c),
    )
    route = [first]
    remaining.remove(first)
    while remaining:
        previous = instance.coords[nodes[route[-1]]]
        nxt = min(
            remaining,
            key=lambda c: (float(np.linalg.norm(instance.coords[nodes[c]] - previous)), c),
        )
        route.append(nxt)
        remaining.remove(nxt)

    best_cost = route_cost(instance, [route])
    improved = True
    while improved:
        improved = False
        for start in range(len(route) - 1):
            for end in range(start + 2, len(route) + 1):
                candidate = route[:start] + list(reversed(route[start:end])) + route[end:]
                candidate_cost = route_cost(instance, [candidate])
                if candidate_cost < best_cost - 1e-12:
                    route, best_cost, improved = candidate, candidate_cost, True
                    break
            if improved:
                break
    return route


def decode_matrix_to_routes(
    instance: CVRPInstance,
    matrix: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> tuple[list[list[int]], int]:
    """Decode pair probabilities with capacity-constrained agglomerative clustering.

    Customers begin as singleton routes. Candidate route pairs are merged in descending mean
    cross-route affinity while the merged demand fits vehicle capacity. Merges below ``threshold``
    are rejected. The returned routes always cover every customer exactly once when each individual
    demand fits capacity. ``repair_count`` counts affinity components split by the capacity gate.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    scores = _validate_scores(matrix, instance.n_customers)
    demands = instance.customer_demands()
    oversized = np.flatnonzero(demands > instance.capacity + 1e-9)
    if oversized.size:
        raise ValueError(f"customer {int(oversized[0])} demand exceeds vehicle capacity")

    clusters: list[list[int]] = [[i] for i in range(instance.n_customers)]
    loads = [float(demand) for demand in demands]
    repairs = 0
    while True:
        best: tuple[float, int, int] | None = None
        blocked = False
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                affinity = float(np.mean(scores[np.ix_(clusters[left], clusters[right])]))
                if affinity < threshold:
                    continue
                if loads[left] + loads[right] > instance.capacity + 1e-9:
                    blocked = True
                    continue
                candidate = (affinity, -left, -right)
                if best is None or candidate > (best[0], -best[1], -best[2]):
                    best = (affinity, left, right)
        if best is None:
            repairs += int(blocked)
            break
        _, left, right = best
        clusters[left] = sorted(clusters[left] + clusters[right])
        loads[left] += loads[right]
        del clusters[right]
        del loads[right]

    routes = [order_route_nearest_neighbor_two_opt(instance, cluster) for cluster in clusters]
    routes.sort(key=lambda route: route[0] if route else instance.n_customers)
    return routes, repairs


def evaluate_decoded_matrix(
    example: CVRPExample,
    matrix: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> RoutingEvaluation:
    """Decode a predicted matrix and compare its feasible route set with the reference."""
    started = time.perf_counter()
    routes, repair_count = decode_matrix_to_routes(example.instance, matrix, threshold=threshold)
    elapsed = time.perf_counter() - started
    report = validate_routes(example.instance, routes)
    decoded_cost = route_cost(example.instance, routes) if report.feasible else math.inf
    reference_cost = route_cost(example.instance, example.solution.routes)
    gap = decoded_cost - reference_cost
    gap_percent = 100.0 * gap / reference_cost if reference_cost > 0 else math.nan
    sizes = [len(route) for route in routes]
    decoded_matrix = build_constraint_matrix(routes, example.instance.n_customers)
    true_matrix = example.constraint_matrix
    upper = np.triu_indices(example.instance.n_customers, k=1)
    decoded_pairs = decoded_matrix[upper].astype(bool)
    true_pairs = true_matrix[upper].astype(bool)
    pair_accuracy = float(np.mean(decoded_pairs == true_pairs)) if upper[0].size else 1.0
    true_positives = int(np.sum(decoded_pairs & true_pairs))
    false_positives = int(np.sum(decoded_pairs & ~true_pairs))
    false_negatives = int(np.sum(~decoded_pairs & true_pairs))
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    positive_precision = true_positives / precision_denominator if precision_denominator else 0.0
    positive_recall = true_positives / recall_denominator if recall_denominator else 0.0
    size_histogram = {size: sizes.count(size) for size in sorted(set(sizes))}
    singleton_count = size_histogram.get(1, 0)
    reference_num_vehicles = len(example.solution.routes)
    return RoutingEvaluation(
        instance_id=example.instance.instance_id,
        n_customers=example.instance.n_customers,
        feasible=report.feasible,
        violation_count=len(report.violations),
        capacity_violation_count=sum("exceeds capacity" in item for item in report.violations),
        num_vehicles=len(routes),
        reference_num_vehicles=reference_num_vehicles,
        vehicle_delta=len(routes) - reference_num_vehicles,
        vehicle_inflation_ratio=(
            len(routes) / reference_num_vehicles if reference_num_vehicles else math.nan
        ),
        decoded_cost=decoded_cost,
        reference_cost=reference_cost,
        cost_gap=gap,
        cost_gap_percent=gap_percent,
        min_route_size=min(sizes, default=0),
        max_route_size=max(sizes, default=0),
        mean_route_size=float(np.mean(sizes)) if sizes else 0.0,
        singleton_route_count=singleton_count,
        singleton_route_fraction=singleton_count / len(routes) if routes else 0.0,
        route_size_histogram=size_histogram,
        decode_runtime_seconds=elapsed,
        repair_count=repair_count,
        matrix_pair_accuracy=pair_accuracy,
        matrix_true_positive_pairs=true_positives,
        matrix_false_positive_pairs=false_positives,
        matrix_false_negative_pairs=false_negatives,
        matrix_positive_edge_precision=positive_precision,
        matrix_positive_edge_recall=positive_recall,
    )


def summarize_routing_evaluations(
    results: list[RoutingEvaluation],
    *,
    confidence_level: float | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, float | int]:
    """Aggregate decoded routing results without hiding infeasible cases."""
    if not results:
        raise ValueError("cannot summarize an empty routing evaluation")
    feasible = [result for result in results if result.feasible]
    true_positives = sum(result.matrix_true_positive_pairs for result in results)
    false_positives = sum(result.matrix_false_positive_pairs for result in results)
    false_negatives = sum(result.matrix_false_negative_pairs for result in results)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    summary: dict[str, float | int] = {
        "route_num_examples": len(results),
        "route_feasible_count": len(feasible),
        "route_feasible_rate": len(feasible) / len(results),
        "route_violation_count": sum(result.violation_count for result in results),
        "route_capacity_violation_count": sum(
            result.capacity_violation_count for result in results
        ),
        "route_mean_cost_gap_percent": (
            float(np.mean([result.cost_gap_percent for result in feasible]))
            if feasible
            else math.nan
        ),
        "route_mean_num_vehicles": float(np.mean([result.num_vehicles for result in results])),
        "route_mean_vehicle_delta": float(np.mean([result.vehicle_delta for result in results])),
        "route_mean_vehicle_inflation_ratio": float(
            np.mean([result.vehicle_inflation_ratio for result in results])
        ),
        "route_singleton_route_count": sum(result.singleton_route_count for result in results),
        "route_mean_singleton_route_fraction": float(
            np.mean([result.singleton_route_fraction for result in results])
        ),
        "route_mean_decode_runtime_seconds": float(
            np.mean([result.decode_runtime_seconds for result in results])
        ),
        "route_repair_count": sum(result.repair_count for result in results),
        "route_mean_matrix_pair_accuracy": float(
            np.mean([result.matrix_pair_accuracy for result in results])
        ),
        "route_matrix_true_positive_pairs": true_positives,
        "route_matrix_false_positive_pairs": false_positives,
        "route_matrix_false_negative_pairs": false_negatives,
        "route_matrix_positive_edge_precision": (
            true_positives / precision_denominator if precision_denominator else 0.0
        ),
        "route_matrix_positive_edge_recall": (
            true_positives / recall_denominator if recall_denominator else 0.0
        ),
    }
    if confidence_level is not None and feasible:
        interval = bootstrap_mean_interval(
            [result.cost_gap_percent for result in feasible],
            confidence_level=confidence_level,
            num_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        summary.update(
            {
                "route_mean_cost_gap_percent_ci_lower": interval.lower,
                "route_mean_cost_gap_percent_ci_upper": interval.upper,
                "route_mean_cost_gap_percent_ci_confidence_level": interval.confidence_level,
                "route_mean_cost_gap_percent_ci_num_resamples": interval.num_resamples,
                "route_mean_cost_gap_percent_ci_num_observations": interval.num_observations,
                "route_mean_cost_gap_percent_ci_seed": interval.seed,
            }
        )
    return summary
