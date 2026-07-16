"""Route cost and feasibility utilities for CVRP instances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from vrp_diffusion_quantum.data.types import CVRPInstance, LabeledSolution


@dataclass(frozen=True)
class FeasibilityReport:
    """Detailed feasibility result for a route set."""

    feasible: bool
    violations: tuple[str, ...]
    route_loads: tuple[float, ...]
    missing_customers: tuple[int, ...]
    duplicate_customers: tuple[int, ...]
    out_of_range_customers: tuple[int, ...]


def _normalize_customer_index(customer: object, n_customers: int) -> int:
    if isinstance(customer, bool) or not isinstance(customer, Integral):
        raise ValueError(f"customer index {customer!r} must be an integer")
    customer_id = int(customer)
    if not 0 <= customer_id < n_customers:
        raise ValueError(f"customer index {customer_id} out of range [0, {n_customers})")
    return customer_id


def _integral_customer_value(customer: object) -> int | None:
    if isinstance(customer, bool) or not isinstance(customer, Integral):
        return None
    return int(customer)


def route_loads(instance: CVRPInstance, routes: list[list[int]]) -> list[float]:
    """Return total demand served by each route."""
    customer_nodes = instance.customer_node_indices()
    loads: list[float] = []
    for route in routes:
        route_load = 0.0
        for customer in route:
            customer_id = _normalize_customer_index(customer, instance.n_customers)
            route_load += float(instance.demands[customer_nodes[customer_id]])
        loads.append(route_load)
    return loads


def route_cost(instance: CVRPInstance, routes: list[list[int]]) -> float:
    """Compute total Euclidean CVRP route cost, including depot returns."""
    customer_nodes = instance.customer_node_indices()
    depot_coords = instance.coords[instance.depot_index]
    total_cost = 0.0

    for route in routes:
        previous_coords = depot_coords
        for customer in route:
            customer_id = _normalize_customer_index(customer, instance.n_customers)
            current_coords = instance.coords[customer_nodes[customer_id]]
            total_cost += float(np.linalg.norm(previous_coords - current_coords))
            previous_coords = current_coords
        total_cost += float(np.linalg.norm(previous_coords - depot_coords))

    return total_cost


def validate_routes(
    instance: CVRPInstance,
    routes: list[list[int]],
    *,
    capacity_tolerance: float = 1e-9,
) -> FeasibilityReport:
    """Check route coverage, duplicate visits, customer ids, and capacity constraints."""
    customer_nodes = instance.customer_node_indices()
    visit_counts: dict[int, int] = {}
    out_of_range_customers: list[int] = []
    violations: list[str] = []
    loads: list[float] = []

    for route_index, route in enumerate(routes):
        route_load = 0.0
        for customer in route:
            try:
                customer_id = _normalize_customer_index(customer, instance.n_customers)
            except ValueError as error:
                violations.append(str(error))
                out_of_range_customer = _integral_customer_value(customer)
                if out_of_range_customer is not None:
                    out_of_range_customers.append(out_of_range_customer)
                continue

            visit_counts[customer_id] = visit_counts.get(customer_id, 0) + 1
            route_load += float(instance.demands[customer_nodes[customer_id]])

        loads.append(route_load)
        if route_load > instance.capacity + capacity_tolerance:
            violations.append(
                f"route {route_index} load {route_load} exceeds capacity {instance.capacity}"
            )

    missing_customers = tuple(
        customer for customer in range(instance.n_customers) if customer not in visit_counts
    )
    duplicate_customers = tuple(
        sorted(customer for customer, count in visit_counts.items() if count > 1)
    )

    for customer in missing_customers:
        violations.append(f"customer {customer} missing from routes")
    for customer in duplicate_customers:
        violations.append(f"customer {customer} appears more than once")

    return FeasibilityReport(
        feasible=not violations,
        violations=tuple(violations),
        route_loads=tuple(loads),
        missing_customers=missing_customers,
        duplicate_customers=duplicate_customers,
        out_of_range_customers=tuple(out_of_range_customers),
    )


def validate_labeled_solution(
    instance: CVRPInstance,
    solution: LabeledSolution,
    *,
    cost_tolerance: float = 1e-6,
    capacity_tolerance: float = 1e-9,
) -> FeasibilityReport:
    """Check CVRP route feasibility plus solution metadata consistency."""
    report = validate_routes(instance, solution.routes, capacity_tolerance=capacity_tolerance)
    violations = list(report.violations)

    if solution.num_vehicles != len(solution.routes):
        violations.append(
            f"num_vehicles {solution.num_vehicles} does not match route count "
            f"{len(solution.routes)}"
        )

    if report.feasible:
        computed_cost = route_cost(instance, solution.routes)
        if not math.isclose(
            solution.cost,
            computed_cost,
            rel_tol=cost_tolerance,
            abs_tol=cost_tolerance,
        ):
            violations.append(
                f"solution cost {solution.cost} does not match computed cost {computed_cost}"
            )

    return FeasibilityReport(
        feasible=not violations,
        violations=tuple(violations),
        route_loads=report.route_loads,
        missing_customers=report.missing_customers,
        duplicate_customers=report.duplicate_customers,
        out_of_range_customers=report.out_of_range_customers,
    )
