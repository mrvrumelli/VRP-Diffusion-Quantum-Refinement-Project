"""Deterministic heuristic baselines for route-membership matrix prediction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix


def all_zero_matrix(example: CVRPExample) -> npt.NDArray[np.float64]:
    """Predict that no pair of customers shares a route."""
    n_customers = example.instance.n_customers
    return np.zeros((n_customers, n_customers), dtype=np.float64)


def _estimated_num_clusters(example: CVRPExample) -> int:
    total_demand = float(np.sum(example.instance.customer_demands()))
    return max(1, int(np.ceil(total_demand / example.instance.capacity)))


def _balanced_cluster_sizes(n_customers: int, n_clusters: int) -> list[int]:
    base_size, remainder = divmod(n_customers, n_clusters)
    return [base_size + int(cluster < remainder) for cluster in range(n_clusters)]


def nearest_neighbor_cluster_matrix(example: CVRPExample) -> npt.NDArray[np.float64]:
    """Build explicit spatial clusters by deterministic nearest-neighbor chaining.

    The estimated cluster count is the total-demand capacity lower bound. Cluster sizes are
    balanced, but individual cluster capacity is intentionally ignored so this remains a purely
    spatial baseline. Each cluster becomes a clique in the returned route-membership matrix.
    """
    n_customers = example.instance.n_customers
    if n_customers <= 1:
        return np.zeros((n_customers, n_customers), dtype=np.float64)

    coords = example.instance.customer_coords()
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    n_clusters = min(_estimated_num_clusters(example), n_customers)
    cluster_sizes = _balanced_cluster_sizes(n_customers, n_clusters)

    unassigned = set(range(n_customers))
    clusters: list[list[int]] = []
    for cluster_size in cluster_sizes:
        first_customer = min(unassigned)
        cluster = [first_customer]
        unassigned.remove(first_customer)
        while len(cluster) < cluster_size:
            current = cluster[-1]
            next_customer = min(
                unassigned,
                key=lambda customer: (float(distances[current, customer]), customer),
            )
            cluster.append(next_customer)
            unassigned.remove(next_customer)
        clusters.append(cluster)

    return build_constraint_matrix(clusters, n_customers).astype(np.float64)


def random_cluster_matrix(example: CVRPExample, *, seed: int) -> npt.NDArray[np.float64]:
    """Randomly partition customers into balanced clusters with an explicit seed."""
    n_customers = example.instance.n_customers
    if n_customers <= 1:
        return np.zeros((n_customers, n_customers), dtype=np.float64)

    n_clusters = min(_estimated_num_clusters(example), n_customers)
    cluster_sizes = _balanced_cluster_sizes(n_customers, n_clusters)
    customer_order = np.random.default_rng(seed).permutation(n_customers).tolist()
    clusters: list[list[int]] = []
    start = 0
    for cluster_size in cluster_sizes:
        end = start + cluster_size
        clusters.append(customer_order[start:end])
        start = end
    return build_constraint_matrix(clusters, n_customers).astype(np.float64)


def capacity_sweep_matrix(example: CVRPExample) -> npt.NDArray[np.float64]:
    """Build capacity-feasible angular sweep routes and convert them to membership M."""
    instance = example.instance
    customer_coords = instance.customer_coords()
    customer_demands = instance.customer_demands()
    if np.any(customer_demands > instance.capacity):
        raise ValueError("customer demand exceeds capacity")

    depot_coords = instance.coords[instance.depot_index]
    offsets = customer_coords - depot_coords
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    ordered_customers = np.argsort(angles, kind="stable").tolist()

    routes: list[list[int]] = []
    route: list[int] = []
    route_load = 0.0
    for customer in ordered_customers:
        demand = float(customer_demands[customer])
        if route and route_load + demand > instance.capacity:
            routes.append(route)
            route = []
            route_load = 0.0
        route.append(customer)
        route_load += demand
    if route:
        routes.append(route)

    return build_constraint_matrix(routes, instance.n_customers).astype(np.float64)


def predict_heuristic_matrices(
    example: CVRPExample, *, random_seed: int
) -> dict[str, npt.NDArray[np.float64]]:
    """Return every supported heuristic prediction for one example."""
    return {
        "all_zero": all_zero_matrix(example),
        "nearest_neighbor": nearest_neighbor_cluster_matrix(example),
        "demand_aware": capacity_sweep_matrix(example),
        "random": random_cluster_matrix(example, seed=random_seed),
    }
