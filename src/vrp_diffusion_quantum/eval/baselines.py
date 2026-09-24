"""Matrix-independent route-construction and matrix baselines for the Phase 5 M ablation.

Distinct from any future Phase-6 classical local-search baseline on QUBO neighborhoods: these
operate on a whole instance with no constraint matrix input at all (``no_M``) or a matrix drawn
independently of any model prediction (``random_M``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.eval.routing import order_route_nearest_neighbor_two_opt

__all__ = ["clarke_wright_routes", "random_constraint_matrix"]


def clarke_wright_routes(instance: CVRPInstance) -> list[list[int]]:
    """Savings-guided capacity-constrained route construction; no constraint matrix input.

    Customers begin as singleton routes. At each step, the pair of clusters whose closest-to-depot
    representatives maximize the classical Clarke-Wright saving
    ``s(i, j) = d(depot, i) + d(depot, j) - d(i, j)`` is merged, provided the merged demand still
    fits vehicle capacity; merging stops once no positive-saving pair remains. Each final cluster
    is then sequenced with :func:`order_route_nearest_neighbor_two_opt`. This is a savings-guided
    clustering + nearest-neighbor/2-opt hybrid, not textbook Clarke-Wright endpoint-adjacency
    bookkeeping — it reuses the repo's existing route-ordering routine rather than maintaining a
    second one, at the cost of not tracking route "ends" explicitly.
    """
    n = instance.n_customers
    demands = instance.customer_demands()
    oversized = np.flatnonzero(demands > instance.capacity + 1e-9)
    if oversized.size:
        raise ValueError(f"customer {int(oversized[0])} demand exceeds vehicle capacity")
    if n < 2:
        return [[i] for i in range(n)]

    nodes = instance.customer_node_indices()
    depot_coords = instance.coords[instance.depot_index]
    customer_coords = instance.coords[nodes]
    dist_to_depot = np.linalg.norm(customer_coords - depot_coords, axis=1)
    dist = np.linalg.norm(customer_coords[:, None, :] - customer_coords[None, :, :], axis=2)

    clusters: list[list[int]] = [[i] for i in range(n)]
    loads = [float(demand) for demand in demands]
    while True:
        best: tuple[float, int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                if loads[left] + loads[right] > instance.capacity + 1e-9:
                    continue
                i = min(clusters[left], key=lambda c: dist_to_depot[c])
                j = min(clusters[right], key=lambda c: dist_to_depot[c])
                saving = float(dist_to_depot[i] + dist_to_depot[j] - dist[i, j])
                candidate = (saving, -left, -right)
                if best is None or candidate > (best[0], -best[1], -best[2]):
                    best = (saving, left, right)
        if best is None or best[0] <= 0.0:
            break
        _, left, right = best
        clusters[left] = clusters[left] + clusters[right]
        loads[left] += loads[right]
        del clusters[right]
        del loads[right]

    routes = [order_route_nearest_neighbor_two_opt(instance, cluster) for cluster in clusters]
    routes.sort(key=lambda route: route[0] if route else n)
    return routes


def random_constraint_matrix(
    n_customers: int,
    *,
    seed: int,
    reference_matrix: npt.ArrayLike | None = None,
    density: float | None = None,
) -> npt.NDArray[np.float64]:
    """Seeded random symmetric zero-diagonal binary matrix — the ``random_M`` ablation floor.

    With ``reference_matrix`` given, the number of true positive pairs is matched exactly, so the
    arm isolates whether ``M``'s *structure* helps from whether its base rate alone helps.
    Otherwise, each unordered customer pair is an i.i.d. Bernoulli(``density``) draw
    (``density`` defaults to 0.5). Passing both is an error — pick one mode explicitly.
    """
    if n_customers < 0:
        raise ValueError(f"n_customers must be non-negative, got {n_customers}")
    if reference_matrix is not None and density is not None:
        raise ValueError("pass at most one of reference_matrix, density")

    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_customers, n_customers), dtype=np.float64)
    if n_customers < 2:
        return matrix

    upper = np.triu_indices(n_customers, k=1)
    num_pairs = upper[0].size

    if reference_matrix is not None:
        ref = np.asarray(reference_matrix, dtype=np.float64)
        if ref.shape != (n_customers, n_customers):
            raise ValueError(
                f"reference_matrix shape {ref.shape} does not match ({n_customers}, {n_customers})"
            )
        num_true = round(float(ref[upper].sum()))
        num_true = max(0, min(num_true, num_pairs))
        chosen = rng.choice(num_pairs, size=num_true, replace=False)
    else:
        rate = 0.5 if density is None else float(density)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"density must be in [0, 1], got {rate}")
        chosen = np.flatnonzero(rng.random(num_pairs) < rate)

    rows = upper[0][chosen]
    cols = upper[1][chosen]
    matrix[rows, cols] = 1.0
    matrix[cols, rows] = 1.0
    return matrix
