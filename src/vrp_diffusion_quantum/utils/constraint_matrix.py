"""Route-to-constraint-matrix conversion (task P1.3).

Converts a labeled CVRP solution's routes into the customer-customer route-membership
constraint matrix `M` described in `AGENTS.md` section 7:

    M[i, j] = 1 iff customer i and customer j are in the same vehicle route
    M[i, i] = 0
    M is symmetric

Depot handling: `routes` must contain customer indices only, with the depot omitted, and
`M` is customer-customer only (shape `[n_customers, n_customers]`). Customer indices are
0-based over `[0, n_customers)`, independent of any depot index used elsewhere.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import numpy.typing as npt


def _normalize_customer_index(customer: object, n_customers: int) -> int:
    if isinstance(customer, bool) or not isinstance(customer, Integral):
        raise ValueError(f"customer index {customer!r} must be an integer")
    customer_id = int(customer)
    if not 0 <= customer_id < n_customers:
        raise ValueError(f"customer index {customer_id} out of range [0, {n_customers})")
    return customer_id


def _normalize_n_customers(n_customers: object) -> int:
    if isinstance(n_customers, bool) or not isinstance(n_customers, Integral):
        raise ValueError(f"n_customers must be an integer, got {n_customers!r}")
    return int(n_customers)


def build_constraint_matrix(routes: list[list[int]], n_customers: int) -> npt.NDArray[np.int64]:
    """Build the customer-customer route-membership constraint matrix M.

    Args:
        routes: One list of customer indices per vehicle route. The depot must be omitted.
            Each customer index must appear in at most one route.
        n_customers: Total number of customers. `M` has shape `[n_customers, n_customers]`.

    Returns:
        `m_true`, an `int64` array of shape `[n_customers, n_customers]` with `m_true[i, j] == 1`
        iff `i` and `j` appear in the same route and `i != j`, symmetric, zero diagonal.

    Raises:
        ValueError: if `n_customers` is negative, a customer index is out of range, or a
            customer index appears in more than one route.
    """
    n_customers = _normalize_n_customers(n_customers)
    if n_customers < 0:
        raise ValueError(f"n_customers must be non-negative, got {n_customers}")

    m_true = np.zeros((n_customers, n_customers), dtype=np.int64)
    assigned_customers: set[int] = set()

    for route in routes:
        normalized_route = [_normalize_customer_index(customer, n_customers) for customer in route]
        for customer in normalized_route:
            if customer in assigned_customers:
                raise ValueError(f"customer {customer} appears more than once across routes")
            assigned_customers.add(customer)

        for customer_i in normalized_route:
            for customer_j in normalized_route:
                if customer_i != customer_j:
                    m_true[customer_i, customer_j] = 1

    return m_true
