"""Structured CVRP data objects (task P1.4).

Field sets follow `AGENTS.md` section 7: `CVRPInstance` for the problem instance,
`LabeledSolution` for a solved route set, and `CVRPExample` bundling both with the
route-membership constraint matrix from
`vrp_diffusion_quantum.utils.constraint_matrix.build_constraint_matrix` (task P1.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix


@dataclass
class CVRPInstance:
    """One CVRP instance. `coords`/`demands` include the depot node.

    `coords[depot_index]` and `demands[depot_index]` are the depot's coordinates and demand
    (conventionally 0). Node array indices therefore run over the depot plus every customer,
    `0 <= i < n_customers + 1`.

    Elsewhere in the pipeline (`LabeledSolution.routes`, `CVRPExample.constraint_matrix`),
    customers are addressed by a separate 0-based *customer id* in `[0, n_customers)` that
    excludes the depot. With the common `depot_index == 0`, customer id `c` corresponds to
    node array index `c + 1`.
    """

    coords: npt.NDArray[np.float64]  # [n_customers + 1, 2]
    demands: npt.NDArray[np.float64]  # [n_customers + 1]
    capacity: float
    depot_index: int
    instance_id: str
    n_customers: int
    seed: int | None
    generator_settings: dict[str, Any]

    def __post_init__(self) -> None:
        if self.n_customers < 0:
            raise ValueError(f"n_customers must be non-negative, got {self.n_customers}")
        n_nodes = self.n_customers + 1
        if self.coords.shape != (n_nodes, 2):
            raise ValueError(
                f"coords shape {self.coords.shape} does not match (n_customers + 1, 2) = "
                f"({n_nodes}, 2)"
            )
        if self.demands.shape != (n_nodes,):
            raise ValueError(
                f"demands shape {self.demands.shape} does not match (n_customers + 1,) = "
                f"({n_nodes},)"
            )
        if not 0 <= self.depot_index < n_nodes:
            raise ValueError(f"depot_index {self.depot_index} out of range [0, {n_nodes})")
        if not np.isfinite(self.coords).all():
            raise ValueError("coords must contain only finite values")
        if not np.isfinite(self.demands).all():
            raise ValueError("demands must contain only finite values")
        if not np.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError(f"capacity must be positive and finite, got {self.capacity}")
        if np.any(self.demands < 0):
            raise ValueError("demands must be non-negative")
        if not np.isclose(self.demands[self.depot_index], 0.0):
            raise ValueError("depot demand must be zero")

    def customer_node_indices(self) -> list[int]:
        """Node array indices for customers `0..n_customers - 1`, in customer-id order.

        Node indices are every index except `depot_index`, ascending, so customer id `c` is at
        node index `customer_node_indices()[c]`.
        """
        return [
            node_index
            for node_index in range(self.n_customers + 1)
            if node_index != self.depot_index
        ]

    def customer_coords(self) -> npt.NDArray[np.float64]:
        """Coordinates for customers `0..n_customers - 1`, in customer-id order."""
        return self.coords[self.customer_node_indices()]

    def customer_demands(self) -> npt.NDArray[np.float64]:
        """Demands for customers `0..n_customers - 1`, in customer-id order."""
        return self.demands[self.customer_node_indices()]


@dataclass
class LabeledSolution:
    """One solved route set for a `CVRPInstance`, e.g. from PyVRP or OR-Tools (task P1.2)."""

    routes: list[list[int]]  # customer indices, depot omitted inside each route
    cost: float
    num_vehicles: int
    feasible: bool
    solver_name: str
    time_budget: float | None
    seed: int | None
    runtime_seconds: float


@dataclass
class CVRPExample:
    """A `CVRPInstance` paired with a `LabeledSolution` and its `constraint_matrix` (task P1.3)."""

    instance: CVRPInstance
    solution: LabeledSolution
    constraint_matrix: npt.NDArray[np.uint8]  # [n_customers, n_customers], m_true

    def __post_init__(self) -> None:
        n_customers = self.instance.n_customers
        if self.constraint_matrix.shape != (n_customers, n_customers):
            raise ValueError(
                f"constraint_matrix shape {self.constraint_matrix.shape} does not match "
                f"(n_customers, n_customers) = ({n_customers}, {n_customers})"
            )
        if not np.issubdtype(self.constraint_matrix.dtype, np.integer):
            raise ValueError("constraint_matrix must use an integer dtype")
        if not np.all(np.isin(self.constraint_matrix, [0, 1])):
            raise ValueError("constraint_matrix must be binary with values 0 or 1")
        if not np.all(np.diag(self.constraint_matrix) == 0):
            raise ValueError("constraint_matrix diagonal must be zero")
        if not np.array_equal(self.constraint_matrix, self.constraint_matrix.T):
            raise ValueError("constraint_matrix must be symmetric")

        expected = build_constraint_matrix(self.solution.routes, n_customers)
        if not np.array_equal(self.constraint_matrix, expected):
            raise ValueError("constraint_matrix does not match solution.routes")

        self.constraint_matrix = self.constraint_matrix.astype(np.uint8, copy=False)
