"""Sanity visualizations for routes and constraint matrices (tasks P1.5, P2.3).

Renders one `CVRPExample`'s routes next to its `constraint_matrix` so the two can be checked
against each other by eye: customers connected by a route line should also be lit up in the
matrix, and vice versa. Exists to catch route/`M` disagreement bugs early, per `AGENTS.md`
Phase 1's done criterion ("manual inspection shows routes and M matrices agree").

`plot_matrix_comparison` renders a predicted `m_prob` next to its `m_true` ground truth, for
Phase 2's qualitative predicted-vs-ground-truth check (task P2.3).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance


def plot_routes(ax: Axes, instance: CVRPInstance, routes: list[list[int]]) -> None:
    """Draw the depot, customers, and each route (depot -> customers -> depot) on `ax`."""
    customer_coords = instance.customer_coords()
    depot_coords = instance.coords[instance.depot_index]

    ax.scatter(
        customer_coords[:, 0],
        customer_coords[:, 1],
        color="lightgray",
        edgecolors="black",
        zorder=2,
        label="customer",
    )
    ax.scatter(
        [depot_coords[0]],
        [depot_coords[1]],
        marker="s",
        s=120,
        color="black",
        zorder=3,
        label="depot",
    )

    color_map = plt.get_cmap("tab10")
    for route_index, route in enumerate(routes):
        route_coords = np.vstack([depot_coords, customer_coords[route], depot_coords])
        ax.plot(
            route_coords[:, 0],
            route_coords[:, 1],
            marker="o",
            color=color_map(route_index % 10),
            zorder=1,
            label=f"route {route_index}",
        )

    ax.set_title("routes")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize="small")


def _plot_matrix_heatmap(
    ax: Axes, matrix: npt.NDArray[np.float64] | npt.NDArray[np.uint8], title: str, cmap: str
) -> None:
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("customer j")
    ax.set_ylabel("customer i")


def plot_constraint_matrix(
    ax: Axes, constraint_matrix: npt.NDArray[np.uint8], title: str = "constraint_matrix"
) -> None:
    """Draw `constraint_matrix` (binary ground truth) as a black/white heatmap on `ax`."""
    _plot_matrix_heatmap(ax, constraint_matrix, title, cmap="Greys")


def plot_matrix_probabilities(
    ax: Axes, m_prob: npt.NDArray[np.float64], title: str = "m_prob"
) -> None:
    """Draw predicted route-membership probabilities `m_prob` as a heatmap on `ax`."""
    _plot_matrix_heatmap(ax, m_prob, title, cmap="Greys")


def plot_example_sanity_check(example: CVRPExample) -> Figure:
    """Build a side-by-side routes + constraint-matrix figure for one `CVRPExample`."""
    fig, (route_ax, matrix_ax) = plt.subplots(1, 2, figsize=(10, 5))
    plot_routes(route_ax, example.instance, example.solution.routes)
    plot_constraint_matrix(matrix_ax, example.constraint_matrix)
    fig.suptitle(f"{example.instance.instance_id} (n_customers={example.instance.n_customers})")
    fig.tight_layout()
    return fig


def plot_matrix_comparison(
    m_true: npt.NDArray[np.uint8], m_prob: npt.NDArray[np.float64], title: str = ""
) -> Figure:
    """Build a 3-panel figure: ground-truth `m_true`, predicted `m_prob`, and |error| (task P2.3).

    Exists to qualitatively check a predicted constraint matrix against ground truth: the error
    panel should be near-white where the model agrees with `m_true` and dark where it doesn't.
    """
    fig, (true_ax, pred_ax, error_ax) = plt.subplots(1, 3, figsize=(15, 5))
    plot_constraint_matrix(true_ax, m_true, title="m_true")
    plot_matrix_probabilities(pred_ax, m_prob, title="m_prob")
    error = np.abs(m_true.astype(np.float64) - m_prob)
    _plot_matrix_heatmap(error_ax, error, title="|m_true - m_prob|", cmap="Reds")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
