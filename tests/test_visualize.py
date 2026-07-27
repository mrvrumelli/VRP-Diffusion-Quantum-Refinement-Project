import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.eval.visualize import (
    plot_constraint_matrix,
    plot_example_sanity_check,
    plot_matrix_comparison,
    plot_matrix_probabilities,
    plot_routes,
)


def _tiny_instance() -> CVRPInstance:
    return CVRPInstance(
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        demands=np.array([0.0, 3.0, 4.0, 2.0]),
        capacity=10.0,
        depot_index=0,
        instance_id="viz_toy",
        n_customers=3,
        seed=0,
        generator_settings={"kind": "hand_crafted"},
    )


def _tiny_solution() -> LabeledSolution:
    return LabeledSolution(
        routes=[[0, 1], [2]],
        cost=12.5,
        num_vehicles=2,
        feasible=True,
        solver_name="hand_checked",
        time_budget=None,
        seed=0,
        runtime_seconds=0.001,
    )


def test_plot_routes_draws_one_line_per_route() -> None:
    fig, ax = plt.subplots()
    routes = [[0, 1], [2]]

    plot_routes(ax, _tiny_instance(), routes)

    assert len(ax.lines) == len(routes)
    plt.close(fig)


def test_plot_constraint_matrix_renders_image() -> None:
    fig, ax = plt.subplots()
    constraint_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

    plot_constraint_matrix(ax, constraint_matrix)

    assert len(ax.images) == 1
    plt.close(fig)


def test_plot_example_sanity_check_saves_to_file(tmp_path: Path) -> None:
    example = make_example(_tiny_instance(), _tiny_solution())

    fig = plot_example_sanity_check(example)
    output_path = tmp_path / "sanity_check.png"
    fig.savefig(output_path)
    plt.close(fig)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_routes_rejects_out_of_range_customer_id() -> None:
    fig, ax = plt.subplots()

    with pytest.raises(IndexError):
        plot_routes(ax, _tiny_instance(), [[5]])
    plt.close(fig)


def test_plot_matrix_probabilities_renders_image() -> None:
    fig, ax = plt.subplots()
    m_prob = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.2], [0.1, 0.2, 0.0]])

    plot_matrix_probabilities(ax, m_prob)

    assert len(ax.images) == 1
    plt.close(fig)


def test_plot_matrix_comparison_renders_three_panels() -> None:
    m_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    m_prob = np.array([[0.0, 0.9, 0.2], [0.8, 0.0, 0.1], [0.2, 0.1, 0.0]])

    fig = plot_matrix_comparison(m_true, m_prob, title="toy example")

    assert len(fig.axes) == 3
    for ax in fig.axes:
        assert len(ax.images) == 1
    plt.close(fig)


def test_plot_matrix_comparison_error_panel_matches_absolute_difference() -> None:
    m_true = np.array([[0, 1], [1, 0]])
    m_prob = np.array([[0.0, 0.3], [0.7, 0.0]])

    fig = plot_matrix_comparison(m_true, m_prob)

    error_image = fig.axes[2].images[0]
    expected_error = np.abs(m_true.astype(np.float64) - m_prob)
    assert np.allclose(error_image.get_array(), expected_error)
    plt.close(fig)


def test_plot_matrix_comparison_saves_to_file(tmp_path: Path) -> None:
    m_true = np.array([[0, 1], [1, 0]])
    m_prob = np.array([[0.0, 0.8], [0.7, 0.0]])

    fig = plot_matrix_comparison(m_true, m_prob)
    output_path = tmp_path / "comparison.png"
    fig.savefig(output_path)
    plt.close(fig)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
