"""Tests for x9 distance- and label-preserving CVRP augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from vrp_diffusion_quantum.data.augment import (
    AUGMENT_NUM,
    D4_NUM_TRANSFORMS,
    PAPER_DEMAND_STRATEGIES,
    augment_example,
    augment_example_d4,
    augment_example_paper_demand,
    augment_example_paper_geometric,
    augment_example_paper_labeled,
    augment_example_rotation,
    expand_examples,
    transform_coords_d4,
    transform_coords_rotation,
)
from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_labeled_solution


def _example() -> CVRPExample:
    coords = np.array([[0.5, 0.5], [0.2, 0.3], [0.8, 0.1], [0.4, 0.9]], dtype=np.float64)
    demands = np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float64)
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=10.0,
        depot_index=0,
        instance_id="aug_test",
        n_customers=3,
        seed=0,
        generator_settings={},
    )
    routes = [[0, 1], [2]]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=2,
        feasible=True,
        solver_name="test",
        time_budget=None,
        seed=0,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_d4_identity_and_count() -> None:
    xy = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    assert np.allclose(transform_coords_d4(xy, 0), xy)
    assert D4_NUM_TRANSFORMS == 8


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


def test_arbitrary_rotation_preserves_pairwise_distances() -> None:
    ex = _example()
    rotated = augment_example_rotation(ex, 37.0)
    np.testing.assert_allclose(
        _pairwise_distances(rotated.instance.coords),
        _pairwise_distances(ex.instance.coords),
        atol=1e-12,
    )
    assert np.array_equal(rotated.constraint_matrix, ex.constraint_matrix)
    assert rotated.solution.routes == ex.solution.routes


def test_augmentation_x9_preserves_labels_and_feasibility() -> None:
    ex = _example()
    expanded = expand_examples([ex])
    assert len(expanded) == AUGMENT_NUM == 9
    coordinate_views = {view.instance.coords.round(12).tobytes() for view in expanded}
    assert len(coordinate_views) == AUGMENT_NUM
    assert np.allclose(expanded[0].instance.coords, ex.instance.coords)
    assert np.allclose(expanded[0].instance.demands, ex.instance.demands)
    for view in expanded[1:5]:
        assert np.array_equal(view.constraint_matrix, ex.constraint_matrix)
        assert view.solution.routes == ex.solution.routes
        assert not np.allclose(view.instance.coords, ex.instance.coords)
    for view in expanded:
        assert validate_labeled_solution(view.instance, view.solution).feasible
        assert np.array_equal(view.constraint_matrix, ex.constraint_matrix)
        assert view.solution.routes == ex.solution.routes
        assert np.array_equal(view.instance.demands, ex.instance.demands)
        np.testing.assert_allclose(
            _pairwise_distances(view.instance.coords),
            _pairwise_distances(ex.instance.coords),
            atol=1e-12,
        )
    for view in expanded[5:]:
        assert "rotation_degrees" in view.instance.generator_settings
        assert not np.allclose(view.instance.coords, ex.instance.coords)


def test_rotation_transform_validates_inputs() -> None:
    try:
        transform_coords_rotation(np.zeros((2, 3)), 45.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        transform_coords_rotation(np.zeros((2, 2)), float("nan"))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_d4_keeps_routes_m() -> None:
    ex = _example()
    aug = augment_example_d4(ex, 3)
    assert np.array_equal(aug.constraint_matrix, ex.constraint_matrix)
    assert aug.solution.routes == ex.solution.routes


def test_augment_example_variant_bounds() -> None:
    ex = _example()
    assert augment_example(ex, 0) is ex or np.allclose(
        augment_example(ex, 0).instance.coords, ex.instance.coords
    )
    try:
        augment_example(ex, 9)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_paper_geometric_inference_has_exactly_eight_distance_preserving_views() -> None:
    example = _example()
    views = [
        augment_example_paper_geometric(example, variant) for variant in range(D4_NUM_TRANSFORMS)
    ]

    assert len(views) == 8
    assert len({view.instance.coords.round(12).tobytes() for view in views}) == 8
    for view in views:
        np.testing.assert_allclose(
            _pairwise_distances(view.instance.coords),
            _pairwise_distances(example.instance.coords),
            atol=1e-12,
        )
        assert np.array_equal(view.constraint_matrix, example.constraint_matrix)
        assert validate_labeled_solution(view.instance, view.solution).feasible


@pytest.mark.parametrize("strategy", PAPER_DEMAND_STRATEGIES)
def test_paper_demand_augmentation_preserves_route_loads_cost_and_matrix(
    strategy: str,
) -> None:
    example = _example()
    augmented = augment_example_paper_demand(
        example,
        strategy,  # type: ignore[arg-type]
        rng=np.random.default_rng(123),
    )
    customer_nodes = example.instance.customer_node_indices()

    for route in example.solution.routes:
        nodes = [customer_nodes[customer] for customer in route]
        assert augmented.instance.demands[nodes].sum() == pytest.approx(
            example.instance.demands[nodes].sum()
        )
        assert sorted(augmented.instance.demands[nodes]) == pytest.approx(
            sorted(example.instance.demands[nodes])
        )
    assert route_cost(augmented.instance, augmented.solution.routes) == pytest.approx(
        example.solution.cost
    )
    assert np.array_equal(augmented.constraint_matrix, example.constraint_matrix)
    assert validate_labeled_solution(augmented.instance, augmented.solution).feasible


def test_paper_shuffle_is_reproducible_and_composes_with_geometry() -> None:
    example = _example()
    first = augment_example_paper_labeled(
        example,
        geometric_variant=6,
        demand_strategy="shuffle",
        rng=np.random.default_rng(987),
    )
    second = augment_example_paper_labeled(
        example,
        geometric_variant=6,
        demand_strategy="shuffle",
        rng=np.random.default_rng(987),
    )

    np.testing.assert_array_equal(first.instance.demands, second.instance.demands)
    np.testing.assert_allclose(first.instance.coords, second.instance.coords)
    assert first.solution.routes == example.solution.routes
    assert np.array_equal(first.constraint_matrix, example.constraint_matrix)


def test_paper_demand_augmentation_uses_customer_ids_with_nonzero_depot_index() -> None:
    coords = np.array([[0.1, 0.2], [0.8, 0.1], [0.5, 0.5], [0.4, 0.9]], dtype=np.float64)
    demands = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float64)
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=10.0,
        depot_index=2,
        instance_id="nonzero_depot",
        n_customers=3,
        seed=4,
        generator_settings={},
    )
    routes = [[0, 2], [1]]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=2,
        feasible=True,
        solver_name="test",
        time_budget=None,
        seed=4,
        runtime_seconds=0.0,
    )
    example = make_example(instance, solution)

    augmented = augment_example_paper_demand(example, "reverse")

    np.testing.assert_array_equal(augmented.instance.demands, np.array([3.0, 2.0, 0.0, 1.0]))
    assert validate_labeled_solution(augmented.instance, solution).feasible


def test_paper_augmentation_rejects_invalid_variant_and_strategy() -> None:
    example = _example()
    with pytest.raises(ValueError, match="variant"):
        augment_example_paper_geometric(example, 8)
    with pytest.raises(ValueError, match="strategy"):
        augment_example_paper_demand(example, "unknown")  # type: ignore[arg-type]
