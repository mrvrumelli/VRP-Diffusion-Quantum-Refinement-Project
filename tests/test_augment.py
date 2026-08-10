"""Tests for x9 distance- and label-preserving CVRP augmentation."""

from __future__ import annotations

import numpy as np

from vrp_diffusion_quantum.data.augment import (
    AUGMENT_NUM,
    D4_NUM_TRANSFORMS,
    augment_example,
    augment_example_d4,
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
