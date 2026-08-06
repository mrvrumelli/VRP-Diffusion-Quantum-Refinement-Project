"""Tests for ×9 CVRP augmentation (original + 4 geo + 4 demand)."""

from __future__ import annotations

import numpy as np

from vrp_diffusion_quantum.data.augment import (
    AUGMENT_NUM,
    D4_NUM_TRANSFORMS,
    augment_example,
    augment_example_d4,
    augment_example_demand_shuffle,
    augment_example_node_shuffle,
    expand_examples,
    transform_coords_d4,
)
from vrp_diffusion_quantum.data.types import CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.data.dataset import make_example


def _example():
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
    solution = LabeledSolution(
        routes=[[0, 1], [2]],
        cost=1.0,
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


def test_node_shuffle_preserves_m_structure() -> None:
    ex = _example()
    rng = np.random.default_rng(1)
    sh = augment_example_node_shuffle(ex, rng=rng)
    assert sh.constraint_matrix.shape == ex.constraint_matrix.shape
    assert int(sh.constraint_matrix.sum()) == int(ex.constraint_matrix.sum())
    assert sh.instance.demands[0] == 0.0


def test_demand_shuffle_keeps_m_and_coords() -> None:
    ex = _example()
    rng = np.random.default_rng(2)
    sh = augment_example_demand_shuffle(ex, rng=rng)
    assert np.array_equal(sh.constraint_matrix, ex.constraint_matrix)
    assert np.allclose(sh.instance.coords, ex.instance.coords)
    assert sh.solution.routes == ex.solution.routes
    assert set(sh.instance.customer_demands().tolist()) == set(
        ex.instance.customer_demands().tolist()
    )


def test_augmentation_x9_original_geo_demand() -> None:
    ex = _example()
    expanded = expand_examples([ex])
    assert len(expanded) == AUGMENT_NUM == 9
    assert np.allclose(expanded[0].instance.coords, ex.instance.coords)
    assert np.allclose(expanded[0].instance.demands, ex.instance.demands)
    for view in expanded[1:5]:
        assert np.array_equal(view.constraint_matrix, ex.constraint_matrix)
        assert view.solution.routes == ex.solution.routes
        assert not np.allclose(view.instance.coords, ex.instance.coords)
    for view in expanded[5:]:
        assert np.array_equal(view.constraint_matrix, ex.constraint_matrix)
        assert np.allclose(view.instance.coords, ex.instance.coords)
        assert set(view.instance.customer_demands().tolist()) == set(
            ex.instance.customer_demands().tolist()
        )


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
