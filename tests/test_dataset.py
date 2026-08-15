import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import (
    collate_batch,
    load_dataset,
    load_example,
    make_example,
    save_example,
)
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution


def _tiny_instance(instance_id: str = "toy_0", n_customers: int = 3) -> CVRPInstance:
    return CVRPInstance(
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        demands=np.array([0.0, 3.0, 4.0, 2.0]),
        capacity=10.0,
        depot_index=0,
        instance_id=instance_id,
        n_customers=n_customers,
        seed=0,
        generator_settings={"kind": "hand_crafted"},
    )


def _small_instance(instance_id: str = "small") -> CVRPInstance:
    return CVRPInstance(
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        demands=np.array([0.0, 3.0, 4.0]),
        capacity=10.0,
        depot_index=0,
        instance_id=instance_id,
        n_customers=2,
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


def _tiny_example(instance_id: str = "toy_0") -> CVRPExample:
    return make_example(_tiny_instance(instance_id), _tiny_solution())


def test_cvrp_instance_rejects_coords_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="coords shape"):
        CVRPInstance(
            coords=np.zeros((2, 2)),
            demands=np.zeros(4),
            capacity=10.0,
            depot_index=0,
            instance_id="bad",
            n_customers=3,
            seed=0,
            generator_settings={},
        )


def test_cvrp_instance_rejects_demands_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="demands shape"):
        CVRPInstance(
            coords=np.zeros((4, 2)),
            demands=np.zeros(2),
            capacity=10.0,
            depot_index=0,
            instance_id="bad",
            n_customers=3,
            seed=0,
            generator_settings={},
        )


def test_cvrp_instance_rejects_out_of_range_depot() -> None:
    with pytest.raises(ValueError, match="depot_index"):
        CVRPInstance(
            coords=np.zeros((4, 2)),
            demands=np.zeros(4),
            capacity=10.0,
            depot_index=4,
            instance_id="bad",
            n_customers=3,
            seed=0,
            generator_settings={},
        )


def test_cvrp_instance_rejects_negative_n_customers() -> None:
    with pytest.raises(ValueError, match="n_customers"):
        CVRPInstance(
            coords=np.zeros((0, 2)),
            demands=np.zeros(0),
            capacity=10.0,
            depot_index=0,
            instance_id="bad",
            n_customers=-1,
            seed=0,
            generator_settings={},
        )


def test_cvrp_instance_rejects_non_finite_coords() -> None:
    with pytest.raises(ValueError, match="coords"):
        CVRPInstance(
            coords=np.array([[0.0, 0.0], [np.nan, 1.0]]),
            demands=np.array([0.0, 1.0]),
            capacity=10.0,
            depot_index=0,
            instance_id="bad",
            n_customers=1,
            seed=0,
            generator_settings={},
        )


def test_cvrp_instance_rejects_negative_demand() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CVRPInstance(
            coords=np.zeros((2, 2)),
            demands=np.array([0.0, -1.0]),
            capacity=10.0,
            depot_index=0,
            instance_id="bad",
            n_customers=1,
            seed=0,
            generator_settings={},
        )


def test_cvrp_instance_rejects_non_zero_depot_demand() -> None:
    with pytest.raises(ValueError, match="depot demand"):
        CVRPInstance(
            coords=np.zeros((2, 2)),
            demands=np.array([1.0, 2.0]),
            capacity=10.0,
            depot_index=0,
            instance_id="bad",
            n_customers=1,
            seed=0,
            generator_settings={},
        )


def test_cvrp_instance_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        CVRPInstance(
            coords=np.zeros((2, 2)),
            demands=np.array([0.0, 1.0]),
            capacity=0.0,
            depot_index=0,
            instance_id="bad",
            n_customers=1,
            seed=0,
            generator_settings={},
        )


def test_customer_node_indices_excludes_depot_in_order() -> None:
    instance = _tiny_instance()
    assert instance.customer_node_indices() == [1, 2, 3]


def test_customer_node_indices_handles_non_zero_depot() -> None:
    instance = CVRPInstance(
        coords=np.zeros((4, 2)),
        demands=np.zeros(4),
        capacity=10.0,
        depot_index=2,
        instance_id="offset_depot",
        n_customers=3,
        seed=0,
        generator_settings={},
    )
    assert instance.customer_node_indices() == [0, 1, 3]


def test_customer_coords_matches_customer_node_indices() -> None:
    instance = _tiny_instance()
    expected = instance.coords[[1, 2, 3]]
    assert np.array_equal(instance.customer_coords(), expected)


def test_make_example_derives_constraint_matrix_from_routes() -> None:
    example = _tiny_example()
    assert example.constraint_matrix.shape == (3, 3)
    # routes=[[0, 1], [2]] -> customers 0 and 1 share a route, customer 2 is alone.
    assert example.constraint_matrix[0, 1] == 1
    assert example.constraint_matrix[1, 0] == 1
    assert example.constraint_matrix[0, 2] == 0


def test_cvrp_example_rejects_constraint_matrix_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="constraint_matrix shape"):
        CVRPExample(
            instance=_tiny_instance(),
            solution=_tiny_solution(),
            constraint_matrix=np.zeros((2, 2), dtype=np.int64),
        )


def test_cvrp_example_rejects_non_integer_constraint_matrix() -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        CVRPExample(
            instance=_tiny_instance(),
            solution=_tiny_solution(),
            constraint_matrix=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )


def test_cvrp_example_rejects_non_binary_constraint_matrix() -> None:
    with pytest.raises(ValueError, match="binary"):
        CVRPExample(
            instance=_tiny_instance(),
            solution=_tiny_solution(),
            constraint_matrix=np.array([[0, 2, 0], [1, 0, 0], [0, 0, 0]]),
        )


def test_cvrp_example_rejects_non_zero_diagonal() -> None:
    with pytest.raises(ValueError, match="diagonal"):
        CVRPExample(
            instance=_tiny_instance(),
            solution=_tiny_solution(),
            constraint_matrix=np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
        )


def test_cvrp_example_rejects_asymmetric_constraint_matrix() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        CVRPExample(
            instance=_tiny_instance(),
            solution=_tiny_solution(),
            constraint_matrix=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        )


def test_cvrp_example_rejects_constraint_matrix_that_disagrees_with_routes() -> None:
    with pytest.raises(ValueError, match=r"solution\.routes"):
        CVRPExample(
            instance=_tiny_instance(),
            solution=_tiny_solution(),
            constraint_matrix=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        )


def test_save_and_load_example_round_trips(tmp_path: Path) -> None:
    example = _tiny_example()
    example_path = tmp_path / "toy_0.json"

    save_example(example, example_path)
    loaded = load_example(example_path)

    assert np.array_equal(loaded.instance.coords, example.instance.coords)
    assert np.array_equal(loaded.instance.demands, example.instance.demands)
    assert loaded.instance.capacity == example.instance.capacity
    assert loaded.instance.depot_index == example.instance.depot_index
    assert loaded.instance.instance_id == example.instance.instance_id
    assert loaded.instance.n_customers == example.instance.n_customers
    assert loaded.instance.seed == example.instance.seed
    assert loaded.instance.generator_settings == example.instance.generator_settings

    assert loaded.solution.routes == example.solution.routes
    assert loaded.solution.cost == example.solution.cost
    assert loaded.solution.num_vehicles == example.solution.num_vehicles
    assert loaded.solution.feasible == example.solution.feasible
    assert loaded.solution.solver_name == example.solution.solver_name

    assert np.array_equal(loaded.constraint_matrix, example.constraint_matrix)
    assert loaded.constraint_matrix.dtype == np.uint8


def test_load_example_rejects_float_constraint_matrix_in_json(tmp_path: Path) -> None:
    example = _tiny_example()
    example_path = tmp_path / "toy_0.json"
    save_example(example, example_path)
    payload = json.loads(example_path.read_text())
    payload["constraint_matrix"][0][1] = 0.5
    example_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="integer dtype"):
        load_example(example_path)


def test_save_example_creates_parent_directories(tmp_path: Path) -> None:
    example = _tiny_example()
    nested_path = tmp_path / "nested" / "dir" / "toy_0.json"

    save_example(example, nested_path)

    assert nested_path.is_file()


def test_load_dataset_reads_all_examples_sorted(tmp_path: Path) -> None:
    save_example(_tiny_example("toy_1"), tmp_path / "toy_1.json")
    save_example(_tiny_example("toy_0"), tmp_path / "toy_0.json")

    examples = load_dataset(tmp_path)

    assert [example.instance.instance_id for example in examples] == ["toy_0", "toy_1"]


def test_load_dataset_ignores_subset_manifest(tmp_path: Path) -> None:
    save_example(_tiny_example(), tmp_path / "toy_0.json")
    (tmp_path / "subset_manifest.json").write_text(json.dumps({"count": 1}))
    (tmp_path / "training_label_manifest.json").write_text(json.dumps({"count": 1}))

    examples = load_dataset(tmp_path)

    assert [example.instance.instance_id for example in examples] == ["toy_0"]


def test_load_dataset_on_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_dataset(tmp_path) == []


def test_collate_batch_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_batch([])


def test_collate_batch_produces_expected_shapes_for_uniform_size() -> None:
    examples = [_tiny_example("toy_0"), _tiny_example("toy_1")]

    batch = collate_batch(examples)

    assert batch.coords.shape == (2, 4, 2)
    assert batch.demands.shape == (2, 4)
    assert batch.node_mask.shape == (2, 4)
    assert batch.depot_index.shape == (2,)
    assert batch.customer_node_indices.shape == (2, 3)
    assert batch.capacity.shape == (2,)
    assert batch.constraint_matrix.shape == (2, 3, 3)
    assert batch.customer_mask.shape == (2, 3)
    assert batch.cost.shape == (2,)
    assert batch.num_vehicles.shape == (2,)
    assert batch.feasible.shape == (2,)
    assert torch.all(batch.node_mask)
    assert torch.all(batch.customer_mask)
    assert batch.depot_index.tolist() == [0, 0]
    assert batch.customer_node_indices.tolist() == [[1, 2, 3], [1, 2, 3]]
    assert len(batch.metadata) == 2


def test_collate_batch_pads_smaller_examples_and_masks_padding() -> None:
    small = make_example(
        _small_instance("small"),
        LabeledSolution(
            routes=[[0, 1]],
            cost=5.0,
            num_vehicles=1,
            feasible=True,
            solver_name="hand_checked",
            time_budget=None,
            seed=0,
            runtime_seconds=0.001,
        ),
    )
    large = _tiny_example("large")

    batch = collate_batch([small, large])

    assert batch.constraint_matrix.shape == (2, 3, 3)
    assert batch.customer_mask[0].tolist() == [True, True, False]
    assert batch.customer_mask[1].tolist() == [True, True, True]
    # padded row/col of the smaller example's constraint_matrix must stay zero
    assert torch.all(batch.constraint_matrix[0, 2, :] == 0)
    assert torch.all(batch.constraint_matrix[0, :, 2] == 0)
    assert batch.customer_node_indices[0].tolist() == [1, 2, -1]
    assert batch.customer_node_indices[1].tolist() == [1, 2, 3]
    assert batch.metadata[0]["n_customers"] == 2
    assert batch.metadata[1]["n_customers"] == 3


def test_collate_batch_preserves_non_zero_depot_mapping() -> None:
    instance = CVRPInstance(
        coords=np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
        demands=np.array([3.0, 4.0, 0.0]),
        capacity=10.0,
        depot_index=2,
        instance_id="nonzero_depot",
        n_customers=2,
        seed=0,
        generator_settings={"kind": "hand_crafted"},
    )
    solution = LabeledSolution(
        routes=[[0, 1]],
        cost=4.0,
        num_vehicles=1,
        feasible=True,
        solver_name="hand_checked",
        time_budget=None,
        seed=0,
        runtime_seconds=0.001,
    )

    batch = collate_batch([make_example(instance, solution)])

    assert batch.depot_index.tolist() == [2]
    assert batch.customer_node_indices.tolist() == [[0, 1]]
    assert batch.metadata[0]["depot_index"] == 2
    assert batch.metadata[0]["customer_node_indices"] == [0, 1]


def test_size_homogeneous_batches_no_cross_size_padding() -> None:
    from vrp_diffusion_quantum.data.dataset import size_homogeneous_batches

    small_sol = LabeledSolution(
        routes=[[0, 1]],
        cost=5.0,
        num_vehicles=1,
        feasible=True,
        solver_name="hand_checked",
        time_budget=None,
        seed=0,
        runtime_seconds=0.001,
    )
    examples = [
        make_example(_small_instance("b2a"), small_sol),
        _tiny_example("a3a"),
        make_example(_small_instance("b2b"), small_sol),
        _tiny_example("a3b"),
        _tiny_example("a3c"),
    ]
    batches = list(size_homogeneous_batches(examples, batch_size=2, shuffle=False))
    # n=2: 2 examples → 1 batch; n=3: 3 examples → 2 batches
    assert len(batches) == 3
    for batch in batches:
        ns = {int(m["n_customers"]) for m in batch.metadata}
        assert len(ns) == 1
        assert bool(batch.customer_mask.all())

    expanded = list(
        size_homogeneous_batches(examples, batch_size=4, shuffle=False, augmentation=True)
    )
    # 5 examples x 9 augment views = 45; batch_size 4 gives 12 batches.
    assert len(expanded) == 12
    assert sum(b.constraint_matrix.shape[0] for b in expanded) == 45
