"""Tests for the generate/solve -> CVRPExample bridge (`data.export_examples`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from vrp_diffusion_quantum.data.dataset import load_dataset as load_examples
from vrp_diffusion_quantum.data.export_examples import (
    export_run,
    routes_nodes_to_customer_ids,
)
from vrp_diffusion_quantum.data.generate_cvrp import (
    generate_dataset_from_config,
    save_dataset,
)


def _config(seed: int = 42, size: int = 5, num_instances: int = 3) -> dict[str, Any]:
    return {
        "seed": seed,
        "sizes": [size],
        "num_instances": num_instances,
        "depot_mode": "center",
        "customer_mode": "random",
        "demand_mode": "uniform",
        "demand_bounds_by_n": {str(size): {"low": None, "high": None}},
        "random_demand_bounds": False,
        "route_size": None,
        "capacity_by_n": {str(size): 30},
        "capacity_max_by_n": {str(size): None},
        "capacity_weights_by_n": {str(size): None},
        "random_capacity": False,
        "cluster_decay": 0.04,
        "run_name": "unit_run",
    }


def _fake_labels(num_instances: int, size: int) -> list[dict[str, Any]]:
    # One route visiting every customer as *node* indices (depot = node 0 excluded).
    node_route = list(range(1, size + 1))
    return [
        {
            "instance_id": i,
            "n_customers": size,
            "routes": [node_route],
            "cost": 1.0,
            "num_vehicles": 1,
            "feasible": True,
            "solver_name": "unit",
            "runtime_seconds": 0.0,
            "seed": 7,
            "time_budget": None,
        }
        for i in range(num_instances)
    ]


def _make_run(tmp_path: Path, config: dict[str, Any]) -> Path:
    size = config["sizes"][0]
    run_dir = tmp_path / "run"
    dataset = generate_dataset_from_config(config, size)
    save_dataset(dataset, run_dir / f"cvrp{size}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    labels_dir = run_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"cvrp{size}_labels.json").write_text(
        json.dumps(_fake_labels(len(dataset), size))
    )
    return run_dir


def test_routes_nodes_to_customer_ids_depot_zero() -> None:
    routes = [[1, 3], [4, 2, 5]]
    assert routes_nodes_to_customer_ids(routes, depot_index=0, n_customers=5) == [
        [0, 2],
        [3, 1, 4],
    ]


def test_export_run_raw_roundtrips_through_modeling_loader(tmp_path: Path) -> None:
    config = _config()
    size = config["sizes"][0]
    run_dir = _make_run(tmp_path, config)

    written = export_run(run_dir)
    assert set(written) == {size}
    assert len(written[size]) == config["num_instances"]

    examples = load_examples(run_dir / "examples")
    assert len(examples) == config["num_instances"]

    example = examples[0]
    assert example.instance.n_customers == size
    # Node route [1..5] must become 0-based customer ids [0..4].
    assert example.solution.routes == [list(range(size))]
    # All customers in one route -> fully connected off-diagonal constraint matrix.
    expected = np.ones((size, size), dtype=np.int64) - np.eye(size, dtype=np.int64)
    assert np.array_equal(example.constraint_matrix, expected)
    # Raw values: integer demands and the real capacity (30), not normalized.
    assert example.instance.capacity == pytest.approx(30.0)
    customer_demands = np.delete(example.instance.demands, example.instance.depot_index)
    assert np.all(customer_demands >= 1.0)
    assert np.allclose(customer_demands, np.round(customer_demands))


def test_export_run_works_without_config(tmp_path: Path) -> None:
    config = _config()
    size = config["sizes"][0]
    run_dir = _make_run(tmp_path, config)
    (run_dir / "config.yaml").unlink()

    # Export reads raw values straight from the CSVs; no config.yaml is required.
    written = export_run(run_dir)
    assert set(written) == {size}

    examples = load_examples(run_dir / "examples")
    assert len(examples) == config["num_instances"]
    # Raw values survive: integer demands and the real capacity (30).
    assert examples[0].instance.capacity == pytest.approx(30.0)
