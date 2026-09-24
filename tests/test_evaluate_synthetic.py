"""Tests for the synthetic evaluation script."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluate_synthetic", root / "eval" / "evaluate_synthetic.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load evaluate_synthetic.py")
evaluate_synthetic = importlib.util.module_from_spec(spec)
sys.modules["evaluate_synthetic"] = evaluate_synthetic
spec.loader.exec_module(evaluate_synthetic)

from evaluate_synthetic import (  # noqa: E402
    _dataset_path,
    _reference_path,
    _write_csv,
    evaluate_dataset,
    load_reference_costs,
    summarize,
)

from vrp_diffusion_quantum.data.generate_cvrp import CVRPDataset  # noqa: E402


def test_dataset_path_resolution(tmp_path: Path) -> None:
    # Test directory creation
    (tmp_path / "cvrp20").mkdir()
    (tmp_path / "cvrp20_instances.csv").touch()

    path = _dataset_path(tmp_path, 20)
    assert path.name == "cvrp20" or path.name == "cvrp20_instances.csv"


def test_dataset_path_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no CVRP20 dataset found"):
        _dataset_path(tmp_path, 20)


def test_reference_path_resolution(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "cvrp20_labels.json").touch()

    path = _reference_path(tmp_path, 20)
    assert path == tmp_path / "labels" / "cvrp20_labels.json"


def test_reference_path_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no CVRP20 reference labels found"):
        _reference_path(tmp_path, 20)


def test_load_reference_costs(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    labels_file = tmp_path / "labels" / "cvrp20_labels.json"
    labels_file.write_text(json.dumps([{"cost": 100.5}, {"cost": 200.0}]))

    costs = load_reference_costs(tmp_path, 20)
    assert costs == [100.5, 200.0]


def test_load_reference_costs_invalid_format(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    labels_file = tmp_path / "labels" / "cvrp20_labels.json"
    labels_file.write_text(json.dumps({"cost": 100.5}))  # Not a list

    with pytest.raises(ValueError, match="reference labels must be a JSON list"):
        load_reference_costs(tmp_path, 20)


def test_load_reference_costs_missing_fields(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    labels_file = tmp_path / "labels" / "cvrp20_labels.json"
    labels_file.write_text(json.dumps([{"wrong_key": 100.5}]))

    with pytest.raises(ValueError, match="reference labels must contain numeric cost fields"):
        load_reference_costs(tmp_path, 20)


@patch("evaluate_synthetic.solve_instance")
def test_evaluate_dataset(mock_solve: MagicMock) -> None:
    # Setup mock solution
    mock_solution = MagicMock()
    mock_solution.cost = 105.0
    mock_solution.runtime_seconds = 0.5
    mock_solution.feasible = True
    mock_solution.num_vehicles = 5
    mock_solve.return_value = mock_solution

    # Create dummy dataset
    coords = np.random.rand(2, 21, 2)
    demands = np.random.randint(0, 10, size=(2, 21))
    capacity = np.array([30.0, 30.0])

    dataset = CVRPDataset(
        coords=coords,
        demands=demands,
        capacity=capacity,
        n_customers=20,
        seed=42,
        depot_mode="random",
    )

    reference_costs = [100.0, 105.0]

    rows = evaluate_dataset(
        dataset, solver="pyvrp", time_limit=1.0, seed=0, reference_costs=reference_costs
    )

    assert len(rows) == 2
    assert rows[0]["size"] == 20
    assert rows[0]["cost"] == 105.0
    assert rows[0]["gap"] == pytest.approx(5.0)  # 100 * (105 - 100) / 100
    assert rows[0]["runtime"] == 0.5
    assert rows[0]["feasibility"] is True
    assert rows[0]["number_of_vehicles"] == 5

    assert rows[1]["gap"] == 0.0  # 100 * (105 - 105) / 105

    assert mock_solve.call_count == 2


@patch("evaluate_synthetic.solve_instance")
def test_evaluate_dataset_mismatched_references(mock_solve: MagicMock) -> None:
    coords = np.random.rand(2, 21, 2)
    demands = np.random.randint(0, 10, size=(2, 21))
    capacity = np.array([30.0, 30.0])

    dataset = CVRPDataset(
        coords=coords,
        demands=demands,
        capacity=capacity,
        n_customers=20,
        seed=42,
        depot_mode="random",
    )

    with pytest.raises(
        ValueError, match="reference costs \\(1\\) do not match dataset instances \\(2\\)"
    ):
        evaluate_dataset(dataset, reference_costs=[100.0])


def test_summarize() -> None:
    rows = [
        {
            "size": 20,
            "cost": 105.0,
            "gap": 5.0,
            "runtime": 0.5,
            "feasibility": True,
            "number_of_vehicles": 5,
        },
        {
            "size": 20,
            "cost": 100.0,
            "gap": 0.0,
            "runtime": 0.3,
            "feasibility": True,
            "number_of_vehicles": 4,
        },
        {
            "size": 50,
            "cost": 300.0,
            "gap": 2.0,
            "runtime": 1.0,
            "feasibility": False,
            "number_of_vehicles": 10,
        },
    ]

    summary = summarize(rows)
    assert len(summary) == 2

    assert summary[0]["size"] == 20
    assert summary[0]["instances"] == 2
    assert summary[0]["cost"] == 102.5
    assert summary[0]["gap"] == 2.5
    assert summary[0]["runtime"] == 0.4
    assert summary[0]["feasibility"] == 1.0
    assert summary[0]["number_of_vehicles"] == 4.5

    assert summary[1]["size"] == 50
    assert summary[1]["instances"] == 1
    assert summary[1]["cost"] == 300.0
    assert summary[1]["gap"] == 2.0
    assert summary[1]["runtime"] == 1.0
    assert summary[1]["feasibility"] == 0.0
    assert summary[1]["number_of_vehicles"] == 10.0


def test_write_csv(tmp_path: Path) -> None:
    rows = [
        {
            "size": 20,
            "cost": 105.0,
            "gap": 5.0,
            "runtime": 0.5,
            "feasibility": True,
            "number_of_vehicles": 5,
        },
    ]
    out_path = tmp_path / "results.csv"
    _write_csv(out_path, rows)

    assert out_path.is_file()
    with out_path.open() as f:
        reader = csv.DictReader(f)
        data = list(reader)

    assert len(data) == 1
    assert data[0]["size"] == "20"
    assert data[0]["cost"] == "105.0"
    assert data[0]["feasibility"] == "True"


def test_write_csv_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot write an empty evaluation result"):
        _write_csv(tmp_path / "results.csv", [])
