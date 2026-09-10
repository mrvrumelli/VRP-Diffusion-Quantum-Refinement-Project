"""Tests for the CVRPLIB evaluation script."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluate_cvrplib",
    root / "eval" / "evaluate_cvrplib.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load evaluate_cvrplib.py")
evaluate_cvrplib = importlib.util.module_from_spec(spec)
sys.modules["evaluate_cvrplib"] = evaluate_cvrplib
spec.loader.exec_module(evaluate_cvrplib)

from evaluate_cvrplib import (  # noqa: E402
    DEFAULT_SUBSET_DIR,
    _write_csv,
    discover_records,
    evaluate_record,
    evaluate_subset,
    parse_cvrplib_instance,
    parse_cvrplib_solution,
    select_records,
    subset_hash,
    summarize,
)


def _write_tiny_vrp(path: Path, name: str = "Tiny-n4-k2") -> None:
    path.write_text(
        f"""NAME : {name}
COMMENT : (Tiny CVRPLIB fixture, No of trucks: 2, Optimal value: 8)
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
1 0 0
2 1 0
3 0 1
4 1 1
DEMAND_SECTION
1 0
2 4
3 4
4 4
DEPOT_SECTION
1
-1
EOF
"""
    )


def test_parse_cvrplib_instance_reads_metadata_and_arrays(tmp_path: Path) -> None:
    path = tmp_path / "Tiny-n4-k2.vrp"
    _write_tiny_vrp(path)

    instance = parse_cvrplib_instance(path)

    assert instance.name == "Tiny-n4-k2"
    assert instance.dimension == 4
    assert instance.n_customers == 3
    assert instance.capacity == 10
    assert instance.edge_weight_type == "EUC_2D"
    assert instance.depot_index == 0
    assert instance.vehicles == 2
    assert instance.comment_reference_cost == 8.0
    assert instance.coords.shape == (4, 2)
    assert instance.demands.tolist() == [0.0, 4.0, 4.0, 4.0]
    assert len(instance.sha256) == 64


def test_parse_cvrplib_instance_rejects_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "bad.vrp"
    path.write_text(
        """NAME : bad
TYPE : CVRP
DIMENSION : 2
CAPACITY : 10
NODE_COORD_SECTION
1 0 0
2 1 1
DEPOT_SECTION
1
-1
EOF
"""
    )

    with pytest.raises(ValueError, match="missing DEMAND_SECTION"):
        parse_cvrplib_instance(path)


def test_parse_cvrplib_solution_reads_routes_and_cost(tmp_path: Path) -> None:
    path = tmp_path / "Tiny-n4-k2.sol"
    path.write_text(
        """Route #1: 1 2
Route #2: 3
Cost 8
"""
    )

    solution = parse_cvrplib_solution(path)

    assert solution.routes == [[1, 2], [3]]
    assert solution.cost == 8.0
    assert len(solution.sha256) == 64


def test_discover_records_pairs_same_stem_solution(tmp_path: Path) -> None:
    _write_tiny_vrp(tmp_path / "Tiny-n4-k2.vrp")
    (tmp_path / "Tiny-n4-k2.sol").write_text("Route #1: 1 2 3\nCost: 8\n")

    records = discover_records(tmp_path)

    assert len(records) == 1
    assert records[0].instance.name == "Tiny-n4-k2"
    assert records[0].solution is not None
    assert records[0].reference_cost == 8.0
    assert len(subset_hash(records)) == 64


def test_select_records_samples_reproducibly(tmp_path: Path) -> None:
    for name in ("Tiny-n4-k2", "Tiny-n5-k2", "Tiny-n6-k2"):
        _write_tiny_vrp(tmp_path / f"{name}.vrp", name=name)

    records = discover_records(tmp_path)

    first = select_records(records, sample_size=2, seed=123)
    second = select_records(records, sample_size=2, seed=123)

    assert [record.instance.name for record in first] == [record.instance.name for record in second]


@patch("evaluate_cvrplib.solve_vrplib_problem")
@patch("evaluate_cvrplib.read_vrplib_problem")
def test_evaluate_record_outputs_requested_metrics(
    mock_read: MagicMock,
    mock_solve: MagicMock,
    tmp_path: Path,
) -> None:
    _write_tiny_vrp(tmp_path / "Tiny-n4-k2.vrp")
    (tmp_path / "Tiny-n4-k2.sol").write_text("Route #1: 1 2 3\nCost 8\n")
    record = discover_records(tmp_path)[0]

    result = MagicMock()
    result.cost.return_value = 10
    result.runtime = 0.25
    result.is_feasible.return_value = True
    result.best.num_routes.return_value = 2
    result.num_iterations = 50
    mock_solve.return_value = result

    row = evaluate_record(
        record,
        seed=42,
        max_iterations=50,
        round_func="round",
        selected_subset_hash="abc123",
    )

    assert row["cost"] == 10.0
    assert row["reference_cost"] == 8.0
    assert row["gap"] == pytest.approx(25.0)
    assert row["runtime"] == 0.25
    assert row["feasibility"] is True
    assert row["number_of_vehicles"] == 2
    assert row["iterations"] == 50
    assert row["subset_hash"] == "abc123"
    assert mock_read.call_args.kwargs["round_func"] == "round"
    assert mock_solve.call_args.kwargs["display"] is False


def test_summarize_and_write_csv(tmp_path: Path) -> None:
    rows = [
        {
            "subset": "smoke",
            "subset_hash": "abc",
            "cost": 10.0,
            "gap": 0.0,
            "runtime": 0.2,
            "feasibility": True,
            "number_of_vehicles": 2,
        },
        {
            "subset": "smoke",
            "subset_hash": "abc",
            "cost": 14.0,
            "gap": 10.0,
            "runtime": 0.4,
            "feasibility": False,
            "number_of_vehicles": 3,
        },
    ]

    summary = summarize(rows)
    assert summary == [
        {
            "subset": "smoke",
            "subset_hash": "abc",
            "instances": 2,
            "cost": 12.0,
            "gap": 5.0,
            "runtime": 0.30000000000000004,
            "feasibility": 0.5,
            "number_of_vehicles": 2.5,
        }
    ]

    out_path = tmp_path / "results.csv"
    _write_csv(out_path, rows)
    with out_path.open() as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["subset"] == "smoke"
    assert written[0]["cost"] == "10.0"


def test_default_smoke_subset_evaluates_reproducibly() -> None:
    rows = evaluate_subset(DEFAULT_SUBSET_DIR, seed=42, max_iterations=100)

    assert len(rows) == 1
    row = rows[0]
    assert row["instance"] == "B-n31-k5"
    assert row["cost"] == pytest.approx(672.0)
    assert row["reference_cost"] == pytest.approx(672.0)
    assert row["gap"] == pytest.approx(0.0)
    assert row["feasibility"] is True
    assert row["number_of_vehicles"] == 5
    assert row["round_func"] == "round"
