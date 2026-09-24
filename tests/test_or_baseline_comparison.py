"""Tests for the OR baseline comparison report generator."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vrp_diffusion_quantum.data.dataset import make_example, save_example
from vrp_diffusion_quantum.data.generate_cvrp import CVRPInstance as SolverCVRPInstance
from vrp_diffusion_quantum.data.solve_cvrp import CVRPSolution, SolverName
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare_or_baselines",
    root / "eval" / "compare_or_baselines.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load compare_or_baselines.py")
compare_or_baselines = importlib.util.module_from_spec(spec)
sys.modules["compare_or_baselines"] = compare_or_baselines
spec.loader.exec_module(compare_or_baselines)

from compare_or_baselines import (  # noqa: E402
    _write_csv,
    evaluate_baselines,
    load_selected_examples,
    solver_label,
    summarize,
)


def _example(instance_id: str = "cvrp3_0") -> CVRPExample:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.9, 0.9],
        ],
        dtype=np.float64,
    )
    demands = np.array([0.0, 2.0, 2.0, 6.0], dtype=np.float64)
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=6.0,
        depot_index=0,
        instance_id=instance_id,
        n_customers=3,
        seed=1,
        generator_settings={},
    )
    solution = LabeledSolution(
        routes=[[0, 1], [2]],
        cost=10.0,
        num_vehicles=2,
        feasible=True,
        solver_name="strong_reference",
        time_budget=40.0,
        seed=1,
        runtime_seconds=1.0,
    )
    return make_example(instance, solution)


def test_load_selected_examples_is_reproducible_and_size_balanced(tmp_path: Path) -> None:
    for index in range(4):
        save_example(_example(f"cvrp3_{index}"), tmp_path / f"cvrp3_{index}.json")

    first = load_selected_examples(tmp_path, sizes=[3], instances_per_size=2, seed=123)
    second = load_selected_examples(tmp_path, sizes=[3], instances_per_size=2, seed=123)

    assert [item.instance.instance_id for item in first] == [
        item.instance.instance_id for item in second
    ]
    assert len(first) == 2


def test_load_selected_examples_rejects_insufficient_size(tmp_path: Path) -> None:
    save_example(_example("cvrp3_0"), tmp_path / "cvrp3_0.json")

    with pytest.raises(ValueError, match="size 3 needs 2 examples"):
        load_selected_examples(tmp_path, sizes=[3], instances_per_size=2, seed=123)


def test_solver_label_rejects_unknown_solver() -> None:
    with pytest.raises(ValueError, match="unknown solver"):
        solver_label("missing")  # type: ignore[arg-type]


@patch("compare_or_baselines.solve_instance")
def test_evaluate_baselines_computes_quality_and_best_gap(mock_solve: MagicMock) -> None:
    def fake_solve(
        instance: SolverCVRPInstance,
        *,
        solver: SolverName,
        time_limit: float,
        seed: int,
        instance_id: int,
    ) -> CVRPSolution:
        cost = 11.0 if solver == "pyvrp" else 12.0
        runtime = 0.1 if solver == "pyvrp" else 0.2
        return CVRPSolution(
            instance_id=instance_id,
            n_customers=instance.n_customers,
            routes=[[1, 2], [3]],
            cost=cost,
            num_vehicles=2,
            feasible=True,
            solver_name=solver,
            runtime_seconds=runtime,
            time_budget=time_limit,
            seed=seed,
        )

    mock_solve.side_effect = fake_solve

    rows = evaluate_baselines(
        [_example()],
        solvers=("pyvrp", "ortools"),
        time_budgets=(0.5,),
        seed=7,
    )

    assert len(rows) == 2
    pyvrp = next(row for row in rows if row["solver"] == "pyvrp")
    ortools = next(row for row in rows if row["solver"] == "ortools")
    assert pyvrp["gap_to_reference_percent"] == pytest.approx(10.0)
    assert pyvrp["gap_to_best_observed_percent"] == pytest.approx(0.0)
    assert ortools["gap_to_reference_percent"] == pytest.approx(20.0)
    assert ortools["gap_to_best_observed_percent"] == pytest.approx(100.0 / 11.0)
    assert {call.kwargs["time_limit"] for call in mock_solve.call_args_list} == {0.5}


def test_summarize_reports_runtime_and_solution_quality() -> None:
    rows = [
        {
            "baseline": "PyVRP HGS",
            "solver": "pyvrp",
            "time_budget_seconds": 0.5,
            "size": 3,
            "status": "ok",
            "cost": 11.0,
            "gap_to_reference_percent": 10.0,
            "gap_to_best_observed_percent": 0.0,
            "runtime_seconds": 0.1,
            "runtime_budget_ratio": 0.2,
            "feasibility": True,
            "number_of_vehicles": 2,
        },
        {
            "baseline": "PyVRP HGS",
            "solver": "pyvrp",
            "time_budget_seconds": 0.5,
            "size": 3,
            "status": "ok",
            "cost": 12.0,
            "gap_to_reference_percent": 20.0,
            "gap_to_best_observed_percent": 5.0,
            "runtime_seconds": 0.2,
            "runtime_budget_ratio": 0.4,
            "feasibility": False,
            "number_of_vehicles": 3,
        },
    ]

    summary = summarize(rows)
    overall = next(row for row in summary if row["size"] == "all")

    assert overall["instances"] == 2
    assert overall["success_count"] == 2
    assert overall["feasible_rate"] == pytest.approx(0.5)
    assert overall["mean_gap_to_reference_percent"] == pytest.approx(15.0)
    assert overall["mean_runtime_seconds"] == pytest.approx(0.15)
    assert overall["mean_number_of_vehicles"] == pytest.approx(2.5)


def test_write_csv_rejects_empty_and_writes_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot write an empty OR baseline comparison"):
        _write_csv(tmp_path / "empty.csv", [])

    path = tmp_path / "rows.csv"
    _write_csv(path, [{"solver": "pyvrp", "runtime_seconds": 0.1}])

    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"solver": "pyvrp", "runtime_seconds": "0.1"}]
