"""Integration checks for the reproducible Phase 1 headless pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from scripts.run_data_pipeline import _parse_args, _resolve_run, _summarize

from vrp_diffusion_quantum.data.generate_cvrp import generate_dataset


def _write_configs(tmp_path: Path) -> tuple[Path, Path]:
    generation_config = tmp_path / "generation.yaml"
    generation_config.write_text(
        yaml.safe_dump(
            {
                "config_name": "unit",
                "seed": 11,
                "sizes": [20],
                "num_instances": 2,
                "output_dir": str(tmp_path),
            }
        )
    )
    solve_config = tmp_path / "solve.yaml"
    solve_config.write_text(
        yaml.safe_dump(
            {
                "seed": 19,
                "solver": {
                    "name": "pyvrp",
                    "time_limit": 0.25,
                },
            }
        )
    )
    return generation_config, solve_config


def test_fresh_run_snapshots_complete_solver_settings(tmp_path: Path) -> None:
    generation_config, solve_config = _write_configs(tmp_path)
    args = _parse_args(
        [
            "--gen-config",
            str(generation_config),
            "--solve-config",
            str(solve_config),
            "--data-root",
            str(tmp_path),
            "--run-name",
            "fresh",
            "--workers",
            "2",
        ]
    )

    run_dir, config = _resolve_run(args)

    assert run_dir == tmp_path / "fresh"
    assert config["solver"] == {
        "name": "pyvrp",
        "seed": 19,
        "time_limit": 0.25,
        "no_improvement_seconds": None,
        "fleet_mode": "unlimited",
        "fleet_size": None,
        "workers": 2,
    }


def test_resume_uses_snapshotted_solver_settings_not_new_cli_values(tmp_path: Path) -> None:
    generation_config, solve_config = _write_configs(tmp_path)
    run_dir = tmp_path / "resume"
    run_dir.mkdir()
    saved_config = {
        "seed": 11,
        "sizes": [20],
        "num_instances": 2,
        "solver": {
            "name": "pyvrp",
            "seed": 19,
            "time_limit": 0.25,
            "no_improvement_seconds": None,
            "fleet_mode": "unlimited",
            "fleet_size": None,
            "workers": 1,
        },
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(saved_config))
    args = _parse_args(
        [
            "--gen-config",
            str(generation_config),
            "--solve-config",
            str(solve_config),
            "--data-root",
            str(tmp_path),
            "--run-name",
            "resume",
            "--solver",
            "ortools",
            "--time-limit",
            "9",
            "--workers",
            "4",
        ]
    )

    _, config = _resolve_run(args)

    assert config["solver"] == saved_config["solver"]


def test_summary_contains_data_quality_and_label_speed_statistics() -> None:
    dataset = generate_dataset(20, num_instances=2, seed=3)
    solutions = [
        {
            "cost": 4.0 + index,
            "num_vehicles": 3,
            "runtime_seconds": 0.1 + 0.1 * index,
            "feasible": True,
        }
        for index in range(2)
    ]

    summary = _summarize(20, solutions, dataset)

    assert summary["feasible_rate"] == 1.0
    assert summary["mean_runtime_seconds"] == 0.15
    assert summary["p95_runtime_seconds"] > summary["median_runtime_seconds"]
    assert summary["min_customer_demand"] >= 1
    assert summary["max_customer_demand"] <= 9
    assert summary["mean_capacity"] == 30.0
