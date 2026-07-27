"""End-to-end check for held-out Phase 2 training and baseline evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from scripts.train_matrix_predictor import main

ROOT = Path(__file__).resolve().parents[1]


def test_training_script_uses_held_out_split_and_logs_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "held_out_unit",
                "seed": 0,
                "dataset": {
                    "name": "sanity_cvrp",
                    "path": str(ROOT / "data" / "samples" / "sanity_cvrp"),
                },
                "output": {"root": str(output_root)},
                "model": {"hidden_dim": 8},
                "training": {"epochs": 1, "learning_rate": 0.01},
                "split": {"validation_fraction": 0.5, "test_fraction": 0.0},
                "evaluation": {
                    "threshold": 0.5,
                    "num_calibration_bins": 5,
                    "random_baseline_seed": 7,
                    "max_plots": 0,
                },
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["train_matrix_predictor.py", "--config", str(config_path)])

    main()

    run_dirs = list(output_root.glob("held_out_unit_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["num_examples"] == 2
    assert metrics["num_train_examples"] == 1
    assert metrics["num_validation_examples"] == 1
    assert metrics["random_baseline_seed"] == 7
    assert "validation_model_bce" in metrics
    assert "validation_baseline_all_zero_bce" in metrics
    assert "validation_baseline_nearest_neighbor_bce" in metrics
    assert "validation_baseline_demand_aware_bce" in metrics
    assert "validation_baseline_random_bce" in metrics
    assert (run_dir / "model.pt").is_file()
