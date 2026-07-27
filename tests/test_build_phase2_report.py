"""Tests for artifact-driven Phase 2 internal report generation."""

from scripts.build_phase2_report import build_report


def test_build_report_includes_required_sections_and_baselines() -> None:
    phase1_summary = {
        "dataset_hash": "abc123",
        "solver": {"name": "pyvrp", "time_limit": 1.0},
        "per_size": [
            {
                "size": 20,
                "n": 100,
                "feasible_rate": 1.0,
                "mean_customer_demand": 5.0,
                "min_customer_demand": 1.0,
                "max_customer_demand": 9.0,
                "mean_total_demand": 100.0,
                "mean_capacity": 30.0,
                "mean_vehicles": 4.0,
                "mean_runtime_seconds": 0.5,
                "median_runtime_seconds": 0.4,
                "p95_runtime_seconds": 0.9,
                "total_runtime_seconds": 50.0,
            }
        ],
    }
    phase2_metrics = {
        "seed": 7,
        "num_train_examples": 80,
        "num_validation_examples": 20,
        "num_test_examples": 0,
    }
    for prefix, bce in (
        ("validation_model", 0.2),
        ("validation_baseline_nearest_neighbor", 0.5),
        ("validation_baseline_demand_aware", 0.4),
        ("validation_baseline_random", 0.7),
    ):
        phase2_metrics[f"{prefix}_bce"] = bce
        phase2_metrics[f"{prefix}_auc"] = 0.8
        phase2_metrics[f"{prefix}_precision"] = 0.6
        phase2_metrics[f"{prefix}_recall"] = 0.7
        phase2_metrics[f"{prefix}_f1"] = 0.65
        phase2_metrics[f"{prefix}_calibration_error"] = 0.1
        phase2_metrics[f"{prefix}_capacity_consistency"] = 0.9

    report = build_report(phase1_summary, phase2_metrics)

    assert "## Data quality" in report
    assert "## Label generation speed" in report
    assert "## Held-out supervised M predictor results" in report
    assert "Nearest-neighbor clusters" in report
    assert "Demand-aware clusters" in report
    assert "Random clusters" in report
    assert "`abc123`" in report
