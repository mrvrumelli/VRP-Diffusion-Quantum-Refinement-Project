"""Build the first internal Phase 2 results report from Phase 1/2 run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "report_m2_results.md"


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_format_value(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _metric(metrics: dict[str, Any], prefix: str, name: str) -> object:
    return metrics.get(f"{prefix}_{name}")


def build_report(phase1_summary: dict[str, Any], phase2_metrics: dict[str, Any]) -> str:
    """Render a report containing data quality, label speed, and held-out M results."""
    per_size = phase1_summary.get("per_size", [])
    data_quality_rows = [
        [
            row.get("size"),
            row.get("n"),
            row.get("feasible_rate"),
            row.get("mean_customer_demand"),
            f"{_format_value(row.get('min_customer_demand'))}-"
            f"{_format_value(row.get('max_customer_demand'))}",
            row.get("mean_total_demand"),
            row.get("mean_capacity"),
            row.get("mean_vehicles"),
        ]
        for row in per_size
    ]
    label_speed_rows = [
        [
            row.get("size"),
            row.get("mean_runtime_seconds"),
            row.get("median_runtime_seconds"),
            row.get("p95_runtime_seconds"),
            row.get("total_runtime_seconds"),
        ]
        for row in per_size
    ]

    methods = [
        ("Supervised M predictor", "validation_model"),
        ("Nearest-neighbor clusters", "validation_baseline_nearest_neighbor"),
        ("Demand-aware clusters", "validation_baseline_demand_aware"),
        ("Random clusters", "validation_baseline_random"),
        ("All-zero control", "validation_baseline_all_zero"),
    ]
    predictor_rows = [
        [
            label,
            _metric(phase2_metrics, prefix, "bce"),
            _metric(phase2_metrics, prefix, "auc"),
            _metric(phase2_metrics, prefix, "precision"),
            _metric(phase2_metrics, prefix, "recall"),
            _metric(phase2_metrics, prefix, "f1"),
            _metric(phase2_metrics, prefix, "calibration_error"),
            _metric(phase2_metrics, prefix, "capacity_consistency"),
        ]
        for label, prefix in methods
        if _metric(phase2_metrics, prefix, "bce") is not None
    ]

    dataset_hash = phase1_summary.get("dataset_hash", phase2_metrics.get("dataset_hash", "—"))
    solver = phase1_summary.get("solver", {})
    return f"""# Phase 2 Internal Results Report

## Objective

Evaluate the supervised customer-customer route-membership matrix predictor on held-out Phase 1
labels before starting discrete diffusion. This report makes no route-solver or quantum-advantage
claim.

## Reproducibility

- Dataset hash: `{dataset_hash}`
- Label solver: `{solver.get("name", "—")}`
- Label time budget per instance: `{_format_value(solver.get("time_limit"))}`
- Training seed: `{_format_value(phase2_metrics.get("seed"))}`
- Random-baseline seed: `{_format_value(phase2_metrics.get("random_baseline_seed"))}`
- Train/validation/test examples: `{_format_value(phase2_metrics.get("num_train_examples"))}` /
  `{_format_value(phase2_metrics.get("num_validation_examples"))}` /
  `{_format_value(phase2_metrics.get("num_test_examples"))}`

## Data quality

{
        _table(
            [
                "Size",
                "Instances",
                "Feasible rate",
                "Mean demand",
                "Demand range",
                "Mean total demand",
                "Mean capacity",
                "Mean vehicles",
            ],
            data_quality_rows,
        )
    }

## Label generation speed

{
        _table(
            ["Size", "Mean seconds", "Median seconds", "P95 seconds", "Total seconds"],
            label_speed_rows,
        )
    }

Runtime is solver wall-clock time recorded per instance. Parallel end-to-end wall time may be
lower than the sum shown above.

## Held-out supervised M predictor results

{
        _table(
            [
                "Method",
                "BCE",
                "AUC",
                "Precision",
                "Recall",
                "F1",
                "Calibration",
                "Capacity consistency",
            ],
            predictor_rows,
        )
    }

All methods are evaluated on the exact same held-out examples and threshold. The random baseline
uses the seed recorded in the Phase 2 configuration.

## Interpretation checklist

- Compare the supervised predictor against every heuristic, not only the all-zero control.
- Inspect predicted-versus-ground-truth heatmaps for at least 20 held-out examples.
- Report class imbalance through the positive-pair count in `metrics.json`.
- Treat capacity consistency as a structural proxy, not route feasibility.
- Record failure cases before deciding whether to proceed to discrete diffusion.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-run",
        type=Path,
        required=True,
        help="Phase 1 run directory containing summary.json",
    )
    parser.add_argument(
        "--phase2-run",
        type=Path,
        required=True,
        help="Phase 2 run directory containing metrics.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase1_summary = json.loads((args.phase1_run / "summary.json").read_text())
    phase2_metrics = json.loads((args.phase2_run / "metrics.json").read_text())
    report = build_report(phase1_summary, phase2_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"wrote Phase 2 internal report to {args.output}")


if __name__ == "__main__":
    main()
