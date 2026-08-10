"""Run the P3.6 size- and demand-distribution generalization evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from vrp_diffusion_quantum.data.dataset import load_examples_by_size
from vrp_diffusion_quantum.eval.generalization import (
    GeneralizationCase,
    evaluate_generalization_cases,
)
from vrp_diffusion_quantum.inference.predict_matrix import (
    load_denoiser_checkpoint,
    select_examples_by_size,
)
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.utils.experiment import ExperimentTracker, hash_dataset
from vrp_diffusion_quantum.utils.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "eval" / "diffusion_generalization.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def _resolve_required_path(value: object, *, field: str) -> Path:
    if not value:
        raise ValueError(f"{field} must be set in the evaluation config")
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load_selected(path: Path, sizes: list[int], per_size: int, seed: int) -> list[Any]:
    pool = load_examples_by_size(path, sizes)
    selected = select_examples_by_size(pool, sizes=sizes, per_size=per_size, seed=seed)
    if not selected:
        raise ValueError(f"no examples for sizes {sizes} under {path}")
    return selected


def _write_markdown(path: Path, rows: list[dict[str, Any]], reference_case: str) -> None:
    columns = (
        "case",
        "distribution",
        "num_examples",
        "f1",
        "auc",
        "f1_drop_from_reference",
        "capacity_consistency",
        "runtime_seconds",
    )
    lines = [
        "# P3.6 diffusion generalization",
        "",
        f"Reference case: `{reference_case}`. Positive F1 drop means worse than the reference.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text())
    seed = int(config["seed"])
    device = resolve_device(config.get("device", "auto"))

    checkpoint = _resolve_required_path(config.get("checkpoint"), field="checkpoint")
    selection_cfg = config["selection"]
    selection_path = _resolve_required_path(selection_cfg.get("path"), field="selection.path")
    selection_sizes = [int(size) for size in selection_cfg["sizes"]]
    selection_examples = _load_selected(
        selection_path,
        selection_sizes,
        int(selection_cfg["per_size"]),
        seed,
    )

    cases: list[GeneralizationCase] = []
    case_paths: dict[str, Path] = {}
    for case_index, case_cfg in enumerate(config["cases"]):
        name = str(case_cfg["name"])
        path = _resolve_required_path(case_cfg.get("path"), field=f"cases[{case_index}].path")
        sizes = [int(size) for size in case_cfg["sizes"]]
        examples = _load_selected(path, sizes, int(case_cfg["per_size"]), seed + case_index)
        cases.append(
            GeneralizationCase(
                name=name,
                distribution=str(case_cfg["distribution"]),
                examples=examples,
            )
        )
        case_paths[name] = path

    model, payload = load_denoiser_checkpoint(checkpoint, device=device)
    schedule_cfg = (payload.get("extra") or {}).get("schedule") or {}
    schedule = BernoulliDiffusionSchedule(
        num_timesteps=int(schedule_cfg.get("num_timesteps", 700)),
        beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
        beta_end=float(schedule_cfg.get("beta_end", 2e-2)),
    ).to(device)

    evaluation = evaluate_generalization_cases(
        model,
        schedule,
        selection_examples,
        cases,
        reference_case=str(config["reference_case"]),
        mode=str(config.get("mode", "full_chain")),  # type: ignore[arg-type]
        device=device,
        seed=seed,
        step_stride=int(config.get("step_stride", 1)),
        failure_limit=int(config.get("failure_limit", 10)),
    )
    provenance = {
        "checkpoint": str(checkpoint),
        "selection_dataset_hash": hash_dataset(selection_path),
        "case_dataset_hashes": {name: hash_dataset(path) for name, path in case_paths.items()},
    }

    with ExperimentTracker(
        output_root=ROOT / config["output"]["root"],
        experiment_name=str(config["experiment_name"]),
        config=config,
        seed=seed,
        dataset_path=selection_path,
    ) as tracker:
        rows = evaluation["rows"]
        for row in rows:
            tracker.log_metric_row(row)
        tracker.log_metrics({**evaluation, "provenance": provenance})

        metrics_path = tracker.run_dir / "generalization_metrics.json"
        metrics_path.write_text(json.dumps({**evaluation, "provenance": provenance}, indent=2))
        table_path = tracker.run_dir / "generalization_table.csv"
        with table_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        _write_markdown(
            tracker.run_dir / "generalization_table.md",
            rows,
            str(evaluation["reference_case"]),
        )
        (tracker.run_dir / "failure_cases.json").write_text(
            json.dumps(evaluation["failure_cases"], indent=2)
        )
        tracker.logger.info("P3.6 evaluation complete: %s", tracker.run_dir)
        print(f"wrote {tracker.run_dir}")


if __name__ == "__main__":
    main()
