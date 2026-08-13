"""Run or resume the unified PyVRP/OR-Tools strong-reference audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from vrp_diffusion_quantum.data.label_audit import (
    AcceptancePolicy,
    AuditPolicy,
    run_reference_label_audit,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "data" / "label_audit_strong_s7799.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workers",
        type=int,
        help="override worker count; each worker runs one CPU-bound solver instance",
    )
    return parser.parse_args()


def _rooted(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _int_float_map(raw: dict[Any, Any]) -> dict[int, float]:
    return {int(key): float(value) for key, value in raw.items()}


def main() -> None:
    args = _parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config: dict[str, Any] = yaml.safe_load(config_path.read_text())
    acceptance_cfg = config.get("acceptance") or {}
    acceptance = AcceptancePolicy(
        near_best_relative_tolerance=float(
            acceptance_cfg.get("near_best_relative_tolerance", 0.001)
        ),
        seed_spread_relative_tolerance=float(
            acceptance_cfg.get("seed_spread_relative_tolerance", 0.005)
        ),
        matrix_disagreement_tolerance=float(
            acceptance_cfg.get("matrix_disagreement_tolerance", 0.05)
        ),
        challenger_improvement_relative_tolerance=float(
            acceptance_cfg.get("challenger_improvement_relative_tolerance", 1e-6)
        ),
        minimum_near_best_seeds=int(acceptance_cfg.get("minimum_near_best_seeds", 2)),
    )
    configured_workers = config.get("workers")
    workers = args.workers if args.workers is not None else configured_workers
    if workers is not None and int(workers) < 1:
        raise ValueError("workers must be >= 1")
    pyvrp_cfg = config["pyvrp"]
    base_seeds = tuple(int(seed) for seed in pyvrp_cfg["base_seeds"])
    if len(base_seeds) != 4:
        raise ValueError("the strong-label audit requires exactly four PyVRP base seeds")
    policy = AuditPolicy(
        pyvrp_base_seeds=base_seeds,
        time_budgets_by_size=_int_float_map(pyvrp_cfg["time_budgets_by_size"]),
        ortools_stable_sample_per_size=int(config["ortools"]["stable_sample_per_size"]),
        workers=None if workers is None else int(workers),
        acceptance=acceptance,
    )
    expected = {int(key): int(value) for key, value in config["expected_counts_by_size"].items()}
    metrics = run_reference_label_audit(
        _rooted(str(config["input_dir"])),
        _rooted(str(config["output_dir"])),
        policy=policy,
        expected_counts_by_size=expected,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"audit output: {_rooted(str(config['output_dir']))}")


if __name__ == "__main__":
    main()
