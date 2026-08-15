"""Run the focused diffusion-only sweep after a label policy and GAT are fixed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS: dict[str, dict[str, Any]] = {
    "high": {},
    "uniform": {"t_sample": "uniform"},
    "lr1e4": {"learning_rate": 0.0001},
    "lr5e4": {"learning_rate": 0.0005},
    "pos025": {"pos_weight_power": 0.25},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gat-checkpoint", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4330)
    return parser.parse_args()


def _latest_run(experiment_name: str) -> Path:
    matches = list((ROOT / "outputs/train").glob(f"{experiment_name}_*"))
    if not matches:
        raise FileNotFoundError(f"no run directory found for {experiment_name}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    args = _args()
    checkpoint = (
        args.gat_checkpoint if args.gat_checkpoint.is_absolute() else ROOT / args.gat_checkpoint
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = ROOT / "outputs" / "diffusion_sweep" / stamp
    sweep_dir.mkdir(parents=True)
    template = yaml.safe_load(
        (ROOT / "configs/train/diffusion_denoiser_s7799_policy_v1_smoke_cuda.yaml").read_text()
    )
    results: list[dict[str, Any]] = []
    for variant in args.variants:
        experiment = f"diffusion_selected_{variant}_e{args.epochs}_s{args.seed}"
        config = dict(template)
        config["experiment_name"] = experiment
        config["seed"] = args.seed
        config["dataset"] = {
            "name": "s7799_selected_v2",
            "path": "data/processed/s7799_audit_policy_v2",
        }
        config["validation"] = {"path": "data/processed/s7799_val100_policy_v1"}
        config["model"] = dict(template["model"])
        config["model"]["gat_checkpoint"] = str(checkpoint.relative_to(ROOT))
        config["training"] = dict(template["training"])
        config["training"].update(epochs=args.epochs, max_runtime_seconds=1800)
        config["training"].update(VARIANTS[variant])
        config["sample_eval"] = dict(template["sample_eval"])
        config["sample_eval"].update(every=args.epochs, per_size=2)
        config_path = sweep_dir / f"{variant}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vrp_diffusion_quantum.train.train_diffusion",
                "--config",
                str(config_path),
            ],
            cwd=ROOT,
            check=True,
        )
        run = _latest_run(experiment)
        metrics = json.loads((run / "metrics.json").read_text())
        results.append(
            {
                "variant": variant,
                "overrides": VARIANTS[variant],
                "run": str(run.relative_to(ROOT)),
                "metrics": metrics,
            }
        )
        (sweep_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"sweep_dir={sweep_dir}")
    for result in results:
        metrics = result["metrics"]
        print(
            result["variant"],
            f"auc={metrics['final_val_auc']:.6f}",
            f"bce={metrics['final_val_bce']:.6f}",
            f"f1={metrics['final_val_f1']:.6f}",
        )


if __name__ == "__main__":
    main()
