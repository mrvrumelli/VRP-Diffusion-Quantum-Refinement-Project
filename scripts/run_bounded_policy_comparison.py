"""Run a reproducible, bounded label-policy comparison on one CUDA device."""

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
ARMS = {
    "original": Path("data/processed/label_audit_s7799"),
    "canonical": Path("data/processed/s7799_audit_canonical_v1"),
    "mixed": Path("data/processed/s7799_audit_policy_v1"),
    "selected": Path("data/processed/s7799_audit_policy_v2"),
    "selected100": Path("data/processed/s7799_policy_v2_100_per_size"),
    "selected250": Path("data/processed/s7799_policy_v2_250_per_size"),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4330)
    parser.add_argument("--t-sample", choices=("high", "uniform"), default="high")
    return parser.parse_args()


def _latest_run(experiment_name: str) -> Path:
    matches = list((ROOT / "outputs/train").glob(f"{experiment_name}_*"))
    if not matches:
        raise FileNotFoundError(f"no run directory found for {experiment_name}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    args = _args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    comparison_dir = ROOT / "outputs" / "policy_comparison" / stamp
    comparison_dir.mkdir(parents=True)
    gat_template = yaml.safe_load(
        (ROOT / "configs/train/gat_pretrain_s7799_policy_v1_smoke_cuda.yaml").read_text()
    )
    diffusion_template = yaml.safe_load(
        (ROOT / "configs/train/diffusion_denoiser_s7799_policy_v1_smoke_cuda.yaml").read_text()
    )
    results: list[dict[str, Any]] = []

    for arm in args.arms:
        dataset_path = ARMS[arm]
        gat_experiment = f"gat_policy_compare_{arm}_e{args.epochs}_s{args.seed}"
        gat_config = dict(gat_template)
        gat_config["experiment_name"] = gat_experiment
        gat_config["seed"] = args.seed
        gat_config["dataset"] = {"name": f"s7799_{arm}", "path": str(dataset_path)}
        gat_config["validation"] = {"path": "data/processed/s7799_val100_policy_v1"}
        gat_config["training"] = dict(gat_template["training"])
        gat_config["training"].update(epochs=args.epochs, max_runtime_seconds=900)
        gat_path = comparison_dir / f"gat_{arm}.yaml"
        gat_path.write_text(yaml.safe_dump(gat_config, sort_keys=False))
        _run(["scripts/pretrain_gat_encoder.py", "--config", str(gat_path)])
        gat_run = _latest_run(gat_experiment)
        gat_checkpoint = gat_run / "checkpoints/gat_encoder_best.pt"
        if not gat_checkpoint.is_file():
            raise FileNotFoundError(gat_checkpoint)

        diffusion_experiment = (
            f"diffusion_policy_compare_{arm}_{args.t_sample}_e{args.epochs}_s{args.seed}"
        )
        diffusion_config = dict(diffusion_template)
        diffusion_config["experiment_name"] = diffusion_experiment
        diffusion_config["seed"] = args.seed
        diffusion_config["dataset"] = {"name": f"s7799_{arm}", "path": str(dataset_path)}
        diffusion_config["validation"] = {"path": "data/processed/s7799_val100_policy_v1"}
        diffusion_config["model"] = dict(diffusion_template["model"])
        diffusion_config["model"]["gat_checkpoint"] = str(gat_checkpoint.relative_to(ROOT))
        diffusion_config["training"] = dict(diffusion_template["training"])
        diffusion_config["training"].update(
            epochs=args.epochs,
            max_runtime_seconds=1800,
            t_sample=args.t_sample,
        )
        diffusion_config["sample_eval"] = dict(diffusion_template["sample_eval"])
        diffusion_config["sample_eval"].update(every=args.epochs, per_size=2)
        diffusion_path = comparison_dir / f"diffusion_{arm}.yaml"
        diffusion_path.write_text(yaml.safe_dump(diffusion_config, sort_keys=False))
        _run(
            [
                "-m",
                "vrp_diffusion_quantum.train.train_diffusion",
                "--config",
                str(diffusion_path),
            ]
        )
        diffusion_run = _latest_run(diffusion_experiment)
        metrics = json.loads((diffusion_run / "metrics.json").read_text())
        results.append(
            {
                "arm": arm,
                "dataset_path": str(dataset_path),
                "gat_run": str(gat_run.relative_to(ROOT)),
                "diffusion_run": str(diffusion_run.relative_to(ROOT)),
                "metrics": metrics,
            }
        )
        (comparison_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"comparison_dir={comparison_dir}")
    for result in results:
        metrics = result["metrics"]
        print(
            result["arm"],
            f"auc={metrics['final_val_auc']:.6f}",
            f"bce={metrics['final_val_bce']:.6f}",
            f"f1={metrics['final_val_f1']:.6f}",
            f"seconds={metrics['total_runtime_seconds']:.1f}",
        )


if __name__ == "__main__":
    main()
