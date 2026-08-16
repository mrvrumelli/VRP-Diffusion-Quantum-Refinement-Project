"""Compare frozen reverse-sampling variants on a fixed validation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vrp_diffusion_quantum.data.dataset import load_examples_by_size
from vrp_diffusion_quantum.inference.predict_matrix import (
    evaluate_full_chain_sampling,
    load_denoiser_checkpoint,
    select_examples_by_size,
)
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.utils.experiment import hash_dataset
from vrp_diffusion_quantum.utils.runtime import resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20])
    parser.add_argument("--per-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4350)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def _reference_density(examples: list[Any]) -> float:
    positives = 0.0
    pairs = 0
    for example in examples:
        n = example.instance.n_customers
        upper = np.triu_indices(n, k=1)
        positives += float(example.constraint_matrix[upper].sum())
        pairs += int(upper[0].size)
    return positives / pairs


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    val_dir = args.val_dir.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"validation directory not found: {val_dir}")
    device = resolve_device(args.device)
    pool = load_examples_by_size(val_dir, args.sizes)
    panel = select_examples_by_size(pool, sizes=args.sizes, per_size=args.per_size, seed=args.seed)
    if not panel:
        raise ValueError("fixed validation panel is empty")
    model, payload = load_denoiser_checkpoint(checkpoint, device=device)
    schedule_config = (payload.get("extra") or {}).get("schedule") or {}
    schedule = BernoulliDiffusionSchedule(
        num_timesteps=int(schedule_config.get("num_timesteps", 700)),
        beta_start=float(schedule_config.get("beta_start", 1e-4)),
        beta_end=float(schedule_config.get("beta_end", 2e-2)),
    ).to(device)
    density = _reference_density(panel)
    arms: list[dict[str, Any]] = [
        {
            "name": "exact_stochastic_prior_050",
            "transition_mode": "stochastic",
            "prior_positive_probability": 0.5,
            "step_stride": 1,
        },
        {
            "name": "exact_deterministic_prior_050",
            "transition_mode": "deterministic",
            "prior_positive_probability": 0.5,
            "step_stride": 1,
        },
        {
            "name": "exact_stochastic_realistic_prior",
            "transition_mode": "stochastic",
            "prior_positive_probability": density,
            "step_stride": 1,
        },
        {
            "name": "approx_stride_7_stochastic",
            "transition_mode": "stochastic",
            "prior_positive_probability": 0.5,
            "step_stride": 7,
        },
    ]
    results: list[dict[str, Any]] = []
    for arm in arms:
        started = time.perf_counter()
        metrics = evaluate_full_chain_sampling(
            model,
            schedule,
            panel,
            device=device,
            seed=args.seed,
            threshold=args.threshold,
            adaptive_threshold=False,
            step_stride=int(arm["step_stride"]),
            transition_mode=arm["transition_mode"],
            prior_positive_probability=float(arm["prior_positive_probability"]),
        )
        results.append(
            {**arm, "runtime_seconds": time.perf_counter() - started, "metrics": metrics}
        )
        print(
            f"{arm['name']}: f1={metrics['sample_f1']:.4f} "
            f"route_gap={metrics['route_mean_cost_gap_percent']:.2f}%"
        )

    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "validation_dir": str(val_dir),
        "validation_sha256": hash_dataset(val_dir),
        "panel_instance_ids": [example.instance.instance_id for example in panel],
        "sizes": args.sizes,
        "per_size": args.per_size,
        "seed": args.seed,
        "threshold": args.threshold,
        "device": str(device),
        "num_timesteps": schedule.num_timesteps,
        "reference_pair_positive_density": density,
        "caveat": (
            "stride > 1 uses repeated one-step posteriors at a timestep subsequence and is an "
            "explicitly approximate sampler, not an exact shortened reverse process"
        ),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
