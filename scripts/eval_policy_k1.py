"""Re-score a trained dual-pointer policy checkpoint at chosen ``num_starts`` values.

Used to get a true single-start (K=1) gap comparable to the diffusion pipeline's single-decoded-
sample gap, alongside the best-of-K number the training loop already reports. Reuses the model/
prior config saved in the checkpoint's ``extra`` block rather than requiring the original config
file, so it works even if the source config changes later.

Usage::

    python scripts/eval_policy_k1.py \\
      --checkpoint outputs/policy/<run>/checkpoints/best.pt \\
      --val data/processed/s7799_val100_policy_v1_n20 \\
      --num-starts 1 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vrp_diffusion_quantum.data.dataset import IndexedJSONDataset
from vrp_diffusion_quantum.inference.policy_support import denoiser_prior
from vrp_diffusion_quantum.inference.predict_matrix import load_denoiser_checkpoint
from vrp_diffusion_quantum.inference.solve_instance import load_policy_checkpoint
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.train.train_policy import evaluate_policy

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--num-starts", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy, payload = load_policy_checkpoint(args.checkpoint, device=device)
    extra = payload.get("extra") or {}
    prior_cfg = extra.get("prior") or {}
    seed = int(extra.get("seed", 42))

    denoiser_ckpt = _ROOT / str(prior_cfg["checkpoint"])
    denoiser, denoiser_payload = load_denoiser_checkpoint(denoiser_ckpt, device=device)
    schedule_cfg = (denoiser_payload.get("extra") or {}).get("schedule") or {}
    schedule = BernoulliDiffusionSchedule(
        num_timesteps=int(schedule_cfg.get("num_timesteps", 700)),
        beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
        beta_end=float(schedule_cfg.get("beta_end", 2e-2)),
    ).to(device)
    prior = denoiser_prior(
        denoiser,
        schedule,
        num_inference_steps=(
            int(prior_cfg["num_inference_steps"]) if prior_cfg.get("num_inference_steps") else None
        ),
        step_stride=int(prior_cfg.get("step_stride", 1)),
        threshold=float(prior_cfg.get("threshold", 0.5)),
        seed=seed,
        use_probabilities=bool(prior_cfg.get("use_probabilities", True)),
    )

    val_examples = IndexedJSONDataset(_ROOT / args.val, cache_size=0)
    results: dict[str, dict[str, float]] = {}
    for k in args.num_starts:
        torch.manual_seed(seed)
        metrics = evaluate_policy(
            policy,
            val_examples,
            batch_size=args.batch_size,
            num_starts=k,
            device=device,
            prior=prior,
            check_feasibility=True,
        )
        results[str(k)] = metrics
        print(
            f"num_starts={k}: gap_percent={metrics['gap_percent']:.4f} "
            f"feasible_rate={metrics['feasible_rate']:.4f} "
            f"cost={metrics['cost']:.4f} best_cost={metrics['best_cost']:.4f}"
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
