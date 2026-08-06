"""Sample / predict constraint matrices with the trained CMD denoiser.

Reverse process: start from a near-maximum-entropy Bernoulli prior at ``t = T - 1``, then for
``t = T-1, ..., 0`` predict ``x̂_0`` with
:class:`~vrp_diffusion_quantum.models.constraint_denoiser.ConstraintDenoiser` and sample
``x_{t-1} ~ q(x_{t-1} | x_t, x̂_0)`` via
:meth:`~vrp_diffusion_quantum.models.diffusion.BernoulliDiffusionSchedule.q_posterior_prob`.

Every matrix returned here is **symmetric with a zero diagonal** (and respects an optional
``customer_mask``). Hard ``m_hat`` is binary; ``m_prob`` is in ``[0, 1]`` for heatmaps.

Also owns full-chain scoring helpers and the sample-eval CLI::

    python -m vrp_diffusion_quantum.inference.predict_matrix \\
      --checkpoint outputs/train/<run>/checkpoints/best.pt --per-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.metrics.matrix_metrics import (
    MatrixPrediction,
    compute_matrix_metrics,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule

logger = logging.getLogger(__name__)

__all__ = [
    "MatrixPredictionResult",
    "evaluate_full_chain_sampling",
    "example_to_model_inputs",
    "load_denoiser_checkpoint",
    "predict_matrix_one_shot",
    "sample_constraint_matrix",
    "select_examples_by_size",
    "symmetrize_zero_diagonal",
]

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_VAL = _ROOT / (
    "data/raw/cvrp/"
    "20260801_052847_s42_n20-50-100_x10000_dep-random_cust-random_dem-uniform_"
    "d20-1-9-50-1-9-100-1-9_cap20-30-50-40-100-50/splits/val"
)


@dataclass(frozen=True)
class MatrixPredictionResult:
    """Hard matrix, soft probabilities, and optional reverse-process snapshots."""

    m_hat: np.ndarray  # [n, n] float64 in {0, 1}, symmetric, zero diagonal
    m_prob: np.ndarray  # [n, n] float64 in [0, 1], symmetric, zero diagonal
    trajectory: list[np.ndarray] | None = None  # optional list of hard matrices along the chain
    trajectory_timesteps: list[int] | None = None


def symmetrize_zero_diagonal(
    matrix: Tensor,
    customer_mask: Tensor | None = None,
) -> Tensor:
    """Mirror the strict upper triangle, zero the diagonal, apply optional customer mask."""
    return BernoulliDiffusionSchedule._symmetrize(matrix, customer_mask)


def example_to_model_inputs(
    example: CVRPExample,
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pack one :class:`CVRPExample` into denoiser tensors (batch size 1).

    Returns:
        ``coords, demands, capacity, m_true, customer_mask`` with batch dim ``1``.
    """
    n = example.instance.n_customers
    coords = torch.as_tensor(example.instance.customer_coords(), dtype=torch.float32).unsqueeze(0)
    demands = torch.as_tensor(example.instance.customer_demands(), dtype=torch.float32).unsqueeze(
        0
    )
    capacity = torch.as_tensor([example.instance.capacity], dtype=torch.float32)
    m_true = torch.as_tensor(example.constraint_matrix, dtype=torch.float32).unsqueeze(0)
    customer_mask = torch.ones(1, n, dtype=torch.bool)
    if device is not None:
        coords = coords.to(device)
        demands = demands.to(device)
        capacity = capacity.to(device)
        m_true = m_true.to(device)
        customer_mask = customer_mask.to(device)
    return coords, demands, capacity, m_true, customer_mask


def load_denoiser_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    hidden_dim: int = 64,
    num_layers: int = 3,
    time_embed_dim: int = 64,
) -> tuple[ConstraintDenoiser, dict[str, Any]]:
    """Load ``best.pt`` / ``last.pt`` written by diffusion training; return ``(model, payload)``."""
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra = payload.get("extra") or {}
    model_cfg = extra.get("model") or {}
    model = ConstraintDenoiser(
        hidden_dim=int(model_cfg.get("hidden_dim", hidden_dim)),
        num_layers=int(model_cfg.get("num_layers", num_layers)),
        time_embed_dim=int(model_cfg.get("time_embed_dim", time_embed_dim)),
        node_encoder_type=str(model_cfg.get("node_encoder_type", "linear")),  # type: ignore[arg-type]
        gat_num_layers=int(model_cfg.get("gat_num_layers", 5)),
        gat_num_heads=int(model_cfg.get("gat_num_heads", 4)),
        gat_dropout=float(model_cfg.get("gat_dropout", 0.0)),
        freeze_node_encoder=bool(model_cfg.get("freeze_node_encoder", False)),
    )
    from vrp_diffusion_quantum.models.gat_encoder import compat_layernorm_state_dict

    model.load_state_dict(compat_layernorm_state_dict(payload["model"]))
    model.eval()
    if device is not None:
        model.to(device)
    return model, payload


def _rand_tensor(
    *shape: int,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """``torch.rand`` that tolerates a CPU generator with CUDA ``device``."""
    if generator is not None and torch.device(generator.device).type != device.type:
        return torch.rand(
            *shape, generator=generator, device=generator.device, dtype=torch.float32
        ).to(device=device, dtype=dtype)
    return torch.rand(*shape, generator=generator, device=device, dtype=dtype)


def _sample_prior(
    batch_size: int,
    n_customers: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    customer_mask: Tensor | None,
    generator: torch.Generator | None,
) -> Tensor:
    """Maximum-entropy prior: independent Bern(0.5) on the upper triangle, then symmetrize."""
    noise = _rand_tensor(
        batch_size,
        n_customers,
        n_customers,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    hard = (noise < 0.5).to(dtype=dtype)
    return symmetrize_zero_diagonal(hard, customer_mask)


@torch.no_grad()
def predict_matrix_one_shot(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    *,
    coords: Tensor,
    demands: Tensor,
    capacity: Tensor,
    customer_mask: Tensor | None = None,
    t: int | None = None,
    m_t: Tensor | None = None,
    generator: torch.Generator | None = None,
    threshold: float = 0.5,
) -> MatrixPredictionResult:
    """Single forward: ``m_prob`` / ``m_hat`` from noisy ``m_t`` (default prior at ``T-1``)."""
    batch_size, n_customers, _ = coords.shape
    device = coords.device
    if t is None:
        t = schedule.num_timesteps - 1
    t_tensor = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
    if m_t is None:
        m_t = _sample_prior(
            batch_size,
            n_customers,
            device=device,
            dtype=coords.dtype,
            customer_mask=customer_mask,
            generator=generator,
        )
    m_prob = model.predict_proba(
        coords, demands, capacity, m_t, t_tensor, customer_mask=customer_mask
    )
    m_prob = symmetrize_zero_diagonal(m_prob, customer_mask)
    m_hat = symmetrize_zero_diagonal((m_prob >= threshold).to(dtype=m_prob.dtype), customer_mask)
    return MatrixPredictionResult(
        m_hat=m_hat[0].detach().cpu().numpy().astype(np.float64),
        m_prob=m_prob[0].detach().cpu().numpy().astype(np.float64),
    )


@torch.no_grad()
def sample_constraint_matrix(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    *,
    coords: Tensor,
    demands: Tensor,
    capacity: Tensor,
    customer_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
    threshold: float = 0.5,
    snapshot_every: int | None = None,
    step_stride: int = 1,
    x0_clamp: float = 1e-3,
) -> MatrixPredictionResult:
    """Full reverse chain ``t = T-1 → 0`` → final ``m_hat`` / ``m_prob`` (+ optional snapshots).

    Args:
        snapshot_every: if set (e.g. ``100``), store hard matrices every this many steps plus
            the endpoints, for heatmaps. ``None`` skips the trajectory (less memory).
        step_stride: visit every ``step_stride``-th timestep (plus ``t=0``). Larger values
            shorten the chain; the step still uses the one-step posterior
            ``q(x_{t-1}|x_t, x_0)``, so stride > 1 is an approximate skip (not an exact
            multi-step posterior).
        x0_clamp: clamp soft ``x̂_0`` into ``[x0_clamp, 1 - x0_clamp]`` before the posterior
            so extreme probs do not destabilize multi-step sampling.
    """
    if step_stride < 1:
        raise ValueError(f"step_stride must be >= 1, got {step_stride}")
    model.eval()
    batch_size, n_customers, _ = coords.shape
    device = coords.device
    m_t = _sample_prior(
        batch_size,
        n_customers,
        device=device,
        dtype=coords.dtype,
        customer_mask=customer_mask,
        generator=generator,
    )

    trajectory: list[np.ndarray] = []
    trajectory_t: list[int] = []
    if snapshot_every is not None:
        trajectory.append(m_t[0].detach().cpu().numpy().astype(np.float64))
        trajectory_t.append(schedule.num_timesteps - 1)

    timesteps = list(range(schedule.num_timesteps - 1, -1, -step_stride))
    if timesteps[-1] != 0:
        timesteps.append(0)

    m0_prob = m_t
    for t_int in timesteps:
        t_tensor = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        m0_prob = model.predict_proba(
            coords, demands, capacity, m_t, t_tensor, customer_mask=customer_mask
        )
        m0_prob = symmetrize_zero_diagonal(m0_prob, customer_mask)
        if x0_clamp > 0:
            m0_prob = m0_prob.clamp(x0_clamp, 1.0 - x0_clamp)

        if t_int == 0:
            # Final hard matrix from soft x0 at a fixed 0.5 cut (metrics may retune separately).
            m_t = symmetrize_zero_diagonal(
                (m0_prob >= threshold).to(dtype=m0_prob.dtype), customer_mask
            )
        else:
            post = schedule.q_posterior_prob(m_t, m0_prob, t_tensor)
            noise = _rand_tensor(
                *post.shape, generator=generator, device=post.device, dtype=post.dtype
            )
            m_t = symmetrize_zero_diagonal((noise < post).to(dtype=post.dtype), customer_mask)

        if snapshot_every is not None and (t_int % snapshot_every == 0 or t_int == 0):
            trajectory.append(m_t[0].detach().cpu().numpy().astype(np.float64))
            trajectory_t.append(t_int)

    m_hat = m_t[0].detach().cpu().numpy().astype(np.float64)
    m_prob = m0_prob[0].detach().cpu().numpy().astype(np.float64)
    m_hat = _numpy_symmetrize_zero_diag(m_hat)
    m_prob = _numpy_symmetrize_zero_diag(m_prob)

    logger.info(
        "sampled constraint matrix n=%d T=%d stride=%d density=%.4f",
        n_customers,
        schedule.num_timesteps,
        step_stride,
        float(m_hat.sum()) / max(n_customers * (n_customers - 1), 1),
    )
    return MatrixPredictionResult(
        m_hat=m_hat,
        m_prob=m_prob,
        trajectory=trajectory if snapshot_every is not None else None,
        trajectory_timesteps=trajectory_t if snapshot_every is not None else None,
    )


def _numpy_symmetrize_zero_diag(matrix: np.ndarray) -> np.ndarray:
    upper = np.triu(matrix, k=1)
    out = upper + upper.T
    np.fill_diagonal(out, 0.0)
    return out


def select_examples_by_size(
    examples: Sequence[CVRPExample],
    *,
    sizes: Sequence[int] = (20, 50, 100),
    per_size: int = 4,
    seed: int = 0,
) -> list[CVRPExample]:
    """Pick up to ``per_size`` examples for each customer count in ``sizes``."""
    if per_size < 1:
        raise ValueError(f"per_size must be >= 1, got {per_size}")
    by_size: dict[int, list[CVRPExample]] = {int(s): [] for s in sizes}
    for example in examples:
        n = example.instance.n_customers
        if n in by_size:
            by_size[n].append(example)
    rng = np.random.default_rng(seed)
    selected: list[CVRPExample] = []
    for size in sizes:
        pool = by_size[int(size)]
        if not pool:
            continue
        count = min(per_size, len(pool))
        idxs = rng.choice(len(pool), size=count, replace=False)
        selected.extend(pool[int(i)] for i in idxs)
    return selected


@torch.no_grad()
def evaluate_full_chain_sampling(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    examples: Sequence[CVRPExample],
    *,
    device: torch.device | str | None = None,
    seed: int = 0,
    threshold: float | None = None,
    adaptive_threshold: bool = True,
    step_stride: int = 1,
) -> dict[str, Any]:
    """Run full ``T→0`` sampling; hard F1 from ``m_hat``, AUC/BCE from soft ``m_prob``."""
    if not examples:
        raise ValueError("cannot evaluate full-chain sampling on an empty list")

    model.eval()
    soft_preds: list[MatrixPrediction] = []
    hard_preds: list[MatrixPrediction] = []
    by_size_hard: dict[int, list[MatrixPrediction]] = {}
    hard_thr = 0.5 if threshold is None else float(threshold)

    for i, example in enumerate(examples):
        coords, demands, capacity, _, mask = example_to_model_inputs(example, device=device)
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        sampled = sample_constraint_matrix(
            model,
            schedule,
            coords=coords,
            demands=demands,
            capacity=capacity,
            customer_mask=mask,
            generator=gen,
            threshold=hard_thr,
            snapshot_every=None,
            step_stride=step_stride,
        )
        soft_preds.append(MatrixPrediction.from_example(example, sampled.m_prob))
        hard_preds.append(MatrixPrediction.from_example(example, sampled.m_hat))
        n = example.instance.n_customers
        by_size_hard.setdefault(n, []).append(hard_preds[-1])

    soft_metrics = compute_matrix_metrics(
        soft_preds, threshold=threshold, adaptive_threshold=adaptive_threshold
    )
    # Generated matrices are already binary — do not re-threshold soft probs for F1.
    hard_metrics = compute_matrix_metrics(
        hard_preds, threshold=0.5, adaptive_threshold=False
    )

    out: dict[str, Any] = {
        "sample_f1": float(hard_metrics.f1),
        "sample_precision": float(hard_metrics.precision),
        "sample_recall": float(hard_metrics.recall),
        "sample_accuracy": float(hard_metrics.accuracy),
        "sample_auc": soft_metrics.auc,
        "sample_bce": float(soft_metrics.bce),
        "sample_num_examples": len(examples),
        "sample_threshold": 0.5,
        "sample_soft_threshold": float(soft_metrics.threshold),
        "sample_step_stride": int(step_stride),
    }
    for n in sorted(by_size_hard):
        metrics_n = compute_matrix_metrics(
            by_size_hard[n], threshold=0.5, adaptive_threshold=False
        )
        out[f"sample_f1_n{n}"] = float(metrics_n.f1)
        out[f"sample_num_examples_n{n}"] = len(by_size_hard[n])
    return out


def _parse_sample_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full reverse-chain sample eval for a trained CMD denoiser."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, default=_DEFAULT_VAL)
    parser.add_argument("--per-size", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-heatmaps", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI: score ``best.pt`` with full ``T→0`` sampling (optional heatmaps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrp_diffusion_quantum.data.dataset import load_examples_by_size
    from vrp_diffusion_quantum.utils.runtime import resolve_device

    args = _parse_sample_eval_args()
    device = resolve_device(args.device)
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else _ROOT / args.checkpoint
    val_dir = args.val_dir if args.val_dir.is_absolute() else _ROOT / args.val_dir
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"val dir not found: {val_dir}")

    print(f"device={device}")
    print(f"checkpoint={ckpt}")
    print(f"val_dir={val_dir}")

    model, payload = load_denoiser_checkpoint(ckpt, device=device)
    schedule_cfg = (payload.get("extra") or {}).get("schedule") or {}
    schedule = BernoulliDiffusionSchedule(
        num_timesteps=int(schedule_cfg.get("num_timesteps", 700)),
        beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
        beta_end=float(schedule_cfg.get("beta_end", 2e-2)),
    ).to(device)

    pool = load_examples_by_size(val_dir, list(args.sizes))
    if not pool:
        raise ValueError(f"no examples for sizes {args.sizes} under {val_dir}")
    examples = select_examples_by_size(
        pool, sizes=args.sizes, per_size=args.per_size, seed=args.seed
    )
    print(
        f"scoring {len(examples)} examples "
        f"(per_size={args.per_size}, sizes={args.sizes})"
    )

    metrics = evaluate_full_chain_sampling(
        model, schedule, examples, device=device, seed=args.seed
    )
    print("=== full T→0 vs m_true ===")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = _ROOT / "outputs" / "eval" / ckpt.parent.parent.name
    out_dir = out_dir if out_dir.is_absolute() else _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "sample_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {metrics_path}")

    meta = {
        "checkpoint": str(ckpt),
        "val_dir": str(val_dir),
        "per_size": args.per_size,
        "sizes": args.sizes,
        "seed": args.seed,
        "device": str(device),
        "ckpt_epoch": payload.get("epoch"),
        "ckpt_best": payload.get("best_metric_value"),
    }
    (out_dir / "eval_config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    if args.save_heatmaps > 0:
        n_save = min(args.save_heatmaps, len(examples))
        for i, example in enumerate(examples[:n_save]):
            coords, demands, capacity, _, mask = example_to_model_inputs(
                example, device=device
            )
            sampled = sample_constraint_matrix(
                model,
                schedule,
                coords=coords,
                demands=demands,
                capacity=capacity,
                customer_mask=mask,
                generator=torch.Generator(device="cpu").manual_seed(args.seed + i),
                snapshot_every=None,
            )
            m_true = example.constraint_matrix.astype(float)
            dens = float(sampled.m_hat.sum()) / max(m_true.size - m_true.shape[0], 1)
            fig, axes = plt.subplots(1, 3, figsize=(8, 2.6))
            axes[0].imshow(sampled.m_hat, vmin=0, vmax=1, cmap="viridis")
            axes[0].set_title("t=0 m_hat")
            axes[1].imshow(sampled.m_prob, vmin=0, vmax=1, cmap="viridis")
            axes[1].set_title("m_prob")
            axes[2].imshow(m_true, vmin=0, vmax=1, cmap="viridis")
            axes[2].set_title("m_true")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(
                f"n={example.instance.n_customers} dens={dens:.3f}",
                y=1.05,
            )
            fig.tight_layout()
            out = out_dir / f"heatmap_n{example.instance.n_customers}_{i}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
