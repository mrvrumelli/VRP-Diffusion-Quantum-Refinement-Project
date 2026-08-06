"""Ablation table: P2.1 MatrixPredictor vs P3 diffusion (one-shot + full chain).

Fair P2.1 recipe matches the denoiser data side: full ~27k train, ×9 expand
(original + 4 geo + 4 demand), soft √WBCE.

  python scripts/eval_m_ablation.py --config configs/eval/m_predictor_ablation.yaml

Demand ×4 keeps routes/``M`` fixed (invariance aug). Full-chain hard F1 uses sampled
``m_hat`` @ 0.5; soft AUC/BCE use ``m_prob``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as ff
import yaml

from vrp_diffusion_quantum.data.augment import AUGMENT_NUM, expand_examples
from vrp_diffusion_quantum.data.dataset import load_dataset, load_examples_by_size
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    load_denoiser_checkpoint,
    predict_matrix_one_shot,
    sample_constraint_matrix,
    select_examples_by_size,
)
from vrp_diffusion_quantum.metrics.matrix_metrics import (
    MatrixPrediction,
    compute_matrix_metrics,
)
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor
from vrp_diffusion_quantum.utils.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "eval" / "m_predictor_ablation.yaml"
_LOSS_EPS = 1e-7

_TABLE_COLS = (
    "method",
    "f1",
    "auc",
    "precision",
    "recall",
    "bce",
    "threshold",
    "f1_n20",
    "f1_n50",
    "f1_n100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def _soft_wbce(
    m_prob: torch.Tensor,
    m_true: torch.Tensor,
    *,
    weighted: bool,
    pos_weight_power: float,
) -> torch.Tensor:
    """Off-diagonal BCE on probabilities; optional soft √ class weight (same as denoiser)."""
    n = m_prob.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=m_prob.device)
    prob = torch.clamp(m_prob[mask], _LOSS_EPS, 1.0 - _LOSS_EPS)
    target = m_true[mask].float()
    if not weighted:
        return ff.binary_cross_entropy(prob, target)
    pos = target.sum().clamp_min(1.0)
    neg = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = (neg / pos) ** float(pos_weight_power)
    loss = ff.binary_cross_entropy(prob, target, reduction="none")
    weights = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    return (loss * weights).mean()


def _train_matrix_predictor(
    examples: list[CVRPExample],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    augmentation: bool = False,
    weighted_bce: bool = True,
    pos_weight_power: float = 0.5,
) -> MatrixPredictor:
    torch.manual_seed(seed)
    model = MatrixPredictor(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_pool = expand_examples(examples) if augmentation else examples
    n_views = AUGMENT_NUM if augmentation else 1
    print(
        f"P2.1 fair train: n={len(examples)} epochs={epochs} "
        f"augmentation={augmentation} (×{n_views} → {len(train_pool)}) "
        f"weighted_bce={weighted_bce} pos_weight_power={pos_weight_power} device={device}",
        flush=True,
    )
    for epoch in range(epochs):
        order = torch.randperm(
            len(train_pool), generator=torch.Generator().manual_seed(seed + epoch)
        )
        total = 0.0
        n_steps = 0
        for idx in order.tolist():
            view = train_pool[int(idx)]
            coords = torch.from_numpy(view.instance.customer_coords()).float().to(device)
            demands = torch.from_numpy(view.instance.customer_demands()).float().to(device)
            m_true = torch.from_numpy(view.constraint_matrix).float().to(device)
            optimizer.zero_grad()
            m_prob = model(coords, demands, float(view.instance.capacity))
            loss = _soft_wbce(
                m_prob,
                m_true,
                weighted=weighted_bce,
                pos_weight_power=pos_weight_power,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_steps += 1
        print(
            f"P2.1 epoch={epoch} train_loss={total / max(n_steps, 1):.4f} steps={n_steps}",
            flush=True,
        )
    model.eval()
    return model


@torch.no_grad()
def _score_m_probs(
    examples: list[CVRPExample],
    m_probs: list[Any],
    *,
    m_hats: list[Any] | None = None,
    hard_from_hats: bool = False,
) -> dict[str, float]:
    """Score soft probs; optionally take hard F1/P/R from binary ``m_hats`` (full-chain)."""
    soft_preds = [
        MatrixPrediction.from_example(ex, mp) for ex, mp in zip(examples, m_probs, strict=True)
    ]
    soft = compute_matrix_metrics(soft_preds, adaptive_threshold=True)
    used_thr = float(soft.threshold)

    if hard_from_hats:
        if m_hats is None:
            raise ValueError("hard_from_hats requires m_hats")
        hard_preds = [
            MatrixPrediction.from_example(ex, hat)
            for ex, hat in zip(examples, m_hats, strict=True)
        ]
        hard = compute_matrix_metrics(hard_preds, threshold=0.5, adaptive_threshold=False)
        out: dict[str, float] = {
            "bce": float(soft.bce),
            "auc": float(soft.auc) if soft.auc is not None else float("nan"),
            "precision": float(hard.precision),
            "recall": float(hard.recall),
            "f1": float(hard.f1),
            "threshold": 0.5,
            "calibration_error": float(soft.calibration_error),
            "capacity_consistency": float(soft.capacity_consistency),
            "num_examples": float(len(examples)),
        }
        size_preds = hard_preds
        size_thr = 0.5
    else:
        out = {
            "bce": float(soft.bce),
            "auc": float(soft.auc) if soft.auc is not None else float("nan"),
            "precision": float(soft.precision),
            "recall": float(soft.recall),
            "f1": float(soft.f1),
            "threshold": used_thr,
            "calibration_error": float(soft.calibration_error),
            "capacity_consistency": float(soft.capacity_consistency),
            "num_examples": float(len(examples)),
        }
        size_preds = soft_preds
        size_thr = used_thr

    by_size: dict[int, list[MatrixPrediction]] = {}
    for ex, pred in zip(examples, size_preds, strict=True):
        by_size.setdefault(ex.instance.n_customers, []).append(pred)
    for n, preds_n in sorted(by_size.items()):
        out[f"f1_n{n}"] = float(
            compute_matrix_metrics(
                preds_n, threshold=size_thr, adaptive_threshold=False
            ).f1
        )
    return out


@torch.no_grad()
def _eval_matrix_predictor(
    model: MatrixPredictor, examples: list[CVRPExample], device: torch.device
) -> dict[str, float]:
    m_probs = []
    for example in examples:
        coords = torch.from_numpy(example.instance.customer_coords()).float().to(device)
        demands = torch.from_numpy(example.instance.customer_demands()).float().to(device)
        m_prob = model(coords, demands, float(example.instance.capacity)).detach().cpu().numpy()
        m_probs.append(m_prob.astype("float64"))
    return _score_m_probs(examples, m_probs)


@torch.no_grad()
def _eval_diffusion(
    model: torch.nn.Module,
    schedule: BernoulliDiffusionSchedule,
    examples: list[CVRPExample],
    *,
    device: torch.device,
    mode: str,
    seed: int,
    step_stride: int = 1,
) -> dict[str, float]:
    m_probs = []
    m_hats = []
    for i, example in enumerate(examples):
        coords, demands, capacity, _, mask = example_to_model_inputs(example, device=device)
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        if mode == "one_shot":
            result = predict_matrix_one_shot(
                model,
                schedule,
                coords=coords,
                demands=demands,
                capacity=capacity,
                customer_mask=mask,
                generator=gen,
            )
        elif mode == "full_chain":
            result = sample_constraint_matrix(
                model,
                schedule,
                coords=coords,
                demands=demands,
                capacity=capacity,
                customer_mask=mask,
                generator=gen,
                step_stride=step_stride,
            )
        else:
            raise ValueError(f"unknown diffusion mode: {mode}")
        m_probs.append(result.m_prob)
        m_hats.append(result.m_hat)
    return _score_m_probs(
        examples,
        m_probs,
        m_hats=m_hats,
        hard_from_hats=(mode == "full_chain"),
    )


def _fmt_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if value == value else "nan"
    return str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = " | ".join(f"{c:>12}" for c in _TABLE_COLS)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for c in _TABLE_COLS:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:12.4f}" if v == v else f"{'nan':>12}")
            else:
                cells.append(f"{str(v):>12}")
        print(" | ".join(cells))


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], *, note: str) -> None:
    lines = [
        "# Ablation: P2.1 vs P3 (constraint matrix M)",
        "",
        note,
        "",
        "| " + " | ".join(_TABLE_COLS) + " |",
        "| " + " | ".join("---" for _ in _TABLE_COLS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(row.get(c, "")) for c in _TABLE_COLS) + " |")
    p21 = next(r for r in rows if r["method"] == "P2.1_supervised")
    one = next(r for r in rows if r["method"] == "P3_one_shot")
    full = next(r for r in rows if r["method"] == "P3_full_chain")
    lines.extend(
        [
            "",
            "## Deltas vs P2.1",
            "",
            "| method | ΔF1 | ΔAUC |",
            "| --- | --- | --- |",
            f"| P3_one_shot | {one['f1'] - p21['f1']:+.4f} | {one['auc'] - p21['auc']:+.4f} |",
            f"| P3_full_chain | {full['f1'] - p21['f1']:+.4f} | {full['auc'] - p21['auc']:+.4f} |",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(cfg_path.read_text())
    seed = int(config["seed"])
    device = resolve_device()

    train_path = ROOT / config["dataset"]["train_path"]
    val_path = ROOT / config["dataset"]["val_path"]
    sizes = list(config["eval"]["sizes"])
    per_size = int(config["eval"]["per_size"])
    eval_seed = int(config["eval"].get("seed", 0))
    step_stride = int(config.get("diffusion", {}).get("step_stride", 1))

    print(f"device={device}", flush=True)
    print("loading train…", flush=True)
    train_examples = load_dataset(train_path)
    max_train = int(config.get("max_train_examples") or 0)
    if max_train > 0 and len(train_examples) > max_train:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_examples), generator=g)[:max_train]
        train_examples = [train_examples[int(i)] for i in idx]
    print(
        f"train_examples={len(train_examples)} (max_train_examples={max_train or 'all'})",
        flush=True,
    )

    print("loading val pool…", flush=True)
    val_pool = load_examples_by_size(val_path, sizes)
    eval_examples = select_examples_by_size(
        val_pool, sizes=sizes, per_size=per_size, seed=eval_seed
    )
    print(f"eval_examples={len(eval_examples)} sizes={sizes} per_size={per_size}", flush=True)

    mp_cfg = config["matrix_predictor"]
    print("training P2.1 MatrixPredictor (×9 + soft √WBCE)…", flush=True)
    predictor = _train_matrix_predictor(
        train_examples,
        hidden_dim=int(mp_cfg["hidden_dim"]),
        epochs=int(mp_cfg["epochs"]),
        learning_rate=float(mp_cfg["learning_rate"]),
        device=device,
        seed=seed,
        augmentation=bool(mp_cfg.get("augmentation", False)),
        weighted_bce=bool(mp_cfg.get("weighted_bce", True)),
        pos_weight_power=float(mp_cfg.get("pos_weight_power", 0.5)),
    )

    print("scoring P2.1…", flush=True)
    p21 = _eval_matrix_predictor(predictor, eval_examples, device)
    p21_row = {"method": "P2.1_supervised", **p21}

    ckpt = ROOT / config["diffusion"]["checkpoint"]
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"diffusion checkpoint not found: {ckpt}\n"
            "Train via bash scripts/run_gat_then_diffusion.sh, then set diffusion.checkpoint."
        )
    print(f"loading diffusion checkpoint {ckpt}", flush=True)
    model, payload = load_denoiser_checkpoint(ckpt, device=device)
    schedule_cfg = (payload.get("extra") or {}).get("schedule") or {}
    schedule = BernoulliDiffusionSchedule(
        num_timesteps=int(schedule_cfg.get("num_timesteps", 700)),
        beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
        beta_end=float(schedule_cfg.get("beta_end", 2e-2)),
    ).to(device)

    print("scoring P3 one-shot…", flush=True)
    one_shot = _eval_diffusion(
        model, schedule, eval_examples, device=device, mode="one_shot", seed=eval_seed
    )
    one_shot_row = {"method": "P3_one_shot", **one_shot}

    print(f"scoring P3 full-chain T→0 (step_stride={step_stride})…", flush=True)
    full = _eval_diffusion(
        model,
        schedule,
        eval_examples,
        device=device,
        mode="full_chain",
        seed=eval_seed,
        step_stride=step_stride,
    )
    full_row = {"method": "P3_full_chain", **full}

    rows = [p21_row, one_shot_row, full_row]
    print("\n=== Ablation table: P2.1 vs P3 ===\n")
    _print_table(rows)
    print(
        f"\nP3 one-shot vs P2.1 ΔF1={one_shot_row['f1'] - p21_row['f1']:+.4f}; "
        f"P3 full-chain vs P2.1 ΔF1={full_row['f1'] - p21_row['f1']:+.4f}",
        flush=True,
    )

    out_root = ROOT / config["output"]["root"] / config["experiment_name"]
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "ablation_metrics.json"
    csv_path = out_root / "ablation_table.csv"
    md_path = out_root / "ablation_table.md"
    note = (
        f"P2.1 trained on **{len(train_examples)}** examples, "
        f"×9 augmentation={mp_cfg.get('augmentation')} "
        f"(demand views keep M/routes fixed), "
        f"soft √WBCE (power={mp_cfg.get('pos_weight_power', 0.5)}), "
        f"epochs={mp_cfg['epochs']}. "
        f"P3 checkpoint: `{config['diffusion']['checkpoint']}` "
        f"(prefer last.pt if full-chain F1 > best.pt sample_f1). "
        f"Eval: {len(eval_examples)} examples "
        f"(per_size={per_size}, sizes={sizes}; full-chain hard F1 from m_hat@0.5)."
    )
    metrics_path.write_text(json.dumps({"config": config, "rows": rows}, indent=2))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown_table(md_path, rows, note=note)

    ckpt_out = out_root / "matrix_predictor.pt"
    torch.save(
        {
            "model": predictor.state_dict(),
            "hidden_dim": int(mp_cfg["hidden_dim"]),
            "max_train_examples": len(train_examples),
            "augmentation": bool(mp_cfg.get("augmentation", False)),
            "weighted_bce": bool(mp_cfg.get("weighted_bce", True)),
            "pos_weight_power": float(mp_cfg.get("pos_weight_power", 0.5)),
        },
        ckpt_out,
    )
    print(f"\nwrote {metrics_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {ckpt_out}")


if __name__ == "__main__":
    main()
