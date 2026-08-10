"""Ablation table: P2.1 MatrixPredictor vs P3 diffusion (one-shot + full chain).

Fair P2.1 recipe matches the denoiser data side: full training set, x9 label-preserving
geometric augmentation, soft sqrt-WBCE.

  python scripts/eval_m_ablation.py --config configs/eval/m_predictor_ablation.yaml

Full-chain hard F1 uses sampled ``m_hat`` at 0.5; soft AUC/BCE use ``m_prob``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as ff
import yaml

from vrp_diffusion_quantum.data.augment import AUGMENT_NUM, expand_examples
from vrp_diffusion_quantum.data.dataset import load_dataset, load_examples_by_size
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.eval.matrix_ablation import (
    score_matrix_probabilities,
    validate_disjoint_examples,
)
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    load_denoiser_checkpoint,
    predict_matrix_one_shot,
    sample_constraint_matrix,
    select_examples_by_size,
)
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor
from vrp_diffusion_quantum.utils.experiment import ExperimentTracker, hash_dataset
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
    "runtime_seconds",
    "f1_n20",
    "f1_n50",
    "f1_n100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def _resolve_required_path(value: object, *, field: str) -> Path:
    if not value:
        raise ValueError(f"{field} must be set in the evaluation config")
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


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
        f"augmentation={augmentation} (x{n_views} -> {len(train_pool)}) "
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
def _eval_matrix_predictor(
    model: MatrixPredictor,
    examples: list[CVRPExample],
    device: torch.device,
    *,
    threshold: float | None = None,
    adaptive_threshold: bool = True,
) -> dict[str, float]:
    m_probs = []
    for example in examples:
        coords = torch.from_numpy(example.instance.customer_coords()).float().to(device)
        demands = torch.from_numpy(example.instance.customer_demands()).float().to(device)
        m_prob = model(coords, demands, float(example.instance.capacity)).detach().cpu().numpy()
        m_probs.append(m_prob.astype("float64"))
    return score_matrix_probabilities(
        examples,
        m_probs,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
    )


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
    threshold: float | None,
    adaptive_threshold: bool,
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
    return score_matrix_probabilities(
        examples,
        m_probs,
        m_hats=m_hats,
        hard_from_hats=(mode == "full_chain"),
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
    )


def _fmt_cell(value: object) -> str:
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
                cells.append(f"{v!s:>12}")
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
    device = resolve_device(config.get("device", "auto"))

    train_path = _resolve_required_path(
        config["dataset"].get("train_path"), field="dataset.train_path"
    )
    selection_path = _resolve_required_path(
        config["dataset"].get("selection_path"), field="dataset.selection_path"
    )
    test_path = _resolve_required_path(
        config["dataset"].get("test_path"), field="dataset.test_path"
    )
    if selection_path.resolve() == test_path.resolve():
        raise ValueError("dataset.selection_path and dataset.test_path must be different")
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

    print("loading model-selection pool...", flush=True)
    selection_pool = load_examples_by_size(selection_path, sizes)
    selection_examples = select_examples_by_size(
        selection_pool,
        sizes=sizes,
        per_size=int(config["eval"].get("selection_per_size", per_size)),
        seed=eval_seed,
    )
    print("loading untouched test pool...", flush=True)
    test_pool = load_examples_by_size(test_path, sizes)
    test_examples = select_examples_by_size(
        test_pool, sizes=sizes, per_size=per_size, seed=eval_seed
    )
    validate_disjoint_examples(selection_examples, test_examples)
    print(
        f"selection_examples={len(selection_examples)} test_examples={len(test_examples)} "
        f"sizes={sizes} per_size={per_size}",
        flush=True,
    )

    mp_cfg = config["matrix_predictor"]
    print("training P2.1 MatrixPredictor (x9 + soft sqrt-WBCE)...", flush=True)
    started = time.perf_counter()
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
    predictor_train_runtime = time.perf_counter() - started

    print("selecting P2.1 threshold on validation data...", flush=True)
    p21_selection = _eval_matrix_predictor(predictor, selection_examples, device)
    started = time.perf_counter()
    p21 = _eval_matrix_predictor(
        predictor,
        test_examples,
        device,
        threshold=float(p21_selection["threshold"]),
        adaptive_threshold=False,
    )
    p21_row = {
        "method": "P2.1_supervised",
        **p21,
        "runtime_seconds": time.perf_counter() - started,
    }

    ckpt = _resolve_required_path(
        config["diffusion"].get("checkpoint"), field="diffusion.checkpoint"
    )
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

    print("selecting P3 one-shot threshold on validation data...", flush=True)
    one_shot_selection = _eval_diffusion(
        model,
        schedule,
        selection_examples,
        device=device,
        mode="one_shot",
        seed=eval_seed,
        threshold=None,
        adaptive_threshold=True,
    )
    started = time.perf_counter()
    one_shot = _eval_diffusion(
        model,
        schedule,
        test_examples,
        device=device,
        mode="one_shot",
        seed=eval_seed,
        threshold=float(one_shot_selection["threshold"]),
        adaptive_threshold=False,
    )
    one_shot_row = {
        "method": "P3_one_shot",
        **one_shot,
        "runtime_seconds": time.perf_counter() - started,
    }

    print(f"scoring P3 full-chain T→0 (step_stride={step_stride})…", flush=True)
    started = time.perf_counter()
    full = _eval_diffusion(
        model,
        schedule,
        test_examples,
        device=device,
        mode="full_chain",
        seed=eval_seed,
        step_stride=step_stride,
        threshold=0.5,
        adaptive_threshold=False,
    )
    full_row = {
        "method": "P3_full_chain",
        **full,
        "runtime_seconds": time.perf_counter() - started,
    }

    rows = [p21_row, one_shot_row, full_row]
    print("\n=== Ablation table: P2.1 vs P3 ===\n")
    _print_table(rows)
    print(
        f"\nP3 one-shot vs P2.1 ΔF1={one_shot_row['f1'] - p21_row['f1']:+.4f}; "
        f"P3 full-chain vs P2.1 ΔF1={full_row['f1'] - p21_row['f1']:+.4f}",
        flush=True,
    )

    tracker = ExperimentTracker(
        output_root=ROOT / config["output"]["root"],
        experiment_name=config["experiment_name"],
        config=config,
        seed=seed,
        dataset_path=test_path,
    )
    out_root = tracker.run_dir
    metrics_path = out_root / "ablation_metrics.json"
    csv_path = out_root / "ablation_table.csv"
    md_path = out_root / "ablation_table.md"
    note = (
        f"P2.1 trained on **{len(train_examples)}** examples, "
        f"x9 label-preserving augmentation={mp_cfg.get('augmentation')}, "
        f"soft sqrt-WBCE (power={mp_cfg.get('pos_weight_power', 0.5)}), "
        f"epochs={mp_cfg['epochs']}. "
        f"P3 checkpoint: `{config['diffusion']['checkpoint']}` "
        f"(selected only by validation metrics). "
        f"Test: {len(test_examples)} untouched examples "
        f"(per_size={per_size}, sizes={sizes}; full-chain hard F1 from m_hat@0.5)."
    )
    provenance = {
        "train_dataset_hash": hash_dataset(train_path),
        "selection_dataset_hash": hash_dataset(selection_path),
        "test_dataset_hash": hash_dataset(test_path),
        "predictor_train_runtime_seconds": predictor_train_runtime,
        "selection_thresholds": {
            "P2.1_supervised": p21_selection["threshold"],
            "P3_one_shot": one_shot_selection["threshold"],
            "P3_full_chain": 0.5,
        },
    }
    metrics_path.write_text(
        json.dumps({"config": config, "rows": rows, "provenance": provenance}, indent=2)
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown_table(md_path, rows, note=note)
    for row in rows:
        tracker.log_metric_row(row)
    tracker.log_metrics({"rows": rows, **provenance})

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
    tracker.close()


if __name__ == "__main__":
    main()
