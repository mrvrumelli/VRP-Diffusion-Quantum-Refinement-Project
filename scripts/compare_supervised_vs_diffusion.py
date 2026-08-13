"""Fast, self-contained comparison of P2.1 MatrixPredictor vs P3 diffusion on a data subset.

Trains both models from scratch on the same small, size-stratified subset (each capped by a
wall-clock time budget, ~15 minutes by default), then scores both -- plus the diffusion
model's full reverse-chain generation -- on a disjoint held-out set using the same
leakage-safe metrics as scripts/eval_m_ablation.py
(vrp_diffusion_quantum.eval.matrix_ablation.score_matrix_probabilities).

Unlike the full P3 recipe (GAT pretrain -> frozen encoder -> denoiser, see
scripts/run_gat_then_diffusion.sh), the diffusion model here uses node_encoder_type="linear"
trained end-to-end, so there is no separate GAT-pretrain stage -- this keeps the whole
comparison self-contained and fast, at the cost of matching the tuned recipe less closely.

Example:
  python scripts/compare_supervised_vs_diffusion.py \
      --source cvrp_s7799_n20-50-100_x66667/examples

  python scripts/compare_supervised_vs_diffusion.py \
      --source cvrp_s7799_n20-50-100_x66667/examples \
      --train-per-size 60 --mp-time-budget 600 --diff-time-budget 600 --skip-full-chain
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.eval.matrix_ablation import (
    score_matrix_probabilities,
    validate_disjoint_examples,
)
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    predict_matrix_one_shot,
    sample_constraint_matrix,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor, matrix_bce_loss
from vrp_diffusion_quantum.train.train_diffusion import train_constraint_denoiser
from vrp_diffusion_quantum.utils.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]

_TABLE_COLS = ("method", "f1", "auc", "precision", "recall", "bce", "threshold", "runtime_seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="dir of CVRPExample JSONs (e.g. an exported run's examples/)")
    parser.add_argument("--sizes", type=str, default="20,50,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=None, help="defaults to outputs/eval/compare_nn_vs_diffusion/<timestamp>")

    parser.add_argument("--train-per-size", type=int, default=40, help="examples per size for training (same pool for both models)")
    parser.add_argument("--selection-per-size", type=int, default=8, help="held-out examples per size for threshold selection")
    parser.add_argument("--test-per-size", type=int, default=8, help="untouched examples per size for final scoring")

    parser.add_argument("--mp-hidden-dim", type=int, default=64)
    parser.add_argument("--mp-learning-rate", type=float, default=0.001)
    parser.add_argument("--mp-time-budget", type=float, default=900.0, help="seconds")
    parser.add_argument("--mp-max-epochs", type=int, default=2000, help="safety cap; the time budget usually stops training first")
    parser.add_argument("--mp-patience", type=int, default=20, help="stop early if train_loss stops improving for this many epochs")

    parser.add_argument("--diff-hidden-dim", type=int, default=192)
    parser.add_argument("--diff-num-layers", type=int, default=8)
    parser.add_argument("--diff-time-embed-dim", type=int, default=192)
    parser.add_argument("--diff-learning-rate", type=float, default=3e-4)
    parser.add_argument("--diff-batch-size", type=int, default=16)
    parser.add_argument("--diff-num-timesteps", type=int, default=700)
    parser.add_argument("--diff-time-budget", type=float, default=900.0, help="seconds")
    parser.add_argument("--diff-max-epochs", type=int, default=200, help="safety cap; the time budget usually stops training first")
    parser.add_argument("--no-diff-augmentation", action="store_true", help="disable x9 geometric augmentation (faster per epoch, less data)")

    parser.add_argument("--skip-full-chain", action="store_true", help="skip the expensive full 700-step reverse-diffusion scoring pass")
    parser.add_argument("--full-chain-stride", type=int, default=5, help="visit every Nth diffusion timestep during full-chain generation")

    return parser.parse_args()


def _load_subset_by_size(
    source: Path, sizes: list[int], *, train_n: int, selection_n: int, test_n: int, seed: int
) -> dict[str, list[CVRPExample]]:
    """Deterministically carve disjoint, size-stratified train/selection/test pools."""
    rng = random.Random(seed)
    pools: dict[str, list[CVRPExample]] = {"train": [], "selection": [], "test": []}
    for size in sizes:
        paths = sorted(source.glob(f"cvrp{size}_*.json"))
        if not paths:
            raise FileNotFoundError(f"no cvrp{size}_*.json files under {source}")
        rng.shuffle(paths)
        need = train_n + selection_n + test_n
        if len(paths) < need:
            raise ValueError(f"only {len(paths)} cvrp{size} examples under {source}, need {need}")
        offset = 0
        for split, count in (("train", train_n), ("selection", selection_n), ("test", test_n)):
            for path in paths[offset : offset + count]:
                pools[split].append(load_example(path))
            offset += count
    return pools


def _train_supervised(
    train_examples: list[CVRPExample],
    *,
    hidden_dim: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    time_budget_seconds: float,
    max_epochs: int,
    patience: int,
) -> tuple[MatrixPredictor, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    model = MatrixPredictor(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    epochs_without_improve = 0
    started = time.perf_counter()
    print(f"P2.1 train: n={len(train_examples)} time_budget={time_budget_seconds:.0f}s device={device}", flush=True)
    for epoch in range(max_epochs):
        order = torch.randperm(len(train_examples), generator=torch.Generator().manual_seed(seed + epoch))
        total = 0.0
        for idx in order.tolist():
            example = train_examples[int(idx)]
            coords = torch.from_numpy(example.instance.customer_coords()).float().to(device)
            demands = torch.from_numpy(example.instance.customer_demands()).float().to(device)
            m_true = torch.from_numpy(example.constraint_matrix).float().to(device)
            optimizer.zero_grad()
            m_prob = model(coords, demands, float(example.instance.capacity))
            loss = matrix_bce_loss(m_prob, m_true)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        mean_loss = total / len(train_examples)
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "train_loss": mean_loss, "elapsed_seconds": elapsed})
        if epoch % 10 == 0:
            print(f"P2.1 epoch={epoch} train_loss={mean_loss:.4f} elapsed={elapsed:.1f}s", flush=True)
        if mean_loss < best_loss - 1e-5:
            best_loss = mean_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if elapsed >= time_budget_seconds:
            print(f"P2.1 time budget reached at epoch={epoch} elapsed={elapsed:.1f}s", flush=True)
            break
        if patience > 0 and epochs_without_improve >= patience:
            print(f"P2.1 early stop at epoch={epoch} (patience={patience})", flush=True)
            break
    model.eval()
    return model, history


@torch.no_grad()
def _eval_supervised(
    model: MatrixPredictor,
    examples: list[CVRPExample],
    device: torch.device,
    *,
    threshold: float | None,
    adaptive_threshold: bool,
) -> dict[str, float]:
    m_probs = []
    for example in examples:
        coords = torch.from_numpy(example.instance.customer_coords()).float().to(device)
        demands = torch.from_numpy(example.instance.customer_demands()).float().to(device)
        m_prob = model(coords, demands, float(example.instance.capacity)).detach().cpu().numpy()
        m_probs.append(m_prob.astype("float64"))
    return score_matrix_probabilities(
        examples, m_probs, threshold=threshold, adaptive_threshold=adaptive_threshold
    )


def _train_diffusion(
    train_examples: list[CVRPExample],
    val_examples: list[CVRPExample],
    *,
    hidden_dim: int,
    num_layers: int,
    time_embed_dim: int,
    schedule: BernoulliDiffusionSchedule,
    device: torch.device,
    seed: int,
    learning_rate: float,
    batch_size: int,
    time_budget_seconds: float,
    max_epochs: int,
    augmentation: bool,
) -> tuple[ConstraintDenoiser, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    model = ConstraintDenoiser(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        time_embed_dim=time_embed_dim,
        node_encoder_type="linear",
    )
    history: list[dict[str, Any]] = []

    def on_epoch_end(row: dict[str, Any]) -> None:
        history.append(row)
        print(
            f"P3 epoch={row.get('epoch')} train_loss={row.get('train_loss', float('nan')):.4f} "
            f"val_loss={row.get('val_loss', float('nan')):.4f} val_f1={row.get('val_f1', float('nan')):.4f} "
            f"epoch_time={row.get('epoch_runtime_seconds', float('nan')):.1f}s "
            f"total_time={row.get('total_runtime_seconds', float('nan')):.1f}s",
            flush=True,
        )

    print(
        f"P3 train: n={len(train_examples)} augmentation={augmentation} "
        f"time_budget={time_budget_seconds:.0f}s device={device}",
        flush=True,
    )
    train_constraint_denoiser(
        model,
        schedule,
        train_examples,
        val_examples=val_examples,
        num_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        device=device,
        same_size_batches=True,
        augmentation=augmentation,
        weighted_bce=True,
        pos_weight_power=0.5,
        adaptive_threshold=True,
        t_sample="high",
        sample_eval_examples=None,
        sample_eval_every=None,
        max_runtime_seconds=time_budget_seconds,
        on_epoch_end=on_epoch_end,
    )
    model.eval()
    return model, history


@torch.no_grad()
def _eval_diffusion(
    model: ConstraintDenoiser,
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
                model, schedule, coords=coords, demands=demands, capacity=capacity,
                customer_mask=mask, generator=gen,
            )
        elif mode == "full_chain":
            result = sample_constraint_matrix(
                model, schedule, coords=coords, demands=demands, capacity=capacity,
                customer_mask=mask, generator=gen, step_stride=step_stride,
            )
        else:
            raise ValueError(f"unknown diffusion mode: {mode}")
        m_probs.append(result.m_prob)
        m_hats.append(result.m_hat)
    return score_matrix_probabilities(
        examples, m_probs, m_hats=m_hats, hard_from_hats=(mode == "full_chain"),
        threshold=threshold, adaptive_threshold=adaptive_threshold,
    )


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = " | ".join(f"{c:>14}" for c in _TABLE_COLS)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for c in _TABLE_COLS:
            v = row.get(c, "")
            cells.append(f"{v:14.4f}" if isinstance(v, float) else f"{v!s:>14}")
        print(" | ".join(cells))


def main() -> None:
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s]
    device = resolve_device(args.device)
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output_root = args.output if args.output is not None else (
        ROOT / "outputs" / "eval" / "compare_nn_vs_diffusion"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"device={device} source={source} output={output_root}", flush=True)

    wall_start = time.perf_counter()
    print("loading subset…", flush=True)
    pools = _load_subset_by_size(
        source, sizes,
        train_n=args.train_per_size, selection_n=args.selection_per_size, test_n=args.test_per_size,
        seed=args.seed,
    )
    train_examples, selection_examples, test_examples = pools["train"], pools["selection"], pools["test"]
    validate_disjoint_examples(selection_examples, test_examples)
    validate_disjoint_examples(train_examples, test_examples)
    validate_disjoint_examples(train_examples, selection_examples)
    print(
        f"train={len(train_examples)} selection={len(selection_examples)} test={len(test_examples)} "
        f"sizes={sizes}",
        flush=True,
    )

    # --- P2.1 supervised MatrixPredictor ---
    started = time.perf_counter()
    predictor, mp_history = _train_supervised(
        train_examples,
        hidden_dim=args.mp_hidden_dim, learning_rate=args.mp_learning_rate, device=device,
        seed=args.seed, time_budget_seconds=args.mp_time_budget, max_epochs=args.mp_max_epochs,
        patience=args.mp_patience,
    )
    mp_train_runtime = time.perf_counter() - started
    mp_selection = _eval_supervised(predictor, selection_examples, device, threshold=None, adaptive_threshold=True)
    started = time.perf_counter()
    mp_test = _eval_supervised(
        predictor, test_examples, device,
        threshold=float(mp_selection["threshold"]), adaptive_threshold=False,
    )
    p21_row = {"method": "P2.1_supervised", **mp_test, "runtime_seconds": time.perf_counter() - started}

    # --- P3 diffusion ConstraintDenoiser ---
    schedule = BernoulliDiffusionSchedule(num_timesteps=args.diff_num_timesteps, beta_start=1e-4, beta_end=2e-2)
    started = time.perf_counter()
    denoiser, diff_history = _train_diffusion(
        train_examples, selection_examples,
        hidden_dim=args.diff_hidden_dim, num_layers=args.diff_num_layers,
        time_embed_dim=args.diff_time_embed_dim, schedule=schedule, device=device, seed=args.seed,
        learning_rate=args.diff_learning_rate, batch_size=args.diff_batch_size,
        time_budget_seconds=args.diff_time_budget, max_epochs=args.diff_max_epochs,
        augmentation=not args.no_diff_augmentation,
    )
    diff_train_runtime = time.perf_counter() - started
    schedule = schedule.to(device)

    one_shot_selection = _eval_diffusion(
        denoiser, schedule, selection_examples, device=device, mode="one_shot",
        seed=args.seed, threshold=None, adaptive_threshold=True,
    )
    started = time.perf_counter()
    one_shot_test = _eval_diffusion(
        denoiser, schedule, test_examples, device=device, mode="one_shot", seed=args.seed,
        threshold=float(one_shot_selection["threshold"]), adaptive_threshold=False,
    )
    one_shot_row = {"method": "P3_one_shot", **one_shot_test, "runtime_seconds": time.perf_counter() - started}

    rows = [p21_row, one_shot_row]
    full_row: dict[str, Any] | None = None
    if not args.skip_full_chain:
        print(f"scoring P3 full-chain (step_stride={args.full_chain_stride})…", flush=True)
        started = time.perf_counter()
        full_test = _eval_diffusion(
            denoiser, schedule, test_examples, device=device, mode="full_chain", seed=args.seed,
            step_stride=args.full_chain_stride, threshold=0.5, adaptive_threshold=False,
        )
        full_row = {"method": "P3_full_chain", **full_test, "runtime_seconds": time.perf_counter() - started}
        rows.append(full_row)

    print("\n=== P2.1 supervised vs P3 diffusion (constraint matrix M) ===\n", flush=True)
    _print_table(rows)
    print(f"\nΔF1 one-shot vs P2.1 = {one_shot_row['f1'] - p21_row['f1']:+.4f}", flush=True)
    if full_row is not None:
        print(f"ΔF1 full-chain vs P2.1 = {full_row['f1'] - p21_row['f1']:+.4f}", flush=True)
    print(
        f"\nP2.1 train time={mp_train_runtime:.1f}s ({len(mp_history)} epochs)  "
        f"P3 train time={diff_train_runtime:.1f}s ({len(diff_history)} epochs)  "
        f"total wall time={time.perf_counter() - wall_start:.1f}s",
        flush=True,
    )

    metrics_path = output_root / "comparison_metrics.json"
    metrics_path.write_text(json.dumps({
        "args": vars(args) | {"source": str(source), "sizes": sizes},
        "counts": {
            "train": len(train_examples), "selection": len(selection_examples), "test": len(test_examples),
        },
        "runtimes_seconds": {
            "mp_train": mp_train_runtime, "diff_train": diff_train_runtime,
            "total_wall": time.perf_counter() - wall_start,
        },
        "rows": rows,
        "mp_history": mp_history,
        "diff_history": diff_history,
    }, indent=2, default=str))
    print(f"\nwrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
