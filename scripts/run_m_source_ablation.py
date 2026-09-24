"""Run a small reproducible ablation over constraint-matrix sources.

This is a smoke-scale report generator for the question: how much value comes from
the route-membership constraint matrix ``M``, and does diffusion add value over
simpler matrix sources?

Default input is the committed strong-reference matrix examples:

    outputs/label_audit/s7799_strong_reference/accepted_matrix_examples

The six reported arms are:

* ``no_m_mask``: dense off-diagonal matrix, equivalent to removing the M mask.
* ``ground_truth_m``: oracle labelled route-membership matrix.
* ``predicted_m``: deterministic capacity-and-distance heuristic matrix.
* ``random_m``: seeded random matrix with the training split's positive-pair density.
* ``supervised_m``: tiny supervised MatrixPredictor trained on the selected train split.
* ``diffusion_m``: tiny self-contained ConstraintDenoiser trained on the same train split.

The default sizes are deliberately small enough to run on CPU and produce a reproducible
report from committed data. Treat the result as a controlled smoke ablation, not as the
final paper-scale comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vrp_diffusion_quantum.data.dataset import (
    CVRPBatch,
    IndexedJSONDataset,
    collate_batch,
    size_homogeneous_chunks,
)
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.eval.matrix_ablation import (
    score_matrix_probabilities,
    validate_disjoint_examples,
)
from vrp_diffusion_quantum.eval.routing import (
    RoutingEvaluation,
    evaluate_decoded_matrix,
    summarize_routing_evaluations,
)
from vrp_diffusion_quantum.inference.policy_support import batch_to_device
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    sample_constraint_matrix,
)
from vrp_diffusion_quantum.metrics.matrix_metrics import off_diagonal_pairs
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor, matrix_bce_loss
from vrp_diffusion_quantum.train.train_diffusion import diffusion_matrix_bce_loss
from vrp_diffusion_quantum.utils.experiment import git_commit_hash, hash_dataset
from vrp_diffusion_quantum.utils.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    ROOT / "outputs" / "label_audit" / "s7799_strong_reference" / "accepted_matrix_examples"
)
DEFAULT_REPORT = ROOT / "docs" / "constraint_matrix_diffusion_ablation_report.md"
DEFAULT_ASSET_DIR = ROOT / "docs" / "assets" / "m_source_ablation"

_TABLE_COLS = (
    "method",
    "matrix_f1",
    "matrix_auc",
    "route_gap_percent",
    "delta_gap_vs_no_m",
    "route_feasible_rate",
    "route_num_vehicles",
    "route_vehicle_delta",
    "route_vehicle_inflation_ratio",
    "route_singleton_fraction",
    "route_positive_edge_recall",
    "threshold",
    "runtime_seconds",
    "f1_n20",
    "f1_n50",
    "f1_n100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--train-per-size", type=int, default=4)
    parser.add_argument("--val-per-size", type=int, default=2)
    parser.add_argument("--test-per-size", type=int, default=4)
    parser.add_argument("--supervised-hidden-dim", type=int, default=32)
    parser.add_argument("--supervised-epochs", type=int, default=4)
    parser.add_argument("--supervised-learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=32)
    parser.add_argument("--diffusion-layers", type=int, default=2)
    parser.add_argument("--diffusion-time-embed-dim", type=int, default=32)
    parser.add_argument("--diffusion-timesteps", type=int, default=30)
    parser.add_argument("--diffusion-epochs", type=int, default=1)
    parser.add_argument("--diffusion-batch-size", type=int, default=2)
    parser.add_argument("--diffusion-learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-step-stride", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _normalise_matrix(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64)
    out = np.clip((out + out.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(out, 0.0)
    return out


def dense_no_mask_matrix(example: CVRPExample) -> np.ndarray:
    """Dense off-diagonal matrix: the decoder receives no useful M restriction."""
    n = example.instance.n_customers
    matrix = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def ground_truth_matrix(example: CVRPExample) -> np.ndarray:
    """Oracle route-membership matrix from the labelled solution."""
    return example.constraint_matrix.astype(np.float64)


def capacity_distance_predicted_matrix(example: CVRPExample) -> np.ndarray:
    """Deterministic, non-learned predicted M from capacity feasibility and distance."""
    coords = example.instance.customer_coords()
    demands = example.instance.customer_demands()
    n = example.instance.n_customers
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    off_diag = ~np.eye(n, dtype=bool)
    scale = float(np.median(distances[off_diag])) if off_diag.any() else 1.0
    scale = max(scale, 1e-8)
    pair_fits = demands[:, None] + demands[None, :] <= example.instance.capacity + 1e-9
    score = np.exp(-distances / scale) * pair_fits.astype(np.float64)
    return _normalise_matrix(score)


def random_matrix(
    example: CVRPExample,
    *,
    positive_probability: float,
    seed: int,
    example_index: int,
) -> np.ndarray:
    """Seeded symmetric random hard M with a fixed positive-pair density."""
    if not 0.0 <= positive_probability <= 1.0:
        raise ValueError("positive_probability must be in [0, 1]")
    n = example.instance.n_customers
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, example_index]))
    upper = rng.random((n, n)) < positive_probability
    matrix = np.triu(upper.astype(np.float64), k=1)
    return matrix + matrix.T


def _positive_pair_rate(examples: Sequence[CVRPExample]) -> float:
    pairs = [
        off_diagonal_pairs(example.constraint_matrix.astype(np.float64))
        for example in examples
        if example.instance.n_customers > 1
    ]
    if not pairs:
        raise ValueError("cannot estimate positive-pair rate without off-diagonal pairs")
    return float(np.concatenate(pairs).mean())


def split_examples_by_size(
    data_dir: str | Path,
    *,
    sizes: Sequence[int],
    train_per_size: int,
    val_per_size: int,
    test_per_size: int,
    seed: int,
) -> dict[str, list[CVRPExample]]:
    """Load and split examples deterministically within each CVRP size."""
    for label, value in (
        ("train_per_size", train_per_size),
        ("val_per_size", val_per_size),
        ("test_per_size", test_per_size),
    ):
        if value < 1:
            raise ValueError(f"{label} must be >= 1, got {value}")

    dataset = IndexedJSONDataset(data_dir, sizes=sizes)
    if not dataset:
        raise ValueError(f"no examples found under {data_dir}")

    by_size = dataset.indices_by_size()
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    needed = train_per_size + val_per_size + test_per_size
    for size in sizes:
        candidates = by_size.get(int(size), [])
        if len(candidates) < needed:
            raise ValueError(
                f"size {size} needs {needed} examples, found {len(candidates)} in {data_dir}"
            )
        selected = rng.choice(candidates, size=needed, replace=False).tolist()
        boundaries = (
            ("train", 0, train_per_size),
            ("val", train_per_size, train_per_size + val_per_size),
            ("test", train_per_size + val_per_size, needed),
        )
        for split_name, start, stop in boundaries:
            splits[split_name].extend(dataset[int(index)] for index in selected[start:stop])

    validate_disjoint_examples(splits["train"], splits["val"])
    validate_disjoint_examples(splits["train"], splits["test"])
    validate_disjoint_examples(splits["val"], splits["test"])
    return splits


def _batch_customer_tensors(
    batch: CVRPBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = batch.customer_node_indices.clamp(min=0)
    coords = torch.gather(batch.coords, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    demands = torch.gather(batch.demands, 1, idx)
    mask = batch.customer_mask
    return (
        coords * mask.unsqueeze(-1),
        demands * mask,
        batch.capacity,
        batch.constraint_matrix,
        mask,
    )


def train_supervised_matrix_predictor(
    examples: Sequence[CVRPExample],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> tuple[MatrixPredictor, list[dict[str, float]]]:
    """Train the supervised matrix predictor on a small selected split."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    torch.manual_seed(seed)
    model = MatrixPredictor(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    train_list = list(examples)
    started = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(
            len(train_list),
            generator=torch.Generator(device="cpu").manual_seed(seed + epoch),
        )
        total = 0.0
        for index in order.tolist():
            example = train_list[int(index)]
            coords = torch.from_numpy(example.instance.customer_coords()).float().to(device)
            demands = torch.from_numpy(example.instance.customer_demands()).float().to(device)
            target = torch.from_numpy(example.constraint_matrix).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coords, demands, float(example.instance.capacity))
            loss = matrix_bce_loss(prediction, target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total / len(train_list),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    model.eval()
    return model, history


@torch.no_grad()
def predict_supervised_matrices(
    model: MatrixPredictor,
    examples: Sequence[CVRPExample],
    *,
    device: torch.device,
) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for example in examples:
        coords = torch.from_numpy(example.instance.customer_coords()).float().to(device)
        demands = torch.from_numpy(example.instance.customer_demands()).float().to(device)
        prediction = model(coords, demands, float(example.instance.capacity))
        matrices.append(_normalise_matrix(prediction.detach().cpu().numpy()))
    return matrices


def train_diffusion_matrix_model(
    examples: Sequence[CVRPExample],
    *,
    hidden_dim: int,
    num_layers: int,
    time_embed_dim: int,
    num_timesteps: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> tuple[ConstraintDenoiser, BernoulliDiffusionSchedule, list[dict[str, float]]]:
    """Train a tiny self-contained denoiser for the diffusion M arm."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    torch.manual_seed(seed)
    model = ConstraintDenoiser(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        time_embed_dim=time_embed_dim,
        node_encoder_type="linear",
    ).to(device)
    schedule = BernoulliDiffusionSchedule(num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    train_list = list(examples)
    started = time.perf_counter()
    for epoch in range(epochs):
        total = 0.0
        batches = 0
        generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 + epoch)
        for batch_index, chunk in enumerate(
            size_homogeneous_chunks(train_list, batch_size, generator=generator)
        ):
            batch = batch_to_device(collate_batch(chunk), device)
            coords, demands, capacity, target, mask = _batch_customer_tensors(batch)
            noise_generator = torch.Generator(device="cpu").manual_seed(
                seed + 100_000 * (epoch + 1) + batch_index
            )
            timesteps = schedule.sample_timesteps(
                len(chunk),
                device=device,
                generator=noise_generator,
                mode="uniform",
            )
            noisy = schedule.q_sample(
                target,
                timesteps,
                customer_mask=mask,
                generator=noise_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(coords, demands, capacity, noisy, timesteps, customer_mask=mask)
            loss = diffusion_matrix_bce_loss(
                logits,
                target,
                mask,
                weighted=True,
                pos_weight_power=0.5,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total / max(batches, 1),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    model.eval()
    return model, schedule, history


@torch.no_grad()
def predict_diffusion_matrices(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    examples: Sequence[CVRPExample],
    *,
    device: torch.device,
    seed: int,
    step_stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(m_prob, m_hat)`` for the diffusion M arm."""
    m_probs: list[np.ndarray] = []
    m_hats: list[np.ndarray] = []
    for index, example in enumerate(examples):
        coords, demands, capacity, _, mask = example_to_model_inputs(example, device=device)
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        sampled = sample_constraint_matrix(
            model,
            schedule,
            coords=coords,
            demands=demands,
            capacity=capacity,
            customer_mask=mask,
            generator=generator,
            threshold=0.5,
            step_stride=step_stride,
            transition_mode="deterministic",
        )
        m_probs.append(_normalise_matrix(sampled.m_prob))
        m_hats.append(_normalise_matrix(sampled.m_hat))
    return m_probs, m_hats


def _score_arm(
    method: str,
    examples: Sequence[CVRPExample],
    matrices: Sequence[np.ndarray],
    *,
    threshold: float | None,
    adaptive_threshold: bool,
    route_matrices: Sequence[np.ndarray] | None = None,
    hard_matrices: Sequence[np.ndarray] | None = None,
    hard_from_hats: bool = False,
    runtime_seconds: float,
) -> dict[str, Any]:
    metrics = score_matrix_probabilities(
        examples,
        matrices,
        m_hats=hard_matrices,
        hard_from_hats=hard_from_hats,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
    )
    route_source = route_matrices if route_matrices is not None else matrices
    route_results = [
        evaluate_decoded_matrix(example, matrix, threshold=float(metrics["threshold"]))
        for example, matrix in zip(examples, route_source, strict=True)
    ]
    route_summary = summarize_routing_evaluations(route_results)
    row: dict[str, Any] = {
        "method": method,
        "matrix_f1": metrics["f1"],
        "matrix_auc": metrics["auc"],
        "matrix_precision": metrics["precision"],
        "matrix_recall": metrics["recall"],
        "matrix_bce": metrics["bce"],
        "threshold": metrics["threshold"],
        "runtime_seconds": runtime_seconds,
        "route_gap_percent": route_summary["route_mean_cost_gap_percent"],
        "route_feasible_rate": route_summary["route_feasible_rate"],
        "route_num_vehicles": route_summary["route_mean_num_vehicles"],
        "route_vehicle_delta": route_summary["route_mean_vehicle_delta"],
        "route_vehicle_inflation_ratio": route_summary[
            "route_mean_vehicle_inflation_ratio"
        ],
        "route_singleton_fraction": route_summary["route_mean_singleton_route_fraction"],
        "route_positive_edge_recall": route_summary["route_matrix_positive_edge_recall"],
        "route_decode_runtime_seconds": route_summary["route_mean_decode_runtime_seconds"],
        "route_repair_count": route_summary["route_repair_count"],
        "route_matrix_pair_accuracy": route_summary["route_mean_matrix_pair_accuracy"],
        "num_examples": float(len(examples)),
    }
    for key, value in metrics.items():
        if key.startswith("f1_n"):
            row[key] = value
    by_size: dict[int, list[RoutingEvaluation]] = {}
    for result in route_results:
        by_size.setdefault(result.n_customers, []).append(result)
    for size, results in sorted(by_size.items()):
        row[f"route_gap_n{size}"] = summarize_routing_evaluations(results)[
            "route_mean_cost_gap_percent"
        ]
    return row


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _gap_change_sentence(label: str, row: dict[str, Any]) -> str:
    delta = float(row["delta_gap_vs_no_m"])
    verb = "reduces" if delta >= 0.0 else "increases"
    return f"- {label} {verb} route gap by {abs(delta):.2f} percentage points versus no `M` mask."


def _write_markdown_report(
    path: Path,
    *,
    rows: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
    csv_path: Path,
    metrics_path: Path,
) -> None:
    oracle = next(row for row in rows if row["method"] == "ground_truth_m")
    supervised = next(row for row in rows if row["method"] == "supervised_m")
    diffusion = next(row for row in rows if row["method"] == "diffusion_m")
    lines = [
        "# Ablation Report: Constraint Matrix M and Diffusion",
        "",
        "This smoke ablation uses the committed strong-reference matrix examples and fixed seeds. "
        "It evaluates the routing utility of different route-membership matrix sources with the "
        "same matrix-to-route decoder, so it isolates the value of `M` itself before full policy "
        "training scale.",
        "",
        f"- Data: `{provenance['data_dir']}`",
        f"- Dataset hash: `{provenance['dataset_hash']}`",
        f"- Seed: `{provenance['seed']}`",
        f"- Train/val/test counts: {provenance['counts']}",
        f"- Git commit: `{provenance['git_commit']}`",
        f"- CSV: `{csv_path.relative_to(ROOT)}`",
        f"- Metrics JSON: `{metrics_path.relative_to(ROOT)}`",
        "",
        "## Results",
        "",
        "| " + " | ".join(_TABLE_COLS) + " |",
        "| " + " | ".join("---" for _ in _TABLE_COLS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(col)) for col in _TABLE_COLS) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _gap_change_sentence("Oracle `M`", oracle),
            _gap_change_sentence("Supervised `M`", supervised),
            _gap_change_sentence("Diffusion `M`", diffusion),
            (
                f"- Diffusion vs supervised: matrix F1 delta "
                f"{diffusion['matrix_f1'] - supervised['matrix_f1']:+.4f}, route-gap delta "
                f"{supervised['route_gap_percent'] - diffusion['route_gap_percent']:+.2f} "
                "percentage points."
            ),
            "",
            "The oracle row is the upper bound for this matrix decoder. The no-mask and "
            "random rows "
            "show what is lost when the route-membership structure is absent or uninformative. "
            "The supervised and diffusion rows show whether the learned matrix sources recover "
            "that oracle value in this small self-contained run.",
            "",
            "## Caveat",
            "",
            "This is a CPU-friendly smoke report, not the final paper-scale ablation. "
            "The diffusion "
            "arm trains a tiny linear-encoder denoiser from scratch on the selected split because "
            "no full trained diffusion checkpoint is present in this checkout.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _summarize_history(history: Sequence[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {}
    return {
        "epochs": float(len(history)),
        "first_loss": float(history[0]["train_loss"]),
        "last_loss": float(history[-1]["train_loss"]),
        "elapsed_seconds": float(history[-1]["elapsed_seconds"]),
    }


def _paths_by_split(splits: dict[str, list[CVRPExample]]) -> dict[str, list[str]]:
    return {
        name: [example.instance.instance_id for example in examples]
        for name, examples in splits.items()
    }


def run_ablation(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_dir = _resolve_path(args.data_dir)
    report_path = _resolve_path(args.report)
    asset_dir = _resolve_path(args.asset_dir)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)

    splits = split_examples_by_size(
        data_dir,
        sizes=args.sizes,
        train_per_size=args.train_per_size,
        val_per_size=args.val_per_size,
        test_per_size=args.test_per_size,
        seed=args.seed,
    )
    train_examples = splits["train"]
    val_examples = splits["val"]
    test_examples = splits["test"]
    positive_rate = _positive_pair_rate(train_examples)
    print(
        f"data={data_dir} device={device} seed={args.seed} "
        f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)} "
        f"positive_pair_rate={positive_rate:.4f}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    no_mask = [dense_no_mask_matrix(example) for example in test_examples]
    rows.append(
        _score_arm(
            "no_m_mask",
            test_examples,
            no_mask,
            threshold=0.5,
            adaptive_threshold=False,
            runtime_seconds=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    oracle = [ground_truth_matrix(example) for example in test_examples]
    rows.append(
        _score_arm(
            "ground_truth_m",
            test_examples,
            oracle,
            threshold=0.5,
            adaptive_threshold=False,
            runtime_seconds=time.perf_counter() - started,
        )
    )

    heuristic_val = [capacity_distance_predicted_matrix(example) for example in val_examples]
    heuristic_threshold = score_matrix_probabilities(
        val_examples,
        heuristic_val,
        threshold=None,
        adaptive_threshold=True,
    )["threshold"]
    started = time.perf_counter()
    heuristic = [capacity_distance_predicted_matrix(example) for example in test_examples]
    rows.append(
        _score_arm(
            "predicted_m",
            test_examples,
            heuristic,
            threshold=float(heuristic_threshold),
            adaptive_threshold=False,
            runtime_seconds=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    random = [
        random_matrix(
            example,
            positive_probability=positive_rate,
            seed=args.seed,
            example_index=index,
        )
        for index, example in enumerate(test_examples)
    ]
    rows.append(
        _score_arm(
            "random_m",
            test_examples,
            random,
            threshold=0.5,
            adaptive_threshold=False,
            runtime_seconds=time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    supervised, supervised_history = train_supervised_matrix_predictor(
        train_examples,
        hidden_dim=args.supervised_hidden_dim,
        epochs=args.supervised_epochs,
        learning_rate=args.supervised_learning_rate,
        device=device,
        seed=args.seed,
    )
    supervised_train_runtime = time.perf_counter() - started
    supervised_val = predict_supervised_matrices(supervised, val_examples, device=device)
    supervised_threshold = score_matrix_probabilities(
        val_examples,
        supervised_val,
        threshold=None,
        adaptive_threshold=True,
    )["threshold"]
    started = time.perf_counter()
    supervised_test = predict_supervised_matrices(supervised, test_examples, device=device)
    rows.append(
        _score_arm(
            "supervised_m",
            test_examples,
            supervised_test,
            threshold=float(supervised_threshold),
            adaptive_threshold=False,
            runtime_seconds=supervised_train_runtime + time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    denoiser, schedule, diffusion_history = train_diffusion_matrix_model(
        train_examples,
        hidden_dim=args.diffusion_hidden_dim,
        num_layers=args.diffusion_layers,
        time_embed_dim=args.diffusion_time_embed_dim,
        num_timesteps=args.diffusion_timesteps,
        epochs=args.diffusion_epochs,
        batch_size=args.diffusion_batch_size,
        learning_rate=args.diffusion_learning_rate,
        device=device,
        seed=args.seed,
    )
    diffusion_train_runtime = time.perf_counter() - started
    started = time.perf_counter()
    diffusion_probs, diffusion_hats = predict_diffusion_matrices(
        denoiser,
        schedule,
        test_examples,
        device=device,
        seed=args.seed,
        step_stride=args.diffusion_step_stride,
    )
    rows.append(
        _score_arm(
            "diffusion_m",
            test_examples,
            diffusion_probs,
            hard_matrices=diffusion_hats,
            hard_from_hats=True,
            route_matrices=diffusion_hats,
            threshold=0.5,
            adaptive_threshold=False,
            runtime_seconds=diffusion_train_runtime + time.perf_counter() - started,
        )
    )

    no_m_gap = float(rows[0]["route_gap_percent"])
    for row in rows:
        row["delta_gap_vs_no_m"] = no_m_gap - float(row["route_gap_percent"])

    csv_path = asset_dir / "ablation_table.csv"
    metrics_path = asset_dir / "ablation_metrics.json"
    provenance = {
        "data_dir": str(data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir),
        "dataset_hash": hash_dataset(data_dir),
        "git_commit": git_commit_hash(ROOT),
        "seed": args.seed,
        "device": str(device),
        "sizes": list(args.sizes),
        "counts": {
            "train": len(train_examples),
            "val": len(val_examples),
            "test": len(test_examples),
            "train_per_size": args.train_per_size,
            "val_per_size": args.val_per_size,
            "test_per_size": args.test_per_size,
        },
        "selected_instance_ids": _paths_by_split(splits),
        "positive_pair_rate": positive_rate,
        "supervised": {
            "hidden_dim": args.supervised_hidden_dim,
            "epochs": args.supervised_epochs,
            "learning_rate": args.supervised_learning_rate,
            "history": supervised_history,
            "summary": _summarize_history(supervised_history),
        },
        "diffusion": {
            "hidden_dim": args.diffusion_hidden_dim,
            "layers": args.diffusion_layers,
            "time_embed_dim": args.diffusion_time_embed_dim,
            "timesteps": args.diffusion_timesteps,
            "epochs": args.diffusion_epochs,
            "batch_size": args.diffusion_batch_size,
            "learning_rate": args.diffusion_learning_rate,
            "step_stride": args.diffusion_step_stride,
            "history": diffusion_history,
            "summary": _summarize_history(diffusion_history),
        },
    }
    asset_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows)
    metrics_path.write_text(
        json.dumps(
            {
                "provenance": provenance,
                "rows": rows,
                "args": vars(args),
                "routing_columns": [
                    key for key in rows[0] if key.startswith("route_") or key == "delta_gap_vs_no_m"
                ],
                "row_dicts": [dict(row) for row in rows],
            },
            indent=2,
            default=lambda obj: asdict(obj) if hasattr(obj, "__dataclass_fields__") else str(obj),
        )
        + "\n"
    )
    _write_markdown_report(
        report_path,
        rows=rows,
        provenance=provenance,
        csv_path=csv_path,
        metrics_path=metrics_path,
    )

    print(f"wrote {report_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {metrics_path}", flush=True)
    print("\nmethod | matrix_f1 | route_gap_percent | delta_gap_vs_no_m", flush=True)
    print("--- | --- | --- | ---", flush=True)
    for row in rows:
        print(
            f"{row['method']} | {row['matrix_f1']:.4f} | "
            f"{row['route_gap_percent']:.4f} | {row['delta_gap_vs_no_m']:.4f}",
            flush=True,
        )
    return rows, provenance


def main() -> None:
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
