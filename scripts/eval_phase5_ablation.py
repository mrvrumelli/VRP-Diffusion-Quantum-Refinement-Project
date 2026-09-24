"""Phase 5 kickoff: the full no/random/ground-truth/supervised/diffusion M ablation.

  python scripts/eval_phase5_ablation.py --config configs/eval/phase5_ablation.yaml

Resolving the arm list
-----------------------
``AGENTS.md`` section 8 lists Phase 5's ablation as six comma-separated items: "no `M`,
ground-truth `M`, predicted `M`, random `M`, supervised `M`, diffusion `M`". Its own section 16
("Known risks and mitigations"), however, lists only four: "no-`M`, ground-truth-`M`,
supervised-`M`, and diffusion-`M` ablations" -- and this codebase only has two real
M-predicting model families (the P2.1 non-diffusion ``MatrixPredictor`` and the P3 diffusion
denoiser). Reading the two lists together, "predicted `M`" in section 8 is loose umbrella
phrasing for "supervised `M` + diffusion `M`" jointly, not a distinct sixth arm.

The true arm set is five concepts -- ``no_M``, ``random_M``, ``ground_truth_M``,
``supervised_M``, ``diffusion_M`` -- where ``diffusion_M`` itself splits along two independent
axes: pooled vs. per-size checkpoint (an open recipe question this run is designed to help
answer, see ``docs/stochastic_reference_probe.md``), and one-shot vs. full-chain sampling
(mirroring ``scripts/eval_m_ablation.py``'s existing P3 rows). That yields the 8 table rows
this script actually produces:

    no_M, random_M, ground_truth_M, supervised_M,
    diffusion_M_pooled_one_shot, diffusion_M_pooled_full_chain,
    diffusion_M_persize_one_shot, diffusion_M_persize_full_chain

Every arm ultimately produces one ``[n, n]`` matrix per example (even ``no_M``, via
``build_constraint_matrix`` on its constructed routes), so all 8 rows flow through the same two
scorers: ``score_matrix_probabilities`` (matrix-classification metrics, over the untouched test
split) and ``evaluate_decoded_matrix``/``summarize_routing_evaluations`` (routing/cost-gap
metrics, over the audited accepted-subset pool). A diffusion row is skipped (logged, not a
crash) when its checkpoint isn't configured; a per-size row with partial size coverage still
runs, restricted to the sizes it has a checkpoint for (see ``diffusion_persize_available_sizes``
in the written provenance).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from vrp_diffusion_quantum.data.dataset import load_dataset, load_examples_by_size
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.eval.baselines import clarke_wright_routes, random_constraint_matrix
from vrp_diffusion_quantum.eval.matrix_ablation import (
    predict_matrix_predictor_probs,
    report_instance_id_overlap,
    score_matrix_probabilities,
    train_matrix_predictor,
    validate_disjoint_examples,
)
from vrp_diffusion_quantum.eval.routing import (
    RoutingEvaluation,
    evaluate_decoded_matrix,
    summarize_routing_evaluations,
)
from vrp_diffusion_quantum.inference.predict_matrix import (
    PredictionMode,
    load_denoiser_checkpoint,
    predict_matrix_batch,
    predict_matrix_persize,
    select_examples_by_size,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix
from vrp_diffusion_quantum.utils.experiment import ExperimentTracker, hash_dataset
from vrp_diffusion_quantum.utils.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "eval" / "phase5_ablation.yaml"

_MARKDOWN_COLS = (
    "method",
    "f1",
    "auc",
    "threshold",
    "route_feasible_rate",
    "route_mean_cost_gap_percent",
    "route_mean_num_vehicles",
    "runtime_seconds",
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


def _no_m_matrix(example: CVRPExample) -> np.ndarray:
    routes = clarke_wright_routes(example.instance)
    return build_constraint_matrix(routes, example.instance.n_customers).astype(np.float64)


def _routing_metrics_with_by_size(
    examples: list[CVRPExample],
    matrices: list[np.ndarray],
    *,
    threshold: float,
) -> dict[str, float | int]:
    results: list[RoutingEvaluation] = [
        evaluate_decoded_matrix(example, matrix, threshold=threshold)
        for example, matrix in zip(examples, matrices, strict=True)
    ]
    out: dict[str, float | int] = dict(summarize_routing_evaluations(results))
    by_size: dict[int, list[RoutingEvaluation]] = {}
    for example, result in zip(examples, results, strict=True):
        by_size.setdefault(example.instance.n_customers, []).append(result)
    for size in sorted(by_size):
        size_summary = summarize_routing_evaluations(by_size[size])
        for key, value in size_summary.items():
            if key != "route_num_examples":
                out[f"{key}_n{size}"] = value
    return out


def _build_arm_row(
    method: str,
    *,
    test_examples: list[CVRPExample],
    test_probs: list[np.ndarray],
    test_hats: list[np.ndarray] | None,
    hard_from_hats: bool,
    accepted_examples: list[CVRPExample],
    accepted_probs: list[np.ndarray],
    threshold: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    matrix_metrics = score_matrix_probabilities(
        test_examples,
        test_probs,
        m_hats=test_hats,
        hard_from_hats=hard_from_hats,
        threshold=threshold,
        adaptive_threshold=False,
    )
    routing_metrics = _routing_metrics_with_by_size(
        accepted_examples, accepted_probs, threshold=threshold
    )
    return {
        "method": method,
        "runtime_seconds": runtime_seconds,
        **matrix_metrics,
        **routing_metrics,
    }


def _load_diffusion_checkpoint(
    path: Path, device: torch.device
) -> tuple[ConstraintDenoiser, BernoulliDiffusionSchedule]:
    model, payload = load_denoiser_checkpoint(path, device=device)
    schedule_cfg = (payload.get("extra") or {}).get("schedule") or {}
    schedule = BernoulliDiffusionSchedule(
        num_timesteps=int(schedule_cfg.get("num_timesteps", 700)),
        beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
        beta_end=float(schedule_cfg.get("beta_end", 2e-2)),
    ).to(device)
    return model, schedule


def _diffusion_threshold(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    selection_examples: list[CVRPExample],
    *,
    mode: PredictionMode,
    device: torch.device,
    seed: int,
    step_stride: int,
) -> float:
    if mode == "full_chain":
        return 0.5
    probs, _ = predict_matrix_batch(
        model,
        schedule,
        selection_examples,
        mode=mode,
        device=device,
        seed=seed,
        step_stride=step_stride,
    )
    metrics = score_matrix_probabilities(selection_examples, probs)
    return float(metrics["threshold"])


def _persize_threshold(
    checkpoints_by_size: dict[int, tuple[ConstraintDenoiser, BernoulliDiffusionSchedule]],
    selection_examples: list[CVRPExample],
    *,
    mode: PredictionMode,
    device: torch.device,
    seed: int,
    step_stride: int,
) -> float:
    if mode == "full_chain":
        return 0.5
    probs, _ = predict_matrix_persize(
        checkpoints_by_size,
        selection_examples,
        mode=mode,
        device=device,
        seed=seed,
        step_stride=step_stride,
    )
    metrics = score_matrix_probabilities(selection_examples, probs)
    return float(metrics["threshold"])


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = " | ".join(f"{c:>26}" for c in _MARKDOWN_COLS)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for c in _MARKDOWN_COLS:
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:26.4f}" if v == v else f"{'nan':>26}")
            else:
                cells.append(f"{v!s:>26}")
        print(" | ".join(cells))


def _fmt_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if value == value else "nan"
    return str(value)


def _write_markdown_table(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    note: str,
) -> None:
    lines = [
        "# Phase 5 ablation: no / random / ground-truth / supervised / diffusion M",
        "",
        note,
        "",
        "Full per-size and routing detail (route_mean_cost_gap_percent_n20/50/100, etc.) is in "
        "the sibling `ablation_table.csv` / `ablation_metrics.json`.",
        "",
        "| " + " | ".join(_MARKDOWN_COLS) + " |",
        "| " + " | ".join("---" for _ in _MARKDOWN_COLS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(row.get(c, "")) for c in _MARKDOWN_COLS) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(cfg_path.read_text())
    seed = int(config["seed"])
    device = resolve_device(config.get("device", "auto"))

    ds_cfg = config["dataset"]
    train_path = _resolve_required_path(ds_cfg.get("train_path"), field="dataset.train_path")
    selection_path = _resolve_required_path(
        ds_cfg.get("selection_path"), field="dataset.selection_path"
    )
    test_path = _resolve_required_path(ds_cfg.get("test_path"), field="dataset.test_path")
    accepted_path = _resolve_required_path(
        ds_cfg.get("accepted_matrix_examples_path"),
        field="dataset.accepted_matrix_examples_path",
    )
    if selection_path.resolve() == test_path.resolve():
        raise ValueError("dataset.selection_path and dataset.test_path must be different")

    sizes = list(config["eval"]["sizes"])
    per_size = int(config["eval"]["per_size"])
    eval_seed = int(config["eval"].get("seed", 0))
    diff_cfg = config.get("diffusion", {})
    step_stride = int(diff_cfg.get("step_stride", 1))
    modes: list[PredictionMode] = list(diff_cfg.get("modes", ["one_shot", "full_chain"]))

    print(f"device={device}", flush=True)
    print("loading train…", flush=True)
    train_examples = load_dataset(train_path)
    max_train = int(config.get("max_train_examples") or 0)
    if max_train > 0 and len(train_examples) > max_train:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_examples), generator=g)[:max_train]
        train_examples = [train_examples[int(i)] for i in idx]
    print(f"train_examples={len(train_examples)}", flush=True)

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

    print("loading accepted-subset routing pool...", flush=True)
    accepted_examples = load_examples_by_size(accepted_path, sizes)
    if not accepted_examples:
        raise ValueError(f"no accepted-subset examples found under {accepted_path}")
    print(f"accepted_examples={len(accepted_examples)}", flush=True)

    overlap_report = report_instance_id_overlap(train_examples, accepted_examples)
    print(
        f"accepted-subset/train overlap: {overlap_report['overlap_count']}/"
        f"{overlap_report['pool_b_size']} "
        f"({overlap_report['overlap_fraction']:.1%}) -- audited-reference-quality diagnostic, "
        "not held-out generalization",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    diffusion_persize_available_sizes: dict[str, list[int]] = {}
    predictor_train_runtime = 0.0

    # 1. no_M -- savings-guided heuristic construction, no matrix input at all.
    print("scoring no_M...", flush=True)
    started = time.perf_counter()
    no_m_test = [_no_m_matrix(example) for example in test_examples]
    no_m_accepted = [_no_m_matrix(example) for example in accepted_examples]
    rows.append(
        _build_arm_row(
            "no_M",
            test_examples=test_examples,
            test_probs=no_m_test,
            test_hats=None,
            hard_from_hats=False,
            accepted_examples=accepted_examples,
            accepted_probs=no_m_accepted,
            threshold=0.5,
            runtime_seconds=time.perf_counter() - started,
        )
    )

    # 2. random_M -- density-matched random binary matrix (isolates structure from base rate).
    print("scoring random_M...", flush=True)
    baseline_seed = int(config.get("baselines", {}).get("random_seed", seed + 1))
    started = time.perf_counter()
    random_test = [
        random_constraint_matrix(
            example.instance.n_customers,
            seed=baseline_seed + i,
            reference_matrix=example.constraint_matrix,
        )
        for i, example in enumerate(test_examples)
    ]
    random_accepted = [
        random_constraint_matrix(
            example.instance.n_customers,
            seed=baseline_seed + 1_000_000 + i,
            reference_matrix=example.constraint_matrix,
        )
        for i, example in enumerate(accepted_examples)
    ]
    rows.append(
        _build_arm_row(
            "random_M",
            test_examples=test_examples,
            test_probs=random_test,
            test_hats=None,
            hard_from_hats=False,
            accepted_examples=accepted_examples,
            accepted_probs=random_accepted,
            threshold=0.5,
            runtime_seconds=time.perf_counter() - started,
        )
    )

    # 3. ground_truth_M -- oracle. Matrix metrics are trivially perfect (a pipeline sanity
    # check, not a result); its routing cost-gap is a genuine oracle decode-quality upper bound.
    print("scoring ground_truth_M...", flush=True)
    started = time.perf_counter()
    gt_test = [example.constraint_matrix.astype(np.float64) for example in test_examples]
    gt_accepted = [example.constraint_matrix.astype(np.float64) for example in accepted_examples]
    rows.append(
        _build_arm_row(
            "ground_truth_M",
            test_examples=test_examples,
            test_probs=gt_test,
            test_hats=None,
            hard_from_hats=False,
            accepted_examples=accepted_examples,
            accepted_probs=gt_accepted,
            threshold=0.5,
            runtime_seconds=time.perf_counter() - started,
        )
    )

    # 4. supervised_M -- P2.1 MatrixPredictor, same fair recipe as eval_m_ablation.py.
    mp_cfg = config["matrix_predictor"]
    print("training supervised_M (P2.1 MatrixPredictor, x9 + soft sqrt-WBCE)...", flush=True)
    started = time.perf_counter()
    predictor = train_matrix_predictor(
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
    selection_probs = predict_matrix_predictor_probs(predictor, selection_examples, device)
    supervised_threshold = float(
        score_matrix_probabilities(selection_examples, selection_probs)["threshold"]
    )
    started = time.perf_counter()
    sup_test = predict_matrix_predictor_probs(predictor, test_examples, device)
    sup_accepted = predict_matrix_predictor_probs(predictor, accepted_examples, device)
    rows.append(
        _build_arm_row(
            "supervised_M",
            test_examples=test_examples,
            test_probs=sup_test,
            test_hats=None,
            hard_from_hats=False,
            accepted_examples=accepted_examples,
            accepted_probs=sup_accepted,
            threshold=supervised_threshold,
            runtime_seconds=time.perf_counter() - started,
        )
    )

    # 5-6. diffusion_M_pooled_{one_shot,full_chain}
    pooled_ckpt = diff_cfg.get("pooled", {}).get("checkpoint")
    if pooled_ckpt:
        pooled_path = _resolve_required_path(pooled_ckpt, field="diffusion.pooled.checkpoint")
        pooled_model, pooled_schedule = _load_diffusion_checkpoint(pooled_path, device)
        for mode in modes:
            print(f"scoring diffusion_M_pooled_{mode}...", flush=True)
            threshold = _diffusion_threshold(
                pooled_model,
                pooled_schedule,
                selection_examples,
                mode=mode,
                device=device,
                seed=eval_seed,
                step_stride=step_stride,
            )
            started = time.perf_counter()
            test_probs, test_hats = predict_matrix_batch(
                pooled_model,
                pooled_schedule,
                test_examples,
                mode=mode,
                device=device,
                seed=eval_seed,
                step_stride=step_stride,
            )
            accepted_probs, _ = predict_matrix_batch(
                pooled_model,
                pooled_schedule,
                accepted_examples,
                mode=mode,
                device=device,
                seed=eval_seed,
                step_stride=step_stride,
            )
            rows.append(
                _build_arm_row(
                    f"diffusion_M_pooled_{mode}",
                    test_examples=test_examples,
                    test_probs=test_probs,
                    test_hats=test_hats,
                    hard_from_hats=(mode == "full_chain"),
                    accepted_examples=accepted_examples,
                    accepted_probs=accepted_probs,
                    threshold=threshold,
                    runtime_seconds=time.perf_counter() - started,
                )
            )
    else:
        print(
            "skipping diffusion_M_pooled_* rows: diffusion.pooled.checkpoint not configured",
            flush=True,
        )

    # 7-8. diffusion_M_persize_{one_shot,full_chain}
    persize_cfg = diff_cfg.get("persize", {}).get("checkpoints", {}) or {}
    persize_configured = {int(size): ckpt for size, ckpt in persize_cfg.items() if ckpt}
    if persize_configured:
        persize_models: dict[int, tuple[ConstraintDenoiser, BernoulliDiffusionSchedule]] = {}
        for size, ckpt in persize_configured.items():
            ckpt_path = _resolve_required_path(ckpt, field=f"diffusion.persize.checkpoints.{size}")
            persize_models[size] = _load_diffusion_checkpoint(ckpt_path, device)
        available_sizes = sorted(persize_models)
        print(f"diffusion_M_persize available sizes: {available_sizes}", flush=True)

        selection_persize = [
            e for e in selection_examples if e.instance.n_customers in persize_models
        ]
        test_persize = [e for e in test_examples if e.instance.n_customers in persize_models]
        accepted_persize = [
            e for e in accepted_examples if e.instance.n_customers in persize_models
        ]

        for mode in modes:
            diffusion_persize_available_sizes[mode] = available_sizes
            print(f"scoring diffusion_M_persize_{mode}...", flush=True)
            threshold = _persize_threshold(
                persize_models,
                selection_persize,
                mode=mode,
                device=device,
                seed=eval_seed,
                step_stride=step_stride,
            )
            started = time.perf_counter()
            test_probs, test_hats = predict_matrix_persize(
                persize_models,
                test_persize,
                mode=mode,
                device=device,
                seed=eval_seed,
                step_stride=step_stride,
            )
            accepted_probs, _ = predict_matrix_persize(
                persize_models,
                accepted_persize,
                mode=mode,
                device=device,
                seed=eval_seed,
                step_stride=step_stride,
            )
            rows.append(
                _build_arm_row(
                    f"diffusion_M_persize_{mode}",
                    test_examples=test_persize,
                    test_probs=test_probs,
                    test_hats=test_hats,
                    hard_from_hats=(mode == "full_chain"),
                    accepted_examples=accepted_persize,
                    accepted_probs=accepted_probs,
                    threshold=threshold,
                    runtime_seconds=time.perf_counter() - started,
                )
            )
    else:
        print(
            "skipping diffusion_M_persize_* rows: no diffusion.persize.checkpoints configured",
            flush=True,
        )

    print("\n=== Phase 5 ablation table ===\n")
    _print_table(rows)

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

    provenance = {
        "train_dataset_hash": hash_dataset(train_path),
        "selection_dataset_hash": hash_dataset(selection_path),
        "test_dataset_hash": hash_dataset(test_path),
        "accepted_matrix_examples_dataset_hash": hash_dataset(accepted_path),
        "accepted_subset_train_overlap": overlap_report,
        "diffusion_persize_available_sizes": diffusion_persize_available_sizes,
        "supervised_predictor_train_runtime_seconds": predictor_train_runtime,
    }
    metrics_path.write_text(
        json.dumps({"config": config, "rows": rows, "provenance": provenance}, indent=2)
    )
    other_fieldnames = sorted({key for row in rows for key in row if key != "method"})
    fieldnames = ["method", *other_fieldnames]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    note = (
        f"Test split: {len(test_examples)} examples (per_size={per_size}, sizes={sizes}). "
        f"Accepted-subset routing pool: {len(accepted_examples)} examples. "
        f"Accepted/train overlap: {overlap_report['overlap_count']}/"
        f"{overlap_report['pool_b_size']} ({overlap_report['overlap_fraction']:.1%}) -- see "
        "module docstring."
    )
    _write_markdown_table(md_path, rows, note=note)
    for row in rows:
        tracker.log_metric_row(row)
    tracker.log_metrics({"rows": rows, **provenance})

    print(f"\nwrote {metrics_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    tracker.close()


if __name__ == "__main__":
    main()
