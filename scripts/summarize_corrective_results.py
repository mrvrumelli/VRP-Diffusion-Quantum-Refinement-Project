"""Build reviewable tables from completed corrective experiments, retaining their scope."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vrp_diffusion_quantum.eval.comparison import InstanceResult, metric_value, paired_bootstrap


def records(directory: Path) -> list[InstanceResult]:
    """Read only per-graph records, excluding summaries and timing sidecars."""
    result = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json" or path.name.endswith(".timing.json"):
            continue
        payload = json.loads(path.read_text())
        if "instance_id" in payload and "true_positive" in payload:
            result.append(InstanceResult(**payload))
    return result


def interval_text(payload: dict[str, Any]) -> str:
    return f"{payload['estimate']:.2f} [{payload['lower']:.2f}, {payload['upper']:.2f}]"


def f1_interval(payload: dict[str, Any]) -> str:
    return f"{payload['estimate']:.4f} [{payload['lower']:.4f}, {payload['upper']:.4f}]"


def main() -> None:
    root = Path("outputs/corrective_20260930")
    archive: dict[str, Any] = {"checkpoint_evaluations": {}, "learning_diagnostics": {}}
    lines = [
        "# Corrective experiment results — 2026-09-30",
        "",
        "This is execution evidence for [R1-R12](autonomous_work_priority_tasks_2026-09-30.md).",
        "See the [execution status](corrective_execution_2026-09-30.md) for outstanding gates.",
        "All learned-model results below are development diagnostics, not untouched-test claims.",
        "",
        "**Established so far.** The reverse transition is now a mixture of normalized hard-state",
        "posteriors. The legacy implementation remains available. Fixing this mathematics does",
        "not recover the paper's F1. Increasing steps to 1,000 also does not recover it.",
        "The six-customer depot counterexample establishes information loss; the matched small",
        "training experiments do not establish a dependable depot-aware performance gain.",
        "",
        "Weighted BCE is a clean-target classification surrogate, not an established equivalent",
        "of the paper's variational objective. Removing its positive weight improved calibration",
        "and generated F1 in both small runs, but worsened decoded cost. BatchNorm "
        "showed substantial",
        "seed and mode sensitivity; batch-order shuffling did not reliably cure it. These results",
        "do not quantify the corresponding contributions in the full paper-scale model.",
        "",
        "The external Figure 8 targets remain 0.823 at 50 steps and 0.875 at 1,000; the paper's",
        "averaging/size mix and exact inference reconstruction remain unresolved. "
        "[Paper, Figure 8](https://arxiv.org/pdf/2603.07568v1#page=13).",
        "No new result here establishes paper reproduction or author-code equivalence.",
        "",
        "![Scaling and OOD route evidence](evidence/corrective_20260930/route_evidence.png)",
        "",
        "**Why the F1 gap remains.** There is no demonstrated single cause. Historical",
        "maxima mix different measurements: the roughly 0.6695 result was noisy-time",
        "denoising with threshold selection; the roughly 0.6222 robust result was N20-only",
        "generation. Neither is a matched comparison with Figure 8. On the larger shared",
        "development panel, the corrected full checkpoint scores 0.4682 pooled F1 versus",
        "0.5043 when equally averaging the three sizes. Aggregation matters, but that",
        "difference and the sampler correction do not explain a target near 0.823.",
        "",
        "The remaining evidence points to model/training reconstruction as an unresolved",
        "problem: customer-only inputs discard depot information, weighted classification",
        "does not ensure calibrated reverse probabilities, and BatchNorm can behave very",
        "differently between training and inference. Small controlled runs show these are",
        "testable concerns, but do not assign a percentage of the full model's deficit to",
        "any one factor. The repaired source curve also does not justify treating more",
        "labels as the established remedy. Exact author metric, input and objective details",
        "remain necessary to separate reconstruction differences from training/data limits.",
        "",
        "## Common-panel checkpoint evaluations",
        "",
        "Eight graphs per size; shared source-audited development panel, batch 1, float32,",
        "fixed threshold 0.5, graph-ID-derived sampling seeds. F1 is pooled positive-class F1.",
        "Route gap is the mean of instance-relative gaps; ratio-of-total-cost gap is separate.",
        "Timing includes inference and clustering decode, excluding model loading.",
        "",
        "| Model / sampler / steps / seed | F1 | Mean gap % | Ratio gap % | Graphs |",
        "|---|---:|---:|---:|---:|",
    ]
    for path in sorted(root.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        archive["checkpoint_evaluations"][path.parent.name] = summary
        if "f1" not in summary:
            continue
        lines.append(
            f"| {path.parent.name} | {summary['f1']['estimate']:.4f} | "
            f"{summary['mean_instance_gap_percent']['estimate']:.2f} | "
            f"{summary['ratio_total_gap_percent']['estimate']:.2f} | "
            f"{summary['f1']['num_observations']} |"
        )
    paired_path = root / "paired_comparisons.json"
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        archive["paired_comparisons"] = paired
        lines += [
            "",
            "## Paired changes",
            "",
            "Candidate minus baseline; negative route-gap differences favor the candidate.",
            "Intervals resample graphs within size (10,000 draws); training seeds "
            "are not extra graphs.",
            "These intervals are exploratory and do not correct for searching many candidates.",
            "",
            "| Comparison | Mean-gap difference, 95% CI (pp) |",
            "|---|---:|",
        ]
        for name, comparison in paired.items():
            lines.append(f"| {name} | {interval_text(comparison['mean_instance_gap_percent'])} |")
    confirmation = Path("outputs/corrective_confirmation_20260930")
    archive["expanded_development"] = {}
    lines += [
        "",
        "## Expanded development comparison",
        "",
        "The panel was fixed before these evaluations: 24 graphs per size (72 total),",
        "including the original eight per size. It is larger development evidence, not an",
        "independent test. Corrected 50-step sampler, seed 0; all other settings unchanged.",
        "Intervals below resample graphs, and do not represent training-seed uncertainty.",
        "",
        "| Model | Graphs | Pooled F1, 95% CI | Equal-size F1 | Mean gap %, 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for path in sorted(confirmation.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        archive["expanded_development"][path.parent.name] = summary
        lines.append(
            f"| {summary['checkpoint']} | {summary['f1']['num_observations']} | "
            f"{f1_interval(summary['f1'])} | {summary['equal_size_mean_f1']:.4f} | "
            f"{interval_text(summary['mean_instance_gap_percent'])} |"
        )
    archive["expanded_paired"] = {}
    lines += [
        "",
        "Paper checkpoints on identical size-specific subsets (24 graphs per cell):",
        "",
        "| Checkpoint | Size | F1 | Mean gap % |",
        "|---|---:|---:|---:|",
    ]
    for name in ("full", "pilot", "unfiltered", "n20only"):
        rows = records(confirmation / f"{name}_posterior_mixture_v2_steps50_seed0")
        for size in (20, 50, 100):
            subset = [row for row in rows if row.n_customers == size]
            if len(subset) != 24:
                continue
            lines.append(
                f"| {name} | {size} | {metric_value(subset, 'f1'):.4f} | "
                f"{metric_value(subset, 'mean_instance_gap_percent'):.2f} |"
            )
    lines += [
        "",
        "| Task 5 candidate minus champion | F1 difference, 95% CI | "
        "Mean-gap difference, 95% CI (pp) |",
        "|---|---:|---:|",
    ]
    for size in (20, 50, 100):
        baseline = records(confirmation / f"champion_n{size}_posterior_mixture_v2_steps50_seed0")
        candidate = records(confirmation / f"candidate_n{size}_posterior_mixture_v2_steps50_seed0")
        if len(baseline) != 24 or len(candidate) != 24:
            continue
        comparisons = {
            metric: asdict(
                paired_bootstrap(candidate, baseline, metric=metric, num_resamples=10000)
            )
            for metric in ("f1", "mean_instance_gap_percent", "ratio_total_gap_percent")
        }
        archive["expanded_paired"][str(size)] = comparisons
        lines.append(
            f"| N{size} | {f1_interval(comparisons['f1'])} | "
            f"{interval_text(comparisons['mean_instance_gap_percent'])} |"
        )
    lines += [
        "",
        "## Learning, depot, objective and normalization diagnostics",
        "",
        "One-source sanity: 1,200 updates; other arms: 32 sources per size, 600 updates,",
        "two seeds, width 64, three denoiser/GAT layers, jointly trained GAT, no augmentation.",
        "Final-update checkpoints, with identical split/budget within matched pairs. This is a",
        "reduced-scale diagnostic, not a recreation of frozen-GAT paper training. Train metrics",
        "in the raw summaries use the first eight training examples (N20); "
        "validation pools all sizes.",
        "",
        "| Arm / training seed | Generated validation F1 | Mean route gap % | "
        "High-noise calibration error |",
        "|---|---:|---:|---:|",
    ]
    for path in sorted(Path("outputs/corrective_learning_20260930").glob("*/summary.json")):
        summary = json.loads(path.read_text())
        archive["learning_diagnostics"][path.parent.name] = summary
        metrics = summary["generated_validation"]
        lines.append(
            f"| {path.parent.name} | {metrics['sample_f1']:.4f} | "
            f"{metrics['route_mean_cost_gap_percent']:.2f} | "
            f"{summary['noisy_validation']['999']['calibration_error']:.4f} |"
        )
    lines += [
        "",
        "The tiny run reached generated training F1=1.0 and exact oracle reconstruction at",
        "1, 50 and 1,000 steps. This validates basic fit/endpoints, not generalization.",
        "",
        "Single-size N100 normalization controls use the same 32 N100 sources, eight",
        "development graphs, two seeds and 600 updates; they isolate normalization within",
        "one size rather than conflating it with traversal of mixed-size batches.",
        "",
        "| N100-only arm / seed | Generated F1 | Mean route gap % |",
        "|---|---:|---:|",
    ]
    archive["single_size_normalization"] = {}
    for path in sorted(
        Path("outputs/corrective_normalization_n100_20260930").glob("*/summary.json")
    ):
        summary = json.loads(path.read_text())
        archive["single_size_normalization"][path.parent.name] = summary
        metrics = summary["generated_validation"]
        lines.append(
            f"| {path.parent.name} | {metrics['sample_f1']:.4f} | "
            f"{metrics['route_mean_cost_gap_percent']:.2f} |"
        )
    diagnostic_root = Path("outputs/corrective_diagnostic_evaluation_20260930")
    archive["diagnostic_paired_graph_scoring"] = {}
    lines += [
        "",
        "Additional graph-level rescoring uses seed 9001 derived from graph IDs, unlike",
        "the position-derived seeds in the original small-run summaries above. The paired",
        "effects below compare arms within this additional common protocol. They remain",
        "exploratory development evidence; eight graphs per size cannot settle full-scale effects.",
        "",
        "| Variant minus matched LayerNorm baseline / seed | F1 difference, 95% CI | "
        "Mean-gap difference, 95% CI (pp) |",
        "|---|---:|---:|",
    ]
    for prefix, arms in (
        ("mixed", ("depot", "unweighted", "bn_sorted", "bn_shuffled")),
        ("single", ("bn_shuffled",)),
    ):
        for arm in arms:
            for seed in (4331, 4332):
                suffix = "_n100" if prefix == "single" else ""
                tag = "_posterior_mixture_v2_steps50_seed9001"
                candidate = records(diagnostic_root / f"{prefix}_{arm}_seed{seed}{suffix}{tag}")
                baseline = records(diagnostic_root / f"{prefix}_baseline_seed{seed}{suffix}{tag}")
                expected = 8 if prefix == "single" else 24
                if len(candidate) != expected or len(baseline) != expected:
                    continue
                comparisons = {
                    metric: asdict(paired_bootstrap(candidate, baseline, metric=metric))
                    for metric in ("f1", "mean_instance_gap_percent")
                }
                key = f"{prefix}_{arm}_seed{seed}"
                archive["diagnostic_paired_graph_scoring"][key] = comparisons
                lines.append(
                    f"| {key} | {f1_interval(comparisons['f1'])} | "
                    f"{interval_text(comparisons['mean_instance_gap_percent'])} |"
                )
    lines += [
        "",
        "`bn_sorted` versus the shuffled LayerNorm baseline changes both normalization and",
        "order. The direct BatchNorm-only order contrast below isolates batch traversal.",
        "",
        "| Shuffled minus sorted BatchNorm / seed | F1 difference, 95% CI | "
        "Mean-gap difference, 95% CI (pp) |",
        "|---|---:|---:|",
    ]
    for seed in (4331, 4332):
        tag = f"seed{seed}_posterior_mixture_v2_steps50_seed9001"
        candidate = records(diagnostic_root / f"mixed_bn_shuffled_{tag}")
        baseline = records(diagnostic_root / f"mixed_bn_sorted_{tag}")
        if len(candidate) != 24 or len(baseline) != 24:
            continue
        comparisons = {
            metric: asdict(paired_bootstrap(candidate, baseline, metric=metric))
            for metric in ("f1", "mean_instance_gap_percent")
        }
        archive["diagnostic_paired_graph_scoring"][f"batch_order_seed{seed}"] = comparisons
        lines.append(
            f"| {seed} | {f1_interval(comparisons['f1'])} | "
            f"{interval_text(comparisons['mean_instance_gap_percent'])} |"
        )
    lines += [
        "",
        "The N100-only controls are more stable and their original route-gap point estimates",
        "favor BatchNorm in both seeds. The graph-ID-seeded rescoring reverses that direction",
        "for seed 4332, so the route gain is not dependable across sampling protocols.",
        "The improved stability supports investigating size interactions, but",
        "does not prove that ordering alone explains the full-model deficit: single-size",
        "training also changes exposure per size under the fixed total update budget.",
        "No normalization replacement is selected from these small diagnostics.",
    ]
    for name in ("batchnorm_mode_probe", "batchnorm_statistics", "batchnorm_all_noise_probe"):
        path = Path("outputs/corrective_learning_20260930") / f"{name}.json"
        if path.exists():
            archive[name] = json.loads(path.read_text())
    lines += [
        "",
        "The 48-case CPU clone probe covers four BatchNorm checkpoints, three sizes and",
        "four noise levels. Mean absolute train/eval probability differences reach 0.754",
        "at batch size 1. The original parameters and buffers remain exactly unchanged.",
        "This shows a mode mismatch in these diagnostics, not its share of the paper F1 gap.",
    ]
    lines += [
        "",
        "## Held-out post-hoc calibration",
        "",
        "Fit a size-specific logit intercept on 24 separate calibration graphs, excluding",
        "both the 24-graph diagnostic panel and the expanded 72-graph panel. The fit minimized",
        "unweighted BCE across four fixed noise levels. This is an empirical calibration test,",
        "not an assertion that one offset exactly undoes varying per-batch class weights.",
        "",
        "| Sampling seed | Shift inside reverse chain: F1 | Mean route gap % | "
        "Final-classification-only F1 |",
        "|---|---:|---:|---:|",
    ]
    calibration_root = Path("outputs/corrective_calibration_20260930")
    for seed in (0, 1):
        path = (
            calibration_root
            / f"full_shift_inside_posterior_mixture_v2_steps50_seed{seed}/summary.json"
        )
        if path.exists():
            summary = json.loads(path.read_text())
            final = json.loads((calibration_root / f"final_only_seed{seed}.json").read_text())
            archive[f"calibration_seed{seed}"] = {"inside": summary, "final_only": final}
            lines.append(
                f"| {seed} | {summary['f1']['estimate']:.4f} | "
                f"{summary['mean_instance_gap_percent']['estimate']:.2f} | "
                f"{final['f1']:.4f} |"
            )
    calibration_pairs = calibration_root / "paired_comparisons.json"
    if calibration_pairs.exists():
        paired_calibration = json.loads(calibration_pairs.read_text())
        archive["calibration_paired"] = paired_calibration
        lines += [
            "",
            "| Seed | Inside-chain shift F1 difference, 95% CI | "
            "Mean-gap difference, 95% CI (pp) |",
            "|---|---:|---:|",
        ]
        for seed, comparisons in paired_calibration.items():
            lines.append(
                f"| {seed} | {f1_interval(comparisons['f1'])} | "
                f"{interval_text(comparisons['mean_instance_gap_percent'])} |"
            )
    lines += [
        "",
        "Reject this particular calibration recipe: it worsened generation and route utility.",
        "",
        "## Identical-panel classical and policy comparison",
        "",
        "PyVRP/OR-Tools receive 1 or 3 seconds of search time per instance; actual elapsed",
        "runtime includes solver setup. All 24 graphs are held out of the compared training sets.",
        "Policy rows without `_e2e` isolate decoding using the exact cached champion prior",
        "(corrected sampler, 50 steps, seed 0), and their timing excludes prior generation.",
        "Rows with `_e2e` regenerate the prior and include it in timing. Model "
        "loading is excluded.",
        "The hardware differs (GPU learned methods versus CPU solvers); this is a measured",
        "quality/time comparison, not a claim of identical compute resources.",
        "`matched_policy16` solver rows cap search at each graph's measured K=16 policy",
        "end-to-end time. Solver setup is extra; actual elapsed is also recorded. All",
        "same-size comparisons below use exactly the same graph IDs and reference costs.",
        "",
        "| Method | Graphs | Mean gap % | Ratio gap % | Mean elapsed seconds | Feasible fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in archive["checkpoint_evaluations"].items():
        if "f1" in summary:
            continue
        count = summary["num_instances"]
        lines.append(
            f"| {name} | {count} | "
            f"{summary['mean_instance_gap_percent']['estimate']:.2f} | "
            f"{summary['ratio_total_gap_percent']['estimate']:.2f} | "
            f"{summary['runtime_seconds'] / count:.3f} | "
            f"{summary['feasibility_rate']:.3f} |"
        )
    lines += [
        "",
        "| Size | Method | Mean gap % | Ratio gap % | Mean elapsed seconds |",
        "|---|---|---:|---:|---:|",
    ]
    archive["matched_methods_by_size"] = {}
    for size in (20, 50, 100):
        methods = [
            f"champion_n{size}_posterior_mixture_v2_steps50_seed0",
            f"policy_n{size}_starts16_e2e",
            "pyvrp_seconds1",
            "pyvrp_seconds3",
            "ortools_seconds1",
            "ortools_seconds3",
            "pyvrp_matched_policy16",
            "ortools_matched_policy16",
        ]
        for method in methods:
            rows = [row for row in records(root / method) if row.n_customers == size]
            if len(rows) != 8:
                continue
            result = {
                metric: asdict(paired_bootstrap(rows, metric=metric))
                for metric in ("mean_instance_gap_percent", "ratio_total_gap_percent")
            }
            result["mean_runtime_seconds"] = sum(row.runtime_seconds for row in rows) / len(rows)
            result["feasibility_rate"] = sum(row.feasible for row in rows) / len(rows)
            archive["matched_methods_by_size"][f"n{size}_{method}"] = result
            lines.append(
                f"| {size} | {method} | "
                f"{result['mean_instance_gap_percent']['estimate']:.2f} | "
                f"{result['ratio_total_gap_percent']['estimate']:.2f} | "
                f"{result['mean_runtime_seconds']:.3f} |"
            )
    lines += [
        "",
        "## CVRPLIB expansion and additional defect",
        "",
        "The existing evaluator ignored the declared fleet count. The first expanded run",
        "returned B-n51-k7 cost 1,016 with eight vehicles against a seven-vehicle reference of",
        "1,032. That comparison is invalid. The corrected evaluator caps the declared fleet;",
        "the superseded run is retained only as diagnostic evidence.",
        "",
        "Corrected PyVRP runs use a 3-second search limit, seed 42 with per-instance derivation,",
        "and integer `EUC_2D` nearest rounding. Reference solutions come from the",
        "[official CVRPLIB index](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances/1).",
        "B-n51-k7's solution download was unavailable; its reference is the instance comment,",
        "cross-checked against the index. Other references use the downloaded `.sol` costs.",
        "",
        "| Instance | Vehicles used / cap | Cost | Reference | Gap % |",
        "|---|---:|---:|---:|---:|",
    ]
    csv_path = root / "cvrplib_fleet_fixed.csv"
    if csv_path.exists():
        with csv_path.open() as handle:
            archive["cvrplib"] = list(csv.DictReader(handle))
        for row in archive["cvrplib"]:
            lines.append(
                f"| {row['instance']} | {row['number_of_vehicles']} / "
                f"{row['declared_vehicles']} | {row['cost']} | "
                f"{row['reference_cost']} | {float(row['gap']):.3f} |"
            )
    ood_root = Path("outputs/corrective_ood_20260930")
    if (ood_root / "panel.json").exists():
        panel = json.loads((ood_root / "panel.json").read_text())
        lines += [
            "",
            "## R/C/RC breakdown",
            "",
            "Eight randomly selected stable-label graphs per cell, seed 93829; corrected sampler,",
            "50 steps, sampling seed 0. Coverage shows accepted labels out of the original",
            "1,000 graphs per cell. Conclusions apply to these checkpoints and selected labels.",
            "",
            "| Cell | Stable-label coverage | Champion F1 | Merged F1 | Champion "
            "gap % | Merged gap % | Paired gap difference, 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        archive["ood_cells"] = {}
        for regime in ("r", "c", "rc"):
            for size in (20, 50, 100):
                ids = {
                    entry["instance_id"]
                    for entry in panel["examples"]
                    if entry["regime"] == regime and entry["n_customers"] == size
                }
                baseline = [
                    row
                    for row in records(
                        ood_root / f"champion_n{size}_posterior_mixture_v2_steps50_seed0"
                    )
                    if row.instance_id in ids
                ]
                candidate = [
                    row
                    for row in records(
                        ood_root / f"merged_n{size}_posterior_mixture_v2_steps50_seed0"
                    )
                    if row.instance_id in ids
                ]
                if len(baseline) != 8 or len(candidate) != 8:
                    continue
                interval = asdict(
                    paired_bootstrap(candidate, baseline, metric="mean_instance_gap_percent")
                )
                cell = f"{regime}_n{size}"
                row = {
                    "champion_f1": metric_value(baseline, "f1"),
                    "merged_f1": metric_value(candidate, "f1"),
                    "champion_gap": metric_value(baseline, "mean_instance_gap_percent"),
                    "merged_gap": metric_value(candidate, "mean_instance_gap_percent"),
                    "paired_gap": interval,
                    "coverage": panel["coverage"][cell],
                    "num_graphs": len(baseline),
                    "champion_f1_interval": asdict(paired_bootstrap(baseline, metric="f1")),
                    "merged_f1_interval": asdict(paired_bootstrap(candidate, metric="f1")),
                    "champion_feasibility": sum(item.feasible for item in baseline) / len(baseline),
                    "merged_feasibility": sum(item.feasible for item in candidate) / len(candidate),
                }
                archive["ood_cells"][cell] = row
                lines.append(
                    f"| {cell} | {row['coverage']['accepted']}/1000 | "
                    f"{row['champion_f1']:.4f} | {row['merged_f1']:.4f} | "
                    f"{row['champion_gap']:.2f} | {row['merged_gap']:.2f} | "
                    f"{interval_text(interval)} |"
                )
        lines += [
            "",
            "| Cell | Champion F1, 95% CI | Merged F1, 95% CI | Champion / merged feasible |",
            "|---|---:|---:|---:|",
        ]
        for cell, row in archive["ood_cells"].items():
            lines.append(
                f"| {cell} | {f1_interval(row['champion_f1_interval'])} | "
                f"{f1_interval(row['merged_f1_interval'])} | "
                f"{row['champion_feasibility']:.3f} / {row['merged_feasibility']:.3f} |"
            )
    curve_decision = Path("outputs/corrective_curve_20260930/decision.json")
    if curve_decision.exists():
        decision = json.loads(curve_decision.read_text())
        archive["curve_decision"] = decision
        lines += [
            "",
            "## Source-count curve decision",
            "",
            "Four runs under the [predeclared five-epoch "
            "protocol](corrective_curve_protocol_2026-09-30.md).",
            "",
            "| Training seed | 1,000 minus 500 source mean-gap difference, 95% CI | Passes rule |",
            "|---|---:|---|",
        ]
        for seed, result in decision["seed_results"].items():
            lines.append(
                f"| {seed} | {interval_text(result['paired_gap_difference'])} | "
                f"{result['passes']} |"
            )
        lines += ["", str(decision["action"]), ""]
        lines += [
            "| Sources / seed | F1 | Mean gap % | Optimizer steps | Recorded loop seconds |",
            "|---|---:|---:|---:|---:|",
        ]
        archive["curve_runs"] = {}
        for count in (500, 1000):
            for seed in (4331, 4332):
                curve_root = curve_decision.parent
                summary = json.loads(
                    (
                        curve_root
                        / (
                            f"curve{count}_s{seed}_n100_posterior_mixture_v2_steps50_seed9001/summary.json"
                        )
                    ).read_text()
                )
                csv_paths = list(
                    curve_root.glob(f"corrective_curve_n100_{count}_s{seed}_*/summary.csv")
                )
                with csv_paths[0].open() as handle:
                    training = list(csv.DictReader(handle))[-1]
                archive["curve_runs"][f"{count}_{seed}"] = {
                    "evaluation": summary,
                    "training": training,
                }
                lines.append(
                    f"| {count} / {seed} | {summary['f1']['estimate']:.4f} | "
                    f"{summary['mean_instance_gap_percent']['estimate']:.2f} | "
                    f"{training['total_optimizer_steps']} | "
                    f"{float(training['total_runtime_seconds']):.2f} |"
                )
        lines += [
            "",
            "Recorded loop time includes training and noisy validation through the final epoch,",
            "but excludes the final full-chain validation and subsequent standalone rescoring.",
            "Fixed epochs give the larger source pool more updates. The reversal is specific",
            "to this five-epoch recipe; it does not prove that more data generally harms learning.",
        ]
    # Compare seed outputs directly rather than infer identity from equal aggregate metrics.
    seed_identity = {}
    for name in [f"{kind}_n{size}" for kind in ("champion", "candidate") for size in (20, 50, 100)]:
        left = root / f"{name}_posterior_mixture_v2_steps50_seed0"
        right = root / f"{name}_posterior_mixture_v2_steps50_seed1"
        files = sorted(left.glob("*.npz"))
        if not files or not all((right / path.name).exists() for path in files):
            continue
        exact = sum(
            np.array_equal(np.load(path)["probability"], np.load(right / path.name)["probability"])
            for path in files
        )
        hard = sum(
            np.array_equal(
                np.load(path)["probability"] >= 0.5,
                np.load(right / path.name)["probability"] >= 0.5,
            )
            for path in files
        )
        seed_identity[name] = {
            "graphs": len(files),
            "identical_probabilities": exact,
            "identical_hard_matrices": hard,
        }
    archive["seed_identity_checks"] = seed_identity
    lines += [
        "",
        "Direct seed checks found no exactly identical probability arrays across seeds.",
        "Some hard matrices were identical. Equal aggregate F1/gap therefore does not",
        "establish identical outputs; the archive records counts for each checkpoint.",
    ]
    lines += [
        "",
        "## Remaining interpretation limits",
        "",
        "The 96 reserved test sources remain unscored. Their original references are not yet",
        "a strong final benchmark. No development comparison alone freezes a new champion.",
        "The classical/paper freeze and downstream quantum prerequisites remain unsatisfied.",
        "The [candidate manifest](evidence/corrective_20260930/candidate_baseline.json) records",
        "the defer-freeze decision, checkpoint identities and unresolved author assumptions.",
        "Raw outputs contain checkpoint/panel hashes, graph predictions and both gap definitions.",
        "This report includes only completed artifacts; absent tables are pending, "
        "not negative results.",
        "",
    ]
    Path("docs/corrective_results_2026-09-30.md").write_text("\n".join(lines), encoding="utf-8")
    destination = Path("docs/evidence/corrective_20260930")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(archive, indent=2) + "\n")


if __name__ == "__main__":
    main()
