"""Compare classical OR baselines under matched wall-clock budgets.

The default run uses a small deterministic subset of the committed strong-reference
examples:

    outputs/label_audit/s7799_strong_reference/accepted_matrix_examples

For every selected instance, each solver receives the same configured time budget.
The report records actual runtime, feasibility, vehicle count, mean cost, and gap
to the strong-reference label cost.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from vrp_diffusion_quantum.data.dataset import IndexedJSONDataset
from vrp_diffusion_quantum.data.generate_cvrp import CVRPInstance as SolverCVRPInstance
from vrp_diffusion_quantum.data.solve_cvrp import CVRPSolution, SolverName, solve_instance
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.utils.experiment import git_commit_hash, hash_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    ROOT / "outputs" / "label_audit" / "s7799_strong_reference" / "accepted_matrix_examples"
)
DEFAULT_OUTPUT = ROOT / "eval" / "or_baseline_comparison.csv"
DEFAULT_REPORT = ROOT / "docs" / "or_baseline_comparison.md"
DEFAULT_SOLVERS: tuple[SolverName, ...] = ("pyvrp", "ortools")
DEFAULT_TIME_BUDGETS = (0.25, 0.75)
DEFAULT_SIZES = (20, 50, 100)

_SUMMARY_COLS = (
    "baseline",
    "time_budget_seconds",
    "size",
    "instances",
    "success_count",
    "feasible_rate",
    "mean_gap_to_reference_percent",
    "mean_gap_to_best_observed_percent",
    "mean_cost",
    "mean_runtime_seconds",
    "mean_runtime_budget_ratio",
    "mean_number_of_vehicles",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--instances-per-size", type=int, default=2)
    parser.add_argument(
        "--time-budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_TIME_BUDGETS),
        help="Matched wall-clock seconds per solver and instance.",
    )
    parser.add_argument(
        "--solvers",
        choices=DEFAULT_SOLVERS,
        nargs="+",
        default=list(DEFAULT_SOLVERS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def solver_label(solver: SolverName) -> str:
    if solver == "pyvrp":
        return "PyVRP HGS"
    if solver == "ortools":
        return "OR-Tools GLS"
    raise ValueError(f"unknown solver: {solver!r}")


def _selected_hash(examples: Sequence[CVRPExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        instance = example.instance
        digest.update(instance.instance_id.encode("utf-8"))
        digest.update(f"|n={instance.n_customers}|cap={instance.capacity:.12g}".encode())
        digest.update(np.ascontiguousarray(instance.coords, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(instance.demands, dtype=np.float64).tobytes())
        digest.update(f"|ref={example.solution.cost:.12g}".encode())
    return digest.hexdigest()


def load_selected_examples(
    data_dir: str | Path,
    *,
    sizes: Sequence[int],
    instances_per_size: int,
    seed: int,
) -> list[CVRPExample]:
    """Load a deterministic, size-balanced subset from a path-indexed JSON dataset."""
    if instances_per_size < 1:
        raise ValueError(f"instances_per_size must be >= 1, got {instances_per_size}")
    dataset = IndexedJSONDataset(data_dir, sizes=sizes)
    if not dataset:
        raise ValueError(f"no examples found under {data_dir}")

    rng = np.random.default_rng(seed)
    by_size = dataset.indices_by_size()
    selected: list[CVRPExample] = []
    for size in sizes:
        candidates = by_size.get(int(size), [])
        if len(candidates) < instances_per_size:
            raise ValueError(
                f"size {size} needs {instances_per_size} examples, "
                f"found {len(candidates)} in {data_dir}"
            )
        chosen = rng.choice(candidates, size=instances_per_size, replace=False).tolist()
        selected.extend(dataset[int(index)] for index in chosen)
    return selected


def to_solver_instance(example: CVRPExample) -> SolverCVRPInstance:
    """Convert a labelled-example instance into the solver wrapper's instance type."""
    return SolverCVRPInstance(
        coords=example.instance.coords.astype(np.float64, copy=True),
        demands=example.instance.demands.astype(np.float64, copy=True),
        capacity=float(example.instance.capacity),
        depot_index=int(example.instance.depot_index),
    )


def _run_seed(
    *,
    seed: int,
    solver: SolverName,
    time_budget: float,
    size: int,
    selected_index: int,
) -> int:
    solver_offset = DEFAULT_SOLVERS.index(solver) if solver in DEFAULT_SOLVERS else 99
    budget_ms = round(float(time_budget) * 1000)
    return int(
        np.random.SeedSequence(
            [seed, solver_offset, int(size), int(selected_index), int(budget_ms)]
        ).generate_state(1)[0]
    )


def _percent_gap(cost: float | None, reference_cost: float | None) -> float | None:
    if cost is None or reference_cost is None or reference_cost <= 0:
        return None
    if not math.isfinite(cost):
        return None
    return 100.0 * (cost - reference_cost) / reference_cost


def evaluate_one(
    example: CVRPExample,
    *,
    selected_index: int,
    solver: SolverName,
    time_budget: float,
    seed: int,
) -> dict[str, Any]:
    """Solve one example with one OR baseline and return a detailed metric row."""
    if time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")

    instance_seed = _run_seed(
        seed=seed,
        solver=solver,
        time_budget=time_budget,
        size=example.instance.n_customers,
        selected_index=selected_index,
    )
    started = time.perf_counter()
    solution: CVRPSolution | None = None
    error = ""
    try:
        solution = solve_instance(
            to_solver_instance(example),
            solver=solver,
            time_limit=float(time_budget),
            seed=instance_seed,
            instance_id=selected_index,
        )
        runtime = float(solution.runtime_seconds)
        status = "ok"
    except Exception as exc:  # pragma: no cover - exercised with mocked failures.
        runtime = time.perf_counter() - started
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    cost = None if solution is None else float(solution.cost)
    reference_cost = float(example.solution.cost)
    return {
        "selected_index": selected_index,
        "instance_id": example.instance.instance_id,
        "size": int(example.instance.n_customers),
        "baseline": solver_label(solver),
        "solver": solver,
        "time_budget_seconds": float(time_budget),
        "status": status,
        "cost": cost,
        "reference_cost": reference_cost,
        "gap_to_reference_percent": _percent_gap(cost, reference_cost),
        "runtime_seconds": runtime,
        "runtime_budget_ratio": runtime / float(time_budget),
        "feasibility": bool(solution.feasible) if solution is not None else False,
        "number_of_vehicles": None if solution is None else int(solution.num_vehicles),
        "reference_number_of_vehicles": int(example.solution.num_vehicles),
        "seed": instance_seed,
        "reference_solver": example.solution.solver_name,
        "reference_time_budget": example.solution.time_budget,
        "reference_runtime_seconds": float(example.solution.runtime_seconds),
        "error": error,
    }


def _add_best_observed_gaps(rows: list[dict[str, Any]]) -> None:
    best_by_budget_instance: dict[tuple[float, str], float] = {}
    for row in rows:
        cost = row["cost"]
        if cost is None or not math.isfinite(float(cost)):
            continue
        key = (float(row["time_budget_seconds"]), str(row["instance_id"]))
        current = best_by_budget_instance.get(key)
        best_by_budget_instance[key] = float(cost) if current is None else min(current, float(cost))

    for row in rows:
        key = (float(row["time_budget_seconds"]), str(row["instance_id"]))
        best = best_by_budget_instance.get(key)
        row["best_observed_cost"] = best
        row["gap_to_best_observed_percent"] = _percent_gap(row["cost"], best)


def evaluate_baselines(
    examples: Sequence[CVRPExample],
    *,
    solvers: Sequence[SolverName],
    time_budgets: Sequence[float],
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate all solvers on the same selected examples and matched budgets."""
    if not examples:
        raise ValueError("cannot evaluate an empty example selection")
    if not solvers:
        raise ValueError("at least one solver is required")
    if not time_budgets:
        raise ValueError("at least one time budget is required")

    rows: list[dict[str, Any]] = []
    for budget in time_budgets:
        if budget <= 0:
            raise ValueError(f"time budgets must be positive, got {budget}")
        for selected_index, example in enumerate(examples):
            for solver in solvers:
                rows.append(
                    evaluate_one(
                        example,
                        selected_index=selected_index,
                        solver=solver,
                        time_budget=float(budget),
                        seed=seed,
                    )
                )
    _add_best_observed_gaps(rows)
    return rows


def _mean_optional(values: Sequence[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else fmean(finite)


def _summarize_group(rows: list[dict[str, Any]], *, size: int | str) -> dict[str, Any]:
    budget = float(rows[0]["time_budget_seconds"])
    return {
        "baseline": rows[0]["baseline"],
        "solver": rows[0]["solver"],
        "time_budget_seconds": budget,
        "size": size,
        "instances": len(rows),
        "success_count": sum(row["status"] == "ok" for row in rows),
        "feasible_rate": fmean(1.0 if row["feasibility"] else 0.0 for row in rows),
        "mean_gap_to_reference_percent": _mean_optional(
            [row["gap_to_reference_percent"] for row in rows]
        ),
        "mean_gap_to_best_observed_percent": _mean_optional(
            [row["gap_to_best_observed_percent"] for row in rows]
        ),
        "mean_cost": _mean_optional([row["cost"] for row in rows]),
        "mean_runtime_seconds": _mean_optional([row["runtime_seconds"] for row in rows]),
        "mean_runtime_budget_ratio": _mean_optional([row["runtime_budget_ratio"] for row in rows]),
        "mean_number_of_vehicles": _mean_optional([row["number_of_vehicles"] for row in rows]),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate detailed rows by solver, time budget, and size, including overall rows."""
    if not rows:
        raise ValueError("cannot summarize an empty OR baseline comparison")
    summary: list[dict[str, Any]] = []
    keys = sorted(
        {(str(row["solver"]), float(row["time_budget_seconds"])) for row in rows},
        key=lambda item: (item[1], item[0]),
    )
    for solver, budget in keys:
        solver_budget_rows = [
            row
            for row in rows
            if str(row["solver"]) == solver and float(row["time_budget_seconds"]) == budget
        ]
        summary.append(_summarize_group(solver_budget_rows, size="all"))
        for size in sorted({int(row["size"]) for row in solver_budget_rows}):
            size_rows = [row for row in solver_budget_rows if int(row["size"]) == size]
            summary.append(_summarize_group(size_rows, size=size))
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty OR baseline comparison")
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


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _write_markdown_report(
    path: Path,
    *,
    summary_rows: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
    detail_path: Path,
    summary_path: Path,
) -> None:
    overall_rows = [row for row in summary_rows if row["size"] == "all"]
    best = min(
        overall_rows,
        key=lambda row: (
            math.inf
            if row["mean_gap_to_reference_percent"] is None
            else float(row["mean_gap_to_reference_percent"])
        ),
    )
    lines = [
        "# OR Baseline Comparison",
        "",
        "This report compares classical OR baselines under matched wall-clock budgets. "
        "Each selected instance is solved by every baseline at the same per-instance "
        "time limit; quality is measured against the committed strong-reference label cost.",
        "",
        f"- Data: `{provenance['data_dir']}`",
        f"- Dataset hash: `{provenance['dataset_hash']}`",
        f"- Selected subset hash: `{provenance['selected_subset_hash']}`",
        f"- Seed: `{provenance['seed']}`",
        f"- Time budgets: `{provenance['time_budgets_seconds']}` seconds per instance",
        f"- Selected counts by size: `{provenance['selected_counts_by_size']}`",
        f"- Git commit: `{provenance['git_commit']}`",
        f"- Detailed CSV: `{detail_path.relative_to(ROOT)}`",
        f"- Summary JSON: `{summary_path.relative_to(ROOT)}`",
        "",
        "## OR Comparison Table",
        "",
        "| " + " | ".join(_SUMMARY_COLS) + " |",
        "| " + " | ".join("---" for _ in _SUMMARY_COLS) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(_format_cell(row.get(col)) for col in _SUMMARY_COLS) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- Best overall quality in this run: {best['baseline']} at "
                f"{best['time_budget_seconds']:.2f}s with mean reference gap "
                f"{best['mean_gap_to_reference_percent']:.2f}%."
            ),
            "- Runtime fairness is enforced by matching the configured budget per solver "
            "and instance.",
            "- Actual runtime includes instance conversion and solver setup overhead, "
            "so it can exceed the configured solver time limit on short budgets.",
            "",
            "## Caveat",
            "",
            "This is a smoke-scale OR comparison over a deterministic subset, not a full "
            "benchmark sweep. It is meant to make runtime and solution-quality reporting "
            "reproducible inside this checkout.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_comparison(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = _resolve_path(args.data_dir)
    output_path = _resolve_path(args.output)
    summary_path = (
        output_path.with_name(f"{output_path.stem}_summary.json")
        if args.summary_output is None
        else _resolve_path(args.summary_output)
    )
    report_path = _resolve_path(args.report)
    examples = load_selected_examples(
        data_dir,
        sizes=args.sizes,
        instances_per_size=args.instances_per_size,
        seed=args.seed,
    )
    rows = evaluate_baselines(
        examples,
        solvers=tuple(args.solvers),
        time_budgets=tuple(args.time_budgets),
        seed=args.seed,
    )
    summary_rows = summarize(rows)
    provenance = {
        "data_dir": str(data_dir.relative_to(ROOT) if data_dir.is_relative_to(ROOT) else data_dir),
        "dataset_hash": hash_dataset(data_dir),
        "selected_subset_hash": _selected_hash(examples),
        "seed": int(args.seed),
        "sizes": [int(size) for size in args.sizes],
        "instances_per_size": int(args.instances_per_size),
        "time_budgets_seconds": [float(value) for value in args.time_budgets],
        "solvers": list(args.solvers),
        "selected_counts_by_size": {
            int(size): sum(example.instance.n_customers == int(size) for example in examples)
            for size in args.sizes
        },
        "selected_instance_ids": [example.instance.instance_id for example in examples],
        "git_commit": git_commit_hash(ROOT),
    }

    _write_csv(output_path, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {"provenance": provenance, "summary": summary_rows, "rows": rows},
            indent=2,
            default=str,
        )
        + "\n"
    )
    _write_markdown_report(
        report_path,
        summary_rows=summary_rows,
        provenance=provenance,
        detail_path=output_path,
        summary_path=summary_path,
    )

    print(f"wrote {output_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    print("\nbaseline | budget_s | size | gap_ref_% | runtime_s | feasible", flush=True)
    print("--- | --- | --- | --- | --- | ---", flush=True)
    for row in summary_rows:
        print(
            f"{row['baseline']} | {row['time_budget_seconds']:.2f} | {row['size']} | "
            f"{_format_cell(row['mean_gap_to_reference_percent'])} | "
            f"{_format_cell(row['mean_runtime_seconds'])} | "
            f"{_format_cell(row['feasible_rate'])}",
            flush=True,
        )
    return rows, summary_rows


def main() -> None:
    args = _parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
