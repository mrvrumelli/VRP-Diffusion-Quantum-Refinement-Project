"""Evaluate the synthetic CVRP20/CVRP50/CVRP100 benchmark sets.

The input directory is expected to contain datasets written by
``vrp_diffusion_quantum.data.generate_cvrp.save_dataset``::

    cvrp20_nodes.csv       cvrp20_instances.csv
    cvrp50_nodes.csv       cvrp50_instances.csv
    cvrp100_nodes.csv      cvrp100_instances.csv

Reference costs are optional.  When ``--reference-dir`` is supplied, the script
looks for ``labels/cvrp<N>_labels.json`` (or ``cvrp<N>_labels.json``) and reports
the percentage gap to each reference cost.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vrp_diffusion_quantum.data.generate_cvrp import CVRPDataset, load_dataset
from vrp_diffusion_quantum.data.solve_cvrp import SolverName, solve_instance

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = (20, 50, 100)


def _dataset_path(data_dir: Path, size: int) -> Path:
    """Return the generated-dataset stem for one customer count."""
    candidates = (data_dir / f"cvrp{size}", data_dir / f"cvrp{size}_instances.csv")
    for candidate in candidates:
        if candidate.with_name(f"cvrp{size}_instances.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"no CVRP{size} dataset found under {data_dir}; expected "
        f"cvrp{size}_instances.csv and cvrp{size}_nodes.csv"
    )


def _reference_path(reference_dir: Path, size: int) -> Path:
    candidates = (
        reference_dir / "labels" / f"cvrp{size}_labels.json",
        reference_dir / f"cvrp{size}_labels.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no CVRP{size} reference labels found under {reference_dir}; "
        "expected labels/cvrp<N>_labels.json"
    )


def load_reference_costs(reference_dir: str | Path, size: int) -> list[float]:
    """Load reference costs in dataset order from a generated label JSON file."""
    path = _reference_path(Path(reference_dir), size)
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"reference labels must be a JSON list: {path}")
    try:
        return [float(row["cost"]) for row in payload]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"reference labels must contain numeric cost fields: {path}") from error


def evaluate_dataset(
    dataset: CVRPDataset,
    *,
    solver: SolverName = "pyvrp",
    time_limit: float = 1.0,
    seed: int = 0,
    reference_costs: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Solve a synthetic dataset and return one requested-metric row per instance."""
    if reference_costs is not None and len(reference_costs) != len(dataset):
        raise ValueError(
            f"reference costs ({len(reference_costs)}) do not match "
            f"dataset instances ({len(dataset)})"
        )

    rows: list[dict[str, Any]] = []
    for instance_index, instance in enumerate(dataset):
        instance_seed = int(
            np.random.SeedSequence([seed, dataset.n_customers, instance_index]).generate_state(1)[0]
        )
        solution = solve_instance(
            instance,
            solver=solver,
            time_limit=time_limit,
            seed=instance_seed,
            instance_id=instance_index,
        )
        reference_cost = None if reference_costs is None else reference_costs[instance_index]
        gap = None
        if reference_cost is not None:
            if reference_cost <= 0:
                raise ValueError(f"reference cost must be positive, got {reference_cost}")
            gap = 100.0 * (solution.cost - reference_cost) / reference_cost
        rows.append(
            {
                "size": dataset.n_customers,
                "instance": instance_index,
                "cost": float(solution.cost),
                "gap": gap,
                "runtime": float(solution.runtime_seconds),
                "feasibility": bool(solution.feasible),
                "number_of_vehicles": int(solution.num_vehicles),
                "solver": solver,
                "seed": instance_seed,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate metric rows by synthetic customer count."""
    summary: list[dict[str, Any]] = []
    for size in sorted({int(row["size"]) for row in rows}):
        size_rows = [row for row in rows if int(row["size"]) == size]
        gaps = [float(row["gap"]) for row in size_rows if row["gap"] is not None]
        summary.append(
            {
                "size": size,
                "instances": len(size_rows),
                "cost": float(np.mean([row["cost"] for row in size_rows])),
                "gap": None if not gaps else float(np.mean(gaps)),
                "runtime": float(np.mean([row["runtime"] for row in size_rows])),
                "feasibility": float(np.mean([row["feasibility"] for row in size_rows])),
                "number_of_vehicles": float(
                    np.mean([row["number_of_vehicles"] for row in size_rows])
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty evaluation result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--solver", choices=("pyvrp", "ortools"), default="pyvrp")
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "synthetic_results.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive")

    rows: list[dict[str, Any]] = []
    for size in args.sizes:
        dataset = load_dataset(_dataset_path(args.data_dir, size))
        references = (
            None if args.reference_dir is None else load_reference_costs(args.reference_dir, size)
        )
        rows.extend(
            evaluate_dataset(
                dataset,
                solver=args.solver,
                time_limit=args.time_limit,
                seed=args.seed,
                reference_costs=references,
            )
        )

    _write_csv(args.output, rows)
    summary = summarize(rows)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    for row in summary:
        print(
            f"CVRP{row['size']}: cost={row['cost']:.4f} gap={row['gap']} "
            f"runtime={row['runtime']:.4f}s feasibility={row['feasibility']:.4f} "
            f"number_of_vehicles={row['number_of_vehicles']:.2f}"
        )


if __name__ == "__main__":
    main()
