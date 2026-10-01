"""Neighborhood-selection ablation: diffusion-guided vs random vs geometric selection (Q4).

For every frozen baseline solution and subproblem solver, the refinement loop runs with:

* ``diffusion`` — the two prior-based selectors (uncertain matrix entries, low-confidence edges).
* ``random_any`` / ``random_adjacent`` — each round, the diffusion selectors are run on the
  current routes only to fix how many reorder and exchange neighborhoods of which sizes to try;
  the same number and sizes are then placed at random (``select_random_like``), with exchanges
  between any two routes or only between spatially adjacent routes. One run per ``--seeds``.
* ``geometric`` — the two prior-free selectors (expensive routes, nearby route pairs), as context;
  their counts are not matched.

``--diffusion-types`` picks which prior-based selectors form the diffusion arm (and fix the random
arms' counts), so each selector can be tested on its own; ``--groups`` picks the control arms. The
loop stops after a round without improvement or at ``--rounds``. Per run it records the final gap,
the cost after every round, and for every attempted neighborhood whether it improved the solution
and by how much. Finished instances are appended to ``rows.partial.jsonl`` and skipped on a rerun
with the same arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.quantum.neighborhoods import (
    Neighborhood,
    NeighborhoodType,
    RandomPairs,
    select_neighborhoods,
    select_random_like,
)
from vrp_diffusion_quantum.quantum.refinement import NeighborhoodSelector, SubproblemSolver

ROOT = Path(__file__).resolve().parents[1]
DIFFUSION: tuple[NeighborhoodType, ...] = ("uncertain_m", "low_confidence_edges")
GEOMETRIC: tuple[NeighborhoodType, ...] = ("high_cost_route", "two_route_exchange")
GROUPS = ("diffusion", "random_any", "random_adjacent", "geometric")


def _solver(name: str, options: dict[str, Any]) -> SubproblemSolver:
    from vrp_diffusion_quantum.quantum.refinement import AnnealingQUBOSolver, ClassicalSolver

    if name == "classical":
        return ClassicalSolver()
    if name == "classical_restarts":
        return ClassicalSolver(restarts=options["classical_restarts"], seed=options["seed"])
    if name == "sa_qubo":
        return AnnealingQUBOSolver(
            num_reads=options["reads"], num_sweeps=options["sweeps"], seed=options["seed"]
        )
    raise ValueError(f"unknown solver {name!r}")


def _fixed(types: Sequence[NeighborhoodType]) -> NeighborhoodSelector:
    def select(
        instance: CVRPInstance, routes: list[list[int]], m_prob: npt.NDArray[np.float64] | None
    ) -> list[Neighborhood]:
        return select_neighborhoods(instance, routes, m_prob=m_prob, types=types)

    return select


def _random(
    pairs: RandomPairs, seed: int, types: Sequence[NeighborhoodType]
) -> NeighborhoodSelector:
    rng = np.random.default_rng(seed)

    def select(
        instance: CVRPInstance, routes: list[list[int]], m_prob: npt.NDArray[np.float64] | None
    ) -> list[Neighborhood]:
        template = select_neighborhoods(instance, routes, m_prob=m_prob, types=types)
        return select_random_like(instance, routes, template, rng, pairs=pairs)

    return select


def _arms(seeds: Sequence[int], groups: Sequence[str]) -> list[tuple[str, str, int | None]]:
    arms: list[tuple[str, str, int | None]] = [("diffusion", "diffusion", None)]
    for seed in seeds:
        for group in ("random_any", "random_adjacent"):
            if group in groups:
                arms.append((f"{group}_s{seed}", group, seed))
    if "geometric" in groups:
        arms.append(("geometric", "geometric", None))
    return arms


def _instance_seed(instance_id: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{instance_id}".encode()).digest()[:4], "big")


def _run(task: tuple[Path, dict[str, Any]]) -> list[dict[str, Any]]:
    from vrp_diffusion_quantum.data.dataset import load_example
    from vrp_diffusion_quantum.quantum.refinement import refine_solution
    from vrp_diffusion_quantum.utils.feasibility import validate_routes

    record_path, options = task
    record = json.loads(record_path.read_text())
    instance_id = record["instance_id"]
    instance = load_example(ROOT / record["file"]).instance
    m_prob = np.load(record_path.with_name(f"{instance_id}.m_prob.npz"))["m_prob"]
    m_prob = m_prob.astype(np.float64)
    m_prob = np.clip((m_prob + m_prob.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(m_prob, 0.0)
    reference = float(record["reference_cost"])
    rows = []
    for solver_name in options["solvers"]:
        diffusion_types = tuple(options["diffusion_types"])
        for arm, group, seed in _arms(options["seeds"], options["groups"]):
            if group == "diffusion":
                selector = _fixed(diffusion_types)
            elif group == "geometric":
                selector = _fixed(GEOMETRIC)
            else:
                assert seed is not None
                pairs: RandomPairs = "any" if group == "random_any" else "adjacent"
                selector = _random(pairs, _instance_seed(instance_id, seed), diffusion_types)
            trace = refine_solution(
                instance,
                record["routes"],
                _solver(solver_name, options),
                m_prob=m_prob,
                max_rounds=options["rounds"],
                selector=selector,
            )
            if not validate_routes(instance, trace.routes).feasible:
                raise RuntimeError(f"infeasible result for {instance_id} ({arm}, {solver_name})")
            last = max((step.round_index for step in trace.steps), default=-1)
            costs, current = [], trace.initial_cost
            for index in range(last + 1):
                in_round = [step for step in trace.steps if step.round_index == index]
                if in_round:
                    current = in_round[-1].cost_after
                costs.append(current)
            rows.append(
                {
                    "instance_id": instance_id,
                    "n_customers": instance.n_customers,
                    "solver": solver_name,
                    "arm": arm,
                    "group": group,
                    "seed": seed,
                    "initial_cost": trace.initial_cost,
                    "final_cost": trace.final_cost,
                    "reference_cost": reference,
                    "gap_before": 100.0 * (trace.initial_cost - reference) / reference,
                    "gap_after": 100.0 * (trace.final_cost - reference) / reference,
                    "rounds": last + 1,
                    "cost_by_round": costs,
                    "skipped_stale": trace.skipped_stale,
                    "steps": [
                        [
                            step.round_index,
                            step.neighborhood_type,
                            step.neighborhood_size,
                            100.0 * step.accepted_improvement / step.cost_before,
                            step.runtime_seconds,
                        ]
                        for step in trace.steps
                    ],
                }
            )
    return rows


def _ci(values: npt.NDArray[np.float64], seed: int = 0, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [step for row in rows for step in row["steps"]]
    first = [step for step in steps if step[0] == 0]
    return {
        "runs": len(rows),
        "gap_after": float(np.mean([row["gap_after"] for row in rows])),
        "mean_rounds": float(np.mean([row["rounds"] for row in rows])),
        "attempted_per_run": len(steps) / len(rows),
        "improvement_frequency": float(np.mean([s[3] > 0 for s in steps])) if steps else 0.0,
        "mean_improvement_percent_per_attempt": float(np.mean([s[3] for s in steps]))
        if steps
        else 0.0,
        "round1_improvement_frequency": float(np.mean([s[3] > 0 for s in first])) if first else 0.0,
        "round1_cost_reduction_percent": float(
            np.mean(
                [
                    100.0
                    * (row["initial_cost"] - (row["cost_by_round"] or [row["initial_cost"]])[0])
                    / row["initial_cost"]
                    for row in rows
                ]
            )
        ),
        "mean_solver_seconds": float(np.mean([sum(s[4] for s in row["steps"]) for row in rows])),
    }


def _summarise(rows: list[dict[str, Any]], ids: list[str], solvers: list[str]) -> dict[str, Any]:
    block: dict[str, Any] = {"instances": len(ids)}
    present = [g for g in GROUPS if any(row["group"] == g for row in rows)]
    for solver in solvers:
        mine = [row for row in rows if row["solver"] == solver and row["instance_id"] in ids]
        stats = {
            group: _group_stats([row for row in mine if row["group"] == group]) for group in present
        }
        for group in present[1:]:
            per_instance = []
            for instance_id in ids:
                diff = next(
                    r["gap_after"]
                    for r in mine
                    if r["instance_id"] == instance_id and r["group"] == "diffusion"
                )
                other = [
                    r["gap_after"]
                    for r in mine
                    if r["instance_id"] == instance_id and r["group"] == group
                ]
                if other:
                    per_instance.append(diff - float(np.mean(other)))
            values = np.array(per_instance)
            stats[f"diffusion_minus_{group}_pp"] = {
                "mean": float(values.mean()),
                "ci95": _ci(values),
            }
        block[solver] = stats
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solvers", nargs="+", default=["classical", "sa_qubo"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--reads", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--classical-restarts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--diffusion-types", nargs="+", choices=list(DIFFUSION), default=list(DIFFUSION)
    )
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS[1:]), default=list(GROUPS[1:]))
    args = parser.parse_args()
    options = {
        "solvers": args.solvers,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "reads": args.reads,
        "sweeps": args.sweeps,
        "classical_restarts": args.classical_restarts,
        "seed": args.seed,
        "diffusion_types": args.diffusion_types,
        "groups": args.groups,
    }
    solutions = args.solutions if args.solutions.is_absolute() else ROOT / args.solutions
    records = [p for p in sorted(solutions.glob("*.json")) if p.name != "summary.json"]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    options_file = output / "options.json"
    if options_file.exists() and json.loads(options_file.read_text()) != options:
        raise SystemExit(f"{output} holds a run with different options; use another --output")
    options_file.write_text(json.dumps(options, indent=1) + "\n")
    partial = output / "rows.partial.jsonl"
    rows: list[dict[str, Any]] = []
    if partial.exists():
        rows = [json.loads(line) for line in partial.read_text().splitlines() if line.strip()]
    done = {row["instance_id"] for row in rows}
    pending = [p for p in records if p.stem not in done]
    print(f"{len(done)} instances already done, {len(pending)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, (p, options)) for p in pending]
        for future in as_completed(futures):
            chunk = future.result()
            rows.extend(chunk)
            with partial.open("a") as handle:
                handle.write("".join(json.dumps(row) + "\n" for row in chunk))
            print(f"done {len({row['instance_id'] for row in rows})}/{len(records)}", flush=True)
    rows.sort(key=lambda row: (row["instance_id"], row["solver"], row["arm"]))
    (output / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary: dict[str, Any] = {"solutions": str(args.solutions), "options": options, "by_size": {}}
    for size in sorted({row["n_customers"] for row in rows}):
        ids = sorted({row["instance_id"] for row in rows if row["n_customers"] == size})
        summary["by_size"][str(size)] = _summarise(rows, ids, args.solvers)
    summary["all"] = _summarise(rows, sorted({row["instance_id"] for row in rows}), args.solvers)
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for size, block in [*summary["by_size"].items(), ("all", summary["all"])]:
        for solver in args.solvers:
            stats = block[solver]
            line = " | ".join(
                f"{group} gap {stats[group]['gap_after']:.2f} freq "
                f"{stats[group]['improvement_frequency']:.3f}"
                for group in GROUPS
                if group in stats
            )
            contrasts = " | ".join(
                f"{key} {value['mean']:+.2f} [{value['ci95'][0]:+.2f}, {value['ci95'][1]:+.2f}]"
                for key, value in stats.items()
                if key.endswith("_pp")
            )
            print(f"N{size} {solver}: {line} | {contrasts}")


if __name__ == "__main__":
    main()
