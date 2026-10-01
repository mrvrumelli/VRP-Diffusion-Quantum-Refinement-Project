"""Run the refinement loop end to end on frozen baseline solutions, once per solver.

Input is a directory written by ``solve_with_baseline.py``. For every instance and solver,
``refine_solution`` starts from the baseline routes with the instance's saved prior and accepts
only feasible strict improvements. Solvers:

* ``classical`` — 2-opt for reorders, relocate/swap for exchanges (one deterministic run each).
* ``classical_restarts`` — the same local search from ``--classical-restarts`` starting points
  per subproblem (the current solution plus random ones). The default equals ``--reads``, which
  matches annealing's sample count; a smaller value can match its wall-clock time instead.
* ``sa_qubo`` / ``sa_qubo_bias`` — simulated annealing on the QUBO with ``--reads`` reads (a fixed
  count, so results do not depend on machine load), without and with the diffusion bias.
* ``sa_qubo_bias+classical`` — the biased annealing result, then polished by the classical loop.

Neighborhood size and count come from ``--max-reorder-size``, ``--max-exchange-size`` and
``--max-per-type`` (defaults: ``NeighborhoodConfig``). The loop stops after ``--rounds`` rounds or
after a round without improvement. Every solver is deterministic, so the cost recorded after each
round of one long run equals the result of a shorter run with that round cap.

Each finished instance is appended to ``rows.partial.jsonl``; a rerun with the same arguments
skips finished instances. At the end ``rows.jsonl`` and a summary are written: route gap before and
after refinement per size and solver, the mean gap after each round, paired bootstrap intervals
over instances, route counts and convergence.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vrp_diffusion_quantum.quantum.neighborhoods import NeighborhoodConfig
from vrp_diffusion_quantum.quantum.refinement import RefinementTrace, SubproblemSolver

ROOT = Path(__file__).resolve().parents[1]
# Configuration name -> solver stages run one after another, each a full refinement loop.
CONFIGS: dict[str, tuple[str, ...]] = {
    "classical": ("classical",),
    "classical_restarts": ("classical_restarts",),
    "sa_qubo": ("sa_qubo",),
    "sa_qubo_bias": ("sa_qubo_bias",),
    "sa_qubo_bias+classical": ("sa_qubo_bias", "classical"),
}
CONTRASTS = (
    ("sa_qubo", "classical"),
    ("sa_qubo", "classical_restarts"),
    ("sa_qubo_bias", "classical_restarts"),
    ("sa_qubo_bias", "sa_qubo"),
    ("classical_restarts", "classical"),
    ("sa_qubo_bias+classical", "classical"),
    ("sa_qubo_bias+classical", "classical_restarts"),
)


def _solver(name: str, options: dict[str, Any]) -> SubproblemSolver:
    from vrp_diffusion_quantum.quantum.qubo_bias import DiffusionBiasConfig
    from vrp_diffusion_quantum.quantum.refinement import AnnealingQUBOSolver, ClassicalSolver

    if name == "classical":
        return ClassicalSolver()
    if name == "classical_restarts":
        return ClassicalSolver(restarts=options["classical_restarts"], seed=options["seed"])
    bias = DiffusionBiasConfig(enabled=name == "sa_qubo_bias", alpha=float(options["alpha"]))
    return AnnealingQUBOSolver(
        num_reads=options["reads"], num_sweeps=options["sweeps"], seed=options["seed"], bias=bias
    )


def _round_profile(trace: RefinementTrace, cap: int) -> dict[str, Any]:
    """Cost and solver time after each round, and whether the loop stopped on its own."""
    last = max((step.round_index for step in trace.steps), default=-1)
    costs, seconds = [], []
    current = trace.initial_cost
    for index in range(last + 1):
        in_round = [step for step in trace.steps if step.round_index == index]
        if in_round:
            current = in_round[-1].cost_after
        costs.append(current)
        seconds.append(float(sum(step.runtime_seconds for step in in_round)))
    last_improved = any(
        step.accepted_improvement > 0.0 for step in trace.steps if step.round_index == last
    )
    return {
        "cost_by_round": costs,
        "solver_seconds_by_round": seconds,
        "converged": not (last_improved and last + 1 >= cap),
    }


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
    config = NeighborhoodConfig(**options["neighborhoods"])
    reference = float(record["reference_cost"])
    rows = []
    # Stage prefix -> (routes, traces, wall seconds), so a polish stage reuses earlier stages.
    finished: dict[tuple[str, ...], tuple[list[list[int]], list[RefinementTrace], float]] = {}
    for name in options["configs"]:
        stages = CONFIGS[name]
        routes, traces, wall = finished.get(stages[:-1], (record["routes"], [], 0.0))
        started = time.perf_counter()
        trace = refine_solution(
            instance,
            routes,
            _solver(stages[-1], options),
            m_prob=m_prob,
            config=config,
            max_rounds=options["rounds"],
        )
        wall += time.perf_counter() - started
        traces = [*traces, trace]
        finished[stages] = (trace.routes, traces, wall)
        if not validate_routes(instance, trace.routes).feasible:
            raise RuntimeError(f"refined routes infeasible for {instance_id} ({name})")
        steps = [step for stage in traces for step in stage.steps]
        accepted_by_type: dict[str, int] = {}
        for step in steps:
            if step.accepted_improvement > 0.0:
                key = step.neighborhood_type
                accepted_by_type[key] = accepted_by_type.get(key, 0) + 1
        initial_cost = traces[0].initial_cost
        profiles = [_round_profile(stage, options["rounds"]) for stage in traces]
        rows.append(
            {
                "instance_id": instance_id,
                "n_customers": instance.n_customers,
                "solver": name,
                "initial_cost": initial_cost,
                "final_cost": trace.final_cost,
                "reference_cost": reference,
                "gap_before": 100.0 * (initial_cost - reference) / reference,
                "gap_after": 100.0 * (trace.final_cost - reference) / reference,
                "routes_before": sum(1 for route in record["routes"] if route),
                "routes_after": sum(1 for route in trace.routes if route),
                "attempted_steps": len(steps),
                "accepted_steps": sum(stage.accepted_steps for stage in traces),
                "skipped_stale": sum(stage.skipped_stale for stage in traces),
                "rounds": [len(profile["cost_by_round"]) for profile in profiles],
                "cost_by_round": [profile["cost_by_round"] for profile in profiles],
                "solver_seconds_by_round": [
                    profile["solver_seconds_by_round"] for profile in profiles
                ],
                "converged": [profile["converged"] for profile in profiles],
                "solver_seconds": float(sum(s.runtime_seconds for s in steps)),
                "wall_seconds": wall,
                "accepted_by_type": accepted_by_type,
                "routes": trace.routes,
            }
        )
    return rows


def _ci(values: np.ndarray, seed: int = 0, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _gap_by_round(selected: list[dict[str, Any]], cap: int) -> list[float]:
    """Mean gap after rounds 1..R for single-stage runs; finished runs carry their final cost."""
    longest = max(len(row["cost_by_round"][0]) for row in selected)
    curve = []
    for index in range(min(longest, cap)):
        gaps = []
        for row in selected:
            costs = row["cost_by_round"][0]
            cost = costs[min(index, len(costs) - 1)] if costs else row["initial_cost"]
            gaps.append(100.0 * (cost - row["reference_cost"]) / row["reference_cost"])
        curve.append(float(np.mean(gaps)))
    return curve


def _summarise(rows: list[dict[str, Any]], ids: list[str], cap: int) -> dict[str, Any]:
    table = {(row["instance_id"], row["solver"]): row for row in rows}
    block: dict[str, Any] = {"instances": len(ids)}
    present = [name for name in CONFIGS if any(row["solver"] == name for row in rows)]
    for name in present:
        selected = [table[(i, name)] for i in ids]
        before = np.array([row["gap_before"] for row in selected])
        after = np.array([row["gap_after"] for row in selected])
        reduction = np.array(
            [100.0 * (r["initial_cost"] - r["final_cost"]) / r["initial_cost"] for r in selected]
        )
        block[name] = {
            "gap_before": float(before.mean()),
            "gap_after": float(after.mean()),
            "gap_after_median": float(np.median(after)),
            "change_pp": float((after - before).mean()),
            "change_pp_ci95": _ci(after - before),
            "cost_reduction_percent": float(reduction.mean()),
            "cost_reduction_percent_ci95": _ci(reduction),
            "gap_closed_percent": float(np.mean(100.0 * (before - after) / before)),
            "improved_instances": int(np.sum(after < before - 1e-9)),
            "at_or_below_reference": int(np.sum(after <= 1e-9)),
            "routes_before": float(np.mean([row["routes_before"] for row in selected])),
            "routes_after": float(np.mean([row["routes_after"] for row in selected])),
            "mean_rounds": float(np.mean([sum(row["rounds"]) for row in selected])),
            "converged_share": float(np.mean([all(row["converged"]) for row in selected])),
            "mean_accepted_steps": float(np.mean([row["accepted_steps"] for row in selected])),
            "mean_attempted_steps": float(np.mean([row["attempted_steps"] for row in selected])),
            "mean_solver_seconds": float(np.mean([row["solver_seconds"] for row in selected])),
            "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in selected])),
        }
        if len(CONFIGS[name]) == 1:
            block[name]["gap_by_round"] = _gap_by_round(selected, cap)
    for first, second in CONTRASTS:
        if first not in present or second not in present:
            continue
        diff = np.array(
            [table[(i, first)]["gap_after"] - table[(i, second)]["gap_after"] for i in ids]
        )
        block[f"{first}_minus_{second}_pp"] = {"mean": float(diff.mean()), "ci95": _ci(diff)}
    return block


def main() -> None:
    defaults = NeighborhoodConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reads", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-reorder-size", type=int, default=defaults.max_reorder_size)
    parser.add_argument("--max-exchange-size", type=int, default=defaults.max_exchange_size)
    parser.add_argument("--max-per-type", type=int, default=defaults.max_per_type)
    parser.add_argument(
        "--classical-restarts", type=int, help="starting points for classical_restarts (--reads)"
    )
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    for name in args.configs:
        prefix = CONFIGS[name][:-1]
        if prefix and not any(
            CONFIGS[earlier] == prefix for earlier in args.configs[: args.configs.index(name)]
        ):
            parser.error(f"{name} needs the configuration for {prefix} listed before it")
    neighborhoods = asdict(
        NeighborhoodConfig(
            max_reorder_size=args.max_reorder_size,
            max_exchange_size=args.max_exchange_size,
            max_per_type=args.max_per_type,
        )
    )
    options = {
        "reads": args.reads,
        "sweeps": args.sweeps,
        "alpha": args.alpha,
        "seed": args.seed,
        "rounds": args.rounds,
        "classical_restarts": args.classical_restarts or args.reads,
        "configs": args.configs,
        "neighborhoods": neighborhoods,
    }
    solutions = args.solutions if args.solutions.is_absolute() else ROOT / args.solutions
    records = [p for p in sorted(solutions.glob("*.json")) if p.name != "summary.json"]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "rows.partial.jsonl"
    options_file = output / "options.json"
    if options_file.exists() and json.loads(options_file.read_text()) != options:
        raise SystemExit(f"{output} holds a run with different options; use another --output")
    options_file.write_text(json.dumps(options, indent=1) + "\n")
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
    rows.sort(key=lambda row: (row["instance_id"], args.configs.index(row["solver"])))
    (output / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary: dict[str, Any] = {"solutions": str(args.solutions), "options": options, "by_size": {}}
    for size in sorted({row["n_customers"] for row in rows}):
        ids = sorted({row["instance_id"] for row in rows if row["n_customers"] == size})
        summary["by_size"][str(size)] = _summarise(rows, ids, args.rounds)
    summary["all"] = _summarise(rows, sorted({row["instance_id"] for row in rows}), args.rounds)
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for size, block in [*summary["by_size"].items(), ("all", summary["all"])]:
        for name in args.configs:
            stats = block[name]
            print(
                f"N{size} {name}: gap {stats['gap_before']:.2f} -> {stats['gap_after']:.2f}%, "
                f"rounds {stats['mean_rounds']:.1f}, converged {stats['converged_share']:.2f}, "
                f"solver {stats['mean_solver_seconds']:.1f}s"
            )


if __name__ == "__main__":
    main()
