"""Matched classical vs QUBO comparison on a frozen neighborhood set.

Every neighborhood in the set is extracted from its frozen starting routes and solved by each
method on the identical subproblem. Methods:

* ``exact_reference`` — Held-Karp (reorder) or exhaustive search (exchange): the optimum of the
  subproblem under the shared true-cost evaluation.
* ``classical_ls`` — one deterministic 2-opt or relocate/swap run from the current solution.
* ``classical_ls_budget`` — random-restart local search until the time budget is spent (the
  first restart is the current solution).
* ``sa_qubo`` / ``sa_qubo_bias`` — simulated annealing on the QUBO under the same time budget,
  without and with the diffusion bias; every sample is decoded/repaired and judged on true cost.
* ``exact_qubo`` — the QUBO's exact minimum where it has at most ``--exact-qubo-max`` variables:
  what the formulation itself prefers.

Budgets are wall-clock per subproblem (the budget unit is an open team decision in
docs/quantum_scope.md; sample counts are recorded too). Results are written as JSON lines plus a
summary of optimum hit rate, excess over the optimum, improvement and runtime per method.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.local_search.baselines import ExchangeSolution, ReorderSolution
from vrp_diffusion_quantum.quantum.neighborhoods import ExchangeSubproblem, ReorderSubproblem

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "exact_reference",
    "classical_ls",
    "classical_ls_budget",
    "sa_qubo",
    "sa_qubo_bias",
    "exact_qubo",
)


def _reorder_restarts(subproblem: ReorderSubproblem, budget: float, seed: int) -> tuple[float, int]:
    from vrp_diffusion_quantum.local_search.baselines import solve_reorder

    rng = np.random.default_rng(seed)
    k = subproblem.size
    best = np.inf
    restarts = 0
    started = time.perf_counter()
    while restarts == 0 or time.perf_counter() - started < budget:
        perm = np.arange(k) if restarts == 0 else rng.permutation(k)
        index = [0, *(int(i) + 1 for i in perm), k + 1]
        shuffled = dataclasses.replace(
            subproblem,
            customers=tuple(subproblem.customers[int(i)] for i in perm),
            distances=subproblem.distances[np.ix_(index, index)],
        )
        best = min(best, solve_reorder(shuffled, "two_opt").cost_after)
        restarts += 1
    return float(best), restarts


def _exchange_restarts(
    instance: CVRPInstance, subproblem: ExchangeSubproblem, budget: float, seed: int
) -> tuple[float, int]:
    from vrp_diffusion_quantum.local_search.baselines import solve_exchange

    rng = np.random.default_rng(seed)
    best = np.inf
    restarts = 0
    started = time.perf_counter()
    while restarts == 0 or time.perf_counter() - started < budget:
        start = (
            subproblem.initial_assignment
            if restarts == 0
            else tuple(int(b) for b in rng.integers(0, 2, subproblem.size))
        )
        shuffled = dataclasses.replace(subproblem, initial_assignment=start)
        solution = solve_exchange(instance, shuffled, "relocate_swap")
        if solution.feasible:
            best = min(best, solution.cost_after)
        restarts += 1
    return float(best), restarts


def _run_instance(
    task: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    from vrp_diffusion_quantum.data.dataset import load_example
    from vrp_diffusion_quantum.local_search.baselines import solve_exchange, solve_reorder
    from vrp_diffusion_quantum.quantum.neighborhoods import (
        Neighborhood,
        extract_exchange_subproblem,
        extract_reorder_subproblem,
    )
    from vrp_diffusion_quantum.quantum.qubo_bias import DiffusionBiasConfig
    from vrp_diffusion_quantum.quantum.qubo_exchange import build_exchange_qubo
    from vrp_diffusion_quantum.quantum.qubo_reorder import build_reorder_qubo
    from vrp_diffusion_quantum.quantum.refinement import AnnealingQUBOSolver, ExactQUBOSolver
    from vrp_diffusion_quantum.utils.feasibility import route_cost

    instance_record, neighborhoods, options = task
    example = load_example(ROOT / instance_record["file"])
    instance = example.instance
    routes = instance_record["routes"]
    m_prob = np.load(ROOT / instance_record["m_prob"])["m_prob"].astype(np.float64)
    m_prob = np.clip((m_prob + m_prob.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(m_prob, 0.0)
    budget = float(options["budget"])
    sa = AnnealingQUBOSolver(
        num_sweeps=options["sweeps"], seed=options["seed"], time_budget_seconds=budget
    )
    sa_bias = dataclasses.replace(
        sa, bias=DiffusionBiasConfig(enabled=True, alpha=float(options["alpha"]))
    )
    exact_qubo = ExactQUBOSolver()
    rows = []
    for entry in neighborhoods:
        neighborhood = Neighborhood(
            entry["neighborhood_type"],
            entry["kind"],
            tuple(entry["route_indices"]),
            tuple(entry["customers"]),
            float(entry["score"]),
            tuple(entry["segment"]) if entry["segment"] is not None else None,
        )
        results: dict[str, dict[str, Any]] = {}
        sub: ReorderSubproblem | ExchangeSubproblem
        exact: ReorderSolution | ExchangeSolution
        ls: ReorderSolution | ExchangeSolution
        if neighborhood.kind == "reorder":
            sub = extract_reorder_subproblem(instance, routes, neighborhood)
            initial = sub.current_cost
            started = time.perf_counter()
            exact = solve_reorder(sub, "exact")
            results["exact_reference"] = {
                "cost": exact.cost_after,
                "seconds": time.perf_counter() - started,
            }
            ls = solve_reorder(sub, "two_opt")
            results["classical_ls"] = {"cost": ls.cost_after, "seconds": ls.runtime_seconds}
            started = time.perf_counter()
            cost, restarts = _reorder_restarts(sub, budget, options["seed"])
            results["classical_ls_budget"] = {
                "cost": cost,
                "seconds": time.perf_counter() - started,
                "samples": restarts,
            }
            vars_count = build_reorder_qubo(sub).qubo.num_variables
            solve = "solve_reorder"
        else:
            sub = extract_exchange_subproblem(instance, routes, neighborhood)
            assert sub.initial_routes is not None
            initial = route_cost(instance, [list(r) for r in sub.initial_routes if r])
            if sub.size > options["max_exhaustive"]:
                continue
            exact = solve_exchange(instance, sub, "exhaustive")
            results["exact_reference"] = {
                "cost": exact.cost_after,
                "seconds": exact.runtime_seconds,
            }
            ls = solve_exchange(instance, sub, "relocate_swap")
            results["classical_ls"] = {
                "cost": ls.cost_after if ls.feasible else float("inf"),
                "seconds": ls.runtime_seconds,
            }
            started = time.perf_counter()
            cost, restarts = _exchange_restarts(instance, sub, budget, options["seed"])
            results["classical_ls_budget"] = {
                "cost": cost,
                "seconds": time.perf_counter() - started,
                "samples": restarts,
            }
            try:
                vars_count = build_exchange_qubo(instance, sub).qubo.num_variables
            except ValueError:
                continue  # fixed customers alone exceed capacity: not a valid subproblem
            solve = "solve_exchange"
        for name, solver in (("sa_qubo", sa), ("sa_qubo_bias", sa_bias)):
            outcome = getattr(solver, solve)(instance, sub, m_prob)
            results[name] = {
                "cost": outcome.candidate_cost if outcome.post_repair_feasible else float("inf"),
                "seconds": outcome.runtime_seconds,
                "samples": outcome.num_samples,
            }
        if vars_count <= options["exact_qubo_max"]:
            outcome = getattr(exact_qubo, solve)(instance, sub, None)
            results["exact_qubo"] = {
                "cost": outcome.candidate_cost if outcome.post_repair_feasible else float("inf"),
                "seconds": outcome.runtime_seconds,
            }
        optimum = results["exact_reference"]["cost"]
        for method, values in results.items():
            cost = values["cost"]
            rows.append(
                {
                    "id": entry["id"],
                    "instance_id": instance_record["instance_id"],
                    "n_customers": instance.n_customers,
                    "neighborhood_type": neighborhood.neighborhood_type,
                    "kind": neighborhood.kind,
                    "neighborhood_size": neighborhood.size,
                    "qubo_num_variables": vars_count,
                    "method": method,
                    "initial_cost": initial,
                    "optimum_cost": optimum,
                    "cost": cost,
                    "hit_optimum": bool(cost <= optimum + 1e-9),
                    "excess_percent": 100.0 * (cost - optimum) / optimum
                    if np.isfinite(cost)
                    else None,
                    "improvement": initial - cost if np.isfinite(cost) else None,
                    "seconds": values["seconds"],
                    "samples": values.get("samples"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-exhaustive", type=int, default=12)
    parser.add_argument("--exact-qubo-max", type=int, default=20)
    args = parser.parse_args()
    neighborhood_set = json.loads((ROOT / args.set).read_text())
    options = {
        "budget": args.budget,
        "sweeps": args.sweeps,
        "alpha": args.alpha,
        "seed": args.seed,
        "max_exhaustive": args.max_exhaustive,
        "exact_qubo_max": args.exact_qubo_max,
    }
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in neighborhood_set["neighborhoods"]:
        by_instance[entry["instance_id"]].append(entry)
    tasks = [
        (record, by_instance[record["instance_id"]], options)
        for record in neighborhood_set["instances"]
    ]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(_run_instance, tasks, chunksize=1):
            rows.extend(chunk)
    (output / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary: dict[str, Any] = {"set": str(args.set), "options": options, "groups": {}}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row["method"])].append(row)
        groups[(row["neighborhood_type"], row["method"])].append(row)
    for (group, method), members in sorted(groups.items()):
        finite = [m for m in members if m["excess_percent"] is not None]
        summary["groups"][f"{group}|{method}"] = {
            "count": len(members),
            "hit_rate": float(np.mean([m["hit_optimum"] for m in members])),
            "mean_excess_percent": float(np.mean([m["excess_percent"] for m in finite]))
            if finite
            else None,
            "failures": len(members) - len(finite),
            "mean_improvement": float(np.mean([m["improvement"] for m in finite]))
            if finite
            else None,
            "mean_seconds": float(np.mean([m["seconds"] for m in members])),
            "mean_samples": float(
                np.mean([m["samples"] for m in members if m["samples"] is not None])
            )
            if any(m["samples"] is not None for m in members)
            else None,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for key, value in summary["groups"].items():
        if key.startswith(("reorder|", "exchange|")):
            print(key, json.dumps(value))


if __name__ == "__main__":
    main()
