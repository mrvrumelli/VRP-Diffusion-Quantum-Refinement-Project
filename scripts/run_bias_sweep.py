"""Diffusion-bias sweep on a frozen neighborhood set: strength, mode and seeds (Q3).

Every neighborhood of the set is extracted from its frozen starting routes and solved by
simulated annealing with a fixed number of reads, once without the bias and once per
``--alphas`` x ``--modes`` setting, for each of ``--seeds``. Unbiased and biased runs of one seed
share the seed. The exact subproblem optimum (Held-Karp or exhaustive assignment search) is the
reference. Results: optimum hit rate and kept excess (after the loop's accept-if-better rule) per
kind and setting, and the paired difference to the unbiased run with a bootstrap over instances.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _run(task: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    from vrp_diffusion_quantum.data.dataset import load_example
    from vrp_diffusion_quantum.local_search.baselines import solve_exchange, solve_reorder
    from vrp_diffusion_quantum.quantum.neighborhoods import (
        Neighborhood,
        extract_exchange_subproblem,
        extract_reorder_subproblem,
    )
    from vrp_diffusion_quantum.quantum.qubo_bias import BiasMode, DiffusionBiasConfig
    from vrp_diffusion_quantum.quantum.qubo_exchange import build_exchange_qubo
    from vrp_diffusion_quantum.quantum.refinement import AnnealingQUBOSolver
    from vrp_diffusion_quantum.utils.feasibility import route_cost

    record, entries, options = task
    instance = load_example(ROOT / record["file"]).instance
    m_prob = np.load(ROOT / record["m_prob"])["m_prob"].astype(np.float64)
    m_prob = np.clip((m_prob + m_prob.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(m_prob, 0.0)
    settings: list[tuple[float, str]] = [(0.0, "none")]
    settings += [(a, m) for a in options["alphas"] for m in options["modes"] if a > 0]
    rows = []
    for entry in entries:
        neighborhood = Neighborhood(
            entry["neighborhood_type"],
            entry["kind"],
            tuple(entry["route_indices"]),
            tuple(entry["customers"]),
            float(entry["score"]),
            tuple(entry["segment"]) if entry["segment"] is not None else None,
        )
        if neighborhood.kind == "reorder":
            sub: Any = extract_reorder_subproblem(instance, record["routes"], neighborhood)
            initial = sub.current_cost
            optimum = solve_reorder(sub, "exact").cost_after
            method = "solve_reorder"
        else:
            sub = extract_exchange_subproblem(instance, record["routes"], neighborhood)
            if sub.size > options["max_exhaustive"]:
                continue
            try:
                build_exchange_qubo(instance, sub)
            except ValueError:
                continue  # fixed customers alone exceed capacity: not a valid subproblem
            assert sub.initial_routes is not None
            initial = route_cost(instance, [list(r) for r in sub.initial_routes if r])
            optimum = solve_exchange(instance, sub, "exhaustive").cost_after
            method = "solve_exchange"
        for seed in options["seeds"]:
            for alpha, mode in settings:
                bias = (
                    DiffusionBiasConfig()
                    if mode == "none"
                    else DiffusionBiasConfig(enabled=True, alpha=alpha, mode=cast(BiasMode, mode))
                )
                solver = AnnealingQUBOSolver(
                    num_reads=options["reads"], num_sweeps=options["sweeps"], seed=seed, bias=bias
                )
                outcome = getattr(solver, method)(instance, sub, m_prob)
                cost = outcome.candidate_cost if outcome.post_repair_feasible else float("inf")
                kept = min(cost, initial)
                rows.append(
                    {
                        "id": entry["id"],
                        "instance_id": record["instance_id"],
                        "kind": neighborhood.kind,
                        "seed": seed,
                        "alpha": alpha,
                        "mode": mode,
                        "hit": bool(kept <= optimum + 1e-9),
                        "kept_excess": 100.0 * (kept - optimum) / optimum,
                        "feasible": bool(outcome.post_repair_feasible),
                        "qubo_variables": outcome.qubo_num_variables,
                    }
                )
    return rows


def _cluster_ci(by_instance: dict[str, list[float]], draws: int = 5000) -> list[float]:
    keys = list(by_instance)
    rng = np.random.default_rng(0)
    means = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        means.append(float(np.mean(np.concatenate([by_instance[keys[i]] for i in pick]))))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--modes", nargs="+", default=["probability", "confidence"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--reads", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--max-exhaustive", type=int, default=12)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    options = {
        "alphas": args.alphas,
        "modes": args.modes,
        "seeds": args.seeds,
        "reads": args.reads,
        "sweeps": args.sweeps,
        "max_exhaustive": args.max_exhaustive,
    }
    neighborhood_set = json.loads((ROOT / args.set).read_text())
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in neighborhood_set["neighborhoods"]:
        by_instance[entry["instance_id"]].append(entry)
    tasks = [(r, by_instance[r["instance_id"]], options) for r in neighborhood_set["instances"]]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"done {count}/{len(tasks)}", flush=True)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    base = {(r["id"], r["seed"]): r for r in rows if r["mode"] == "none"}
    summary: dict[str, Any] = {"set": str(args.set), "options": options, "settings": {}}
    for kind in ("reorder", "exchange"):
        settings = sorted({(r["alpha"], r["mode"]) for r in rows if r["kind"] == kind})
        for alpha, mode in settings:
            sel = [
                r for r in rows if r["kind"] == kind and r["alpha"] == alpha and r["mode"] == mode
            ]
            block: dict[str, Any] = {
                "runs": len(sel),
                "hit_rate": float(np.mean([r["hit"] for r in sel])),
                "kept_excess_percent": float(np.mean([r["kept_excess"] for r in sel])),
                "feasible_rate": float(np.mean([r["feasible"] for r in sel])),
            }
            if mode != "none":
                excess: dict[str, list[float]] = defaultdict(list)
                hits: dict[str, list[float]] = defaultdict(list)
                for r in sel:
                    b = base[(r["id"], r["seed"])]
                    excess[r["instance_id"]].append(r["kept_excess"] - b["kept_excess"])
                    hits[r["instance_id"]].append(float(r["hit"]) - float(b["hit"]))
                block["minus_unbiased_kept_excess_pp"] = float(
                    np.mean(np.concatenate(list(excess.values())))
                )
                block["minus_unbiased_kept_excess_ci95"] = _cluster_ci(excess)
                block["minus_unbiased_hit_rate"] = float(
                    np.mean(np.concatenate(list(hits.values())))
                )
                block["minus_unbiased_hit_rate_ci95"] = _cluster_ci(hits)
            summary["settings"][f"{kind}|{mode}|{alpha:g}"] = block
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for key, block in summary["settings"].items():
        extra = ""
        if "minus_unbiased_kept_excess_pp" in block:
            low, high = block["minus_unbiased_kept_excess_ci95"]
            diff = block["minus_unbiased_kept_excess_pp"]
            extra = f" | vs unbiased {diff:+.2f} [{low:+.2f}, {high:+.2f}]"
        print(
            f"{key:28s} hit {block['hit_rate']:.3f} kept excess "
            f"{block['kept_excess_percent']:.2f}%{extra}"
        )


if __name__ == "__main__":
    main()
