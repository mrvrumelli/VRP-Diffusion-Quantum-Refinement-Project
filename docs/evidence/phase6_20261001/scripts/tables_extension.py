"""Markdown tables for the rounds x neighborhood grid (Phase 6 write-up, section 3)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "outputs/phase6_20261001/extension"
ARMS = [
    ("default", "5 / 8 / 10"),
    ("medium", "7 / 12 / 10"),
    ("large", "10 / 16 / 10"),
    ("default_more", "5 / 8 / 30"),
    ("large_more", "10 / 16 / 30"),
]
SOLVERS = [("classical", "Classical, one pass"), ("classical_restarts", "Classical, 10 restarts"), ("sa_qubo", "SA on QUBO")]


def ci(values, seed=0, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(name, arm):
    path = EXT / f"{name}_{arm}" / "rows.jsonl"
    if not path.exists():
        return None
    return {(r["instance_id"], r["solver"]): r for r in map(json.loads, open(path))}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "development"
    default = load(name, "default")
    print(f"#### {name}: final gap % after convergence (rounds capped at 30)\n")
    print("| Solver | Reorder / exchange size / per round | N20 | N50 | N100 | All | Change vs default limits | Solver s per graph |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for solver, label in SOLVERS:
        for arm, limits in ARMS:
            table = load(name, arm)
            if table is None:
                continue
            rows = [r for (i, s), r in table.items() if s == solver]
            by_size = {n: np.mean([r["gap_after"] for r in rows if r["n_customers"] == n]) for n in (20, 50, 100)}
            overall = np.mean([r["gap_after"] for r in rows])
            seconds = np.mean([r["solver_seconds"] for r in rows])
            change = "—"
            if arm != "default" and default is not None:
                diff = [r["gap_after"] - default[(r["instance_id"], solver)]["gap_after"] for r in rows]
                low, high = ci(diff)
                change = f"{np.mean(diff):+.2f} [{low:+.2f}, {high:+.2f}]"
            print(
                f"| {label} | {limits} | {by_size[20]:.2f} | {by_size[50]:.2f} | {by_size[100]:.2f} | "
                f"{overall:.2f} | {change} | {seconds:.0f} |"
            )
    if default is not None:
        print(f"\n#### {name}: gap % after each round, default limits\n")
        print("| Solver | Before | Round 1 | Round 2 | Round 3 | Round 5 | Round 10 | Converged | Mean rounds |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for solver, label in SOLVERS:
            rows = [r for (i, s), r in default.items() if s == solver]

            def at(r, k):
                costs = r["cost_by_round"][0]
                cost = costs[min(k, len(costs)) - 1] if costs else r["initial_cost"]
                return 100 * (cost - r["reference_cost"]) / r["reference_cost"]

            cells = [np.mean([at(r, k) for r in rows]) for k in (1, 2, 3, 5, 10)]
            print(
                f"| {label} | {np.mean([r['gap_before'] for r in rows]):.2f} | "
                + " | ".join(f"{c:.2f}" for c in cells)
                + f" | {np.mean([r['gap_after'] for r in rows]):.2f} | {np.mean([r['rounds'][0] for r in rows]):.1f} |"
            )


if __name__ == "__main__":
    main()
