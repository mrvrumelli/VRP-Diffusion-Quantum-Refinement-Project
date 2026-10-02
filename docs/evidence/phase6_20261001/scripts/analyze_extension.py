"""Summarise the rounds x neighborhood grid (outputs/phase6_20261001/extension)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "outputs/phase6_20261001/extension"
ARMS = ["default", "medium", "large", "default_more", "large_more"]
SOLVERS = ["classical", "classical_restarts", "sa_qubo"]
ROUND_MARKS = [1, 3, 5, 10]


def ci(values: np.ndarray, seed: int = 0, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def gap(cost: float, row: dict) -> float:
    return 100.0 * (cost - row["reference_cost"]) / row["reference_cost"]


def at_round(row: dict, r: int) -> float:
    costs = row["cost_by_round"][0]
    return gap(costs[min(r, len(costs)) - 1] if costs else row["initial_cost"], row)


def seconds_to_round(row: dict, r: int) -> float:
    return float(sum(row["solver_seconds_by_round"][0][:r]))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "development"
    data = {}
    for arm in ARMS:
        path = EXT / f"{name}_{arm}" / "rows.jsonl"
        if path.exists():
            data[arm] = {(r["instance_id"], r["solver"]): r for r in map(json.loads, open(path))}
    report: dict = {}
    for size in ("20", "50", "100", "all"):
        block: dict = {}
        for arm, table in data.items():
            for solver in SOLVERS:
                rows = sorted(
                    (r for (i, s), r in table.items() if s == solver and (size == "all" or str(r["n_customers"]) == size)),
                    key=lambda r: r["instance_id"],
                )
                if not rows:
                    continue
                before = np.array([r["gap_before"] for r in rows])
                after = np.array([r["gap_after"] for r in rows])
                block[f"{arm}|{solver}"] = {
                    "n": len(rows),
                    "gap_before": round(float(before.mean()), 2),
                    "gap_final": round(float(after.mean()), 2),
                    "gap_final_median": round(float(np.median(after)), 2),
                    "gap_closed_percent": round(float(np.mean(100 * (before - after) / before)), 1),
                    "cost_reduction_percent": round(float(np.mean([100 * (r["initial_cost"] - r["final_cost"]) / r["initial_cost"] for r in rows])), 2),
                    "at_or_below_reference": int(np.sum(after <= 1e-9)),
                    **{f"gap_round{m}": round(float(np.mean([at_round(r, m) for r in rows])), 2) for m in ROUND_MARKS},
                    **{f"solver_s_round{m}": round(float(np.mean([seconds_to_round(r, m) for r in rows])), 1) for m in ROUND_MARKS},
                    "mean_rounds": round(float(np.mean([r["rounds"][0] for r in rows])), 1),
                    "max_rounds": int(max(r["rounds"][0] for r in rows)),
                    "converged_share": round(float(np.mean([r["converged"][0] for r in rows])), 2),
                    "routes_before": round(float(np.mean([r["routes_before"] for r in rows])), 2),
                    "routes_after": round(float(np.mean([r["routes_after"] for r in rows])), 2),
                    "solver_seconds": round(float(np.mean([r["solver_seconds"] for r in rows])), 1),
                    "wall_seconds": round(float(np.mean([r["wall_seconds"] for r in rows])), 1),
                }
                if arm != "default" and "default" in data:
                    base = data["default"]
                    diff = np.array([r["gap_after"] - base[(r["instance_id"], solver)]["gap_after"] for r in rows])
                    block[f"{arm}|{solver}"]["minus_default_pp"] = [round(float(diff.mean()), 2), ci(diff)]
                if solver != "classical":
                    other = np.array([r["gap_after"] - table[(r["instance_id"], "classical")]["gap_after"] for r in rows])
                    block[f"{arm}|{solver}"]["minus_classical_pp"] = [round(float(other.mean()), 2), ci(other)]
                if solver == "sa_qubo":
                    other = np.array([r["gap_after"] - table[(r["instance_id"], "classical_restarts")]["gap_after"] for r in rows])
                    block[f"{arm}|{solver}"]["minus_restarts_pp"] = [round(float(other.mean()), 2), ci(other)]
        report[size] = block
    out = EXT / f"analysis_{name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for size, block in report.items():
        print(f"===== N{size}")
        for key, v in block.items():
            extra = " ".join(f"{k}={v[k]}" for k in ("minus_default_pp", "minus_restarts_pp") if k in v)
            print(
                f"{key:32s} before={v['gap_before']:6.2f} final={v['gap_final']:6.2f} med={v['gap_final_median']:6.2f} "
                f"r1={v['gap_round1']:6.2f} r3={v['gap_round3']:6.2f} r5={v['gap_round5']:6.2f} r10={v['gap_round10']:6.2f} "
                f"rounds={v['mean_rounds']:4.1f}/{v['max_rounds']:2d} conv={v['converged_share']:.2f} routes={v['routes_after']:5.2f} "
                f"s={v['solver_seconds']:6.1f} {extra}"
            )


if __name__ == "__main__":
    main()
