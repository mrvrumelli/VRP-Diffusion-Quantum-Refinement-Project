"""Q4 analysis: diffusion vs random vs geometric selection, plus all four selectors vs geometric.

The all-four arm comes from the extension grid's default arm (same solvers, rounds and limits).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/phase6_20261001"


def ci(values, seed=0, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


name = sys.argv[1] if len(sys.argv) > 1 else "development"
set_name = name.split("_")[0]
rows = [json.loads(l) for l in open(P6 / f"selection/{name}/rows.jsonl")]
default_path = P6 / f"extension/{set_name}_default/rows.jsonl"
GROUPS = [g for g in ("diffusion", "random_any", "random_adjacent", "geometric") if any(r["group"] == g for r in rows)]
allfour = {}
if default_path.exists():
    for l in open(default_path):
        r = json.loads(l)
        allfour[(r["instance_id"], r["solver"])] = r
by = defaultdict(list)
for r in rows:
    by[(r["instance_id"], r["solver"], r["group"])].append(r)
report = {}
for solver in ("classical", "sa_qubo"):
    for size in ("20", "50", "100", "all"):
        ids = sorted({r["instance_id"] for r in rows if size == "all" or str(r["n_customers"]) == size})
        block = {"instances": len(ids)}
        for group in GROUPS:
            runs = [r for i in ids for r in by[(i, solver, group)]]
            steps = [s for r in runs for s in r["steps"]]
            first = [s for s in steps if s[0] == 0]
            block[group] = {
                "gap_after": round(float(np.mean([np.mean([r["gap_after"] for r in by[(i, solver, group)]]) for i in ids])), 2),
                "attempted_per_run": round(len(steps) / len(runs), 1),
                "improvement_frequency": round(float(np.mean([s[3] > 0 for s in steps])), 3),
                "improvement_per_attempt_percent": round(float(np.mean([s[3] for s in steps])), 3),
                "round1_improvement_frequency": round(float(np.mean([s[3] > 0 for s in first])), 3),
                "round1_cost_reduction_percent": round(float(np.mean([100 * (r["initial_cost"] - (r["cost_by_round"] or [r["initial_cost"]])[0]) / r["initial_cost"] for r in runs])), 2),
                "mean_rounds": round(float(np.mean([r["rounds"] for r in runs])), 1),
            }
        gap = {g: np.array([np.mean([r["gap_after"] for r in by[(i, solver, g)]]) for i in ids]) for g in GROUPS}
        for g in GROUPS[1:]:
            d = gap["diffusion"] - gap[g]
            block[f"diffusion_minus_{g}_pp"] = [round(float(d.mean()), 2), ci(d), f"diffusion better on {int(np.sum(d < -1e-9))}/{len(d)}"]
        # Round-1 cost reduction, diffusion vs random adjacent (same starting solution, matched counts)
        r1 = lambda i, g: np.mean([100 * (r["initial_cost"] - (r["cost_by_round"] or [r["initial_cost"]])[0]) / r["initial_cost"] for r in by[(i, solver, g)]])
        d1 = np.array([r1(i, "diffusion") - r1(i, "random_adjacent") for i in ids])
        block["round1_reduction_diffusion_minus_random_adjacent_pp"] = [round(float(d1.mean()), 2), ci(d1)]
        if allfour and "geometric" in GROUPS:
            a = np.array([allfour[(i, solver)]["gap_after"] for i in ids])
            d = a - gap["geometric"]
            block["all_four_gap_after"] = round(float(a.mean()), 2)
            block["all_four_minus_geometric_pp"] = [round(float(d.mean()), 2), ci(d), f"all four better on {int(np.sum(d < -1e-9))}/{len(d)}"]
        report[f"{solver}|N{size}"] = block
(P6 / f"selection/analysis_{name}.json").write_text(json.dumps(report, indent=1) + "\n")
for key, block in report.items():
    print("=====", key, "instances", block["instances"])
    for g in GROUPS:
        print(f"  {g:16s} {json.dumps(block[g])}")
    for k, v in block.items():
        if k.endswith("_pp") or k == "all_four_gap_after":
            print(f"  {k}: {v}")
