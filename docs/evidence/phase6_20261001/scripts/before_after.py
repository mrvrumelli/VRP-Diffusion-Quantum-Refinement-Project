"""Before/after refinement summary from saved loop rows (no new solving)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(rows_path, extra=None):
    rows = [json.loads(l) for l in open(ROOT / rows_path)]
    if extra:
        for r in map(json.loads, open(ROOT / extra)):
            r["solver"] = "classical_x10"
            rows.append(r)
    return rows


def baseline_routes(folder, instance_id):
    rec = json.loads((ROOT / folder / f"{instance_id}.json").read_text())
    return sum(1 for r in rec["routes"] if r)


def ci(v, seed=0, draws=10000):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (draws, len(v)))].mean(axis=1)
    return f"[{np.percentile(m, 2.5):.2f}, {np.percentile(m, 97.5):.2f}]"


for name, rows_path, extra, folder in [
    ("DEVELOPMENT", "outputs/phase6_20261001/loop_development/rows.jsonl",
     "outputs/phase6_20261001/loop_development_classical_x10/rows.jsonl", "outputs/baseline_freeze_20261001/frozen_panel"),
    ("VALIDATION", "outputs/phase6_20261001/loop_validation/rows.jsonl", None, "outputs/baseline_freeze_20261001/frozen_validation"),
]:
    rows = load(rows_path, extra)
    solvers = [s for s in ("classical", "classical_x10", "classical_restarts", "sa_qubo", "sa_qubo_bias", "sa_qubo_bias+classical") if any(r["solver"] == s for r in rows)]
    print("=====", name)
    for size in (20, 50, 100):
        sel0 = [r for r in rows if r["n_customers"] == size and r["solver"] == "classical"]
        before = np.array([r["gap_before"] for r in sel0])
        routes0 = np.mean([baseline_routes(folder, r["instance_id"]) for r in sel0])
        print(f"N{size}: graphs={len(sel0)} gap before={before.mean():.2f}% median={np.median(before):.2f} max={before.max():.2f} routes={routes0:.2f}")
        for s in solvers:
            sel = sorted((r for r in rows if r["n_customers"] == size and r["solver"] == s), key=lambda r: r["instance_id"])
            b = np.array([r["gap_before"] for r in sel]); a = np.array([r["gap_after"] for r in sel])
            red = np.array([100 * (r["initial_cost"] - r["final_cost"]) / r["initial_cost"] for r in sel])
            closed = 100 * (b - a) / b
            nroutes = np.mean([sum(1 for x in r["routes"] if x) for r in sel])
            print(f"   {s:24s} after={a.mean():5.2f}% (median {np.median(a):5.2f}, max {a.max():5.2f})  cost -{red.mean():.2f}% {ci(red)}  gap closed {closed.mean():.0f}%  <=1%: {np.mean(a <= 1.0):.2f}  <=0: {np.mean(a <= 1e-9):.2f}  routes={nroutes:.2f}  s={np.mean([r['wall_seconds'] for r in sel]):.1f}")
