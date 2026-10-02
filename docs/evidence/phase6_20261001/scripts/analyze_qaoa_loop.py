"""In-loop QAOA at small limits: QAOA vs random sampling vs exact and restarted classical."""
import json
import sys
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parent / "extension"
name = sys.argv[1] if len(sys.argv) > 1 else "development"
rows = {(r["instance_id"], r["solver"]): r for r in map(json.loads, open(EXT / f"{name}_qaoa_small/rows.jsonl"))}
for r in map(json.loads, open(EXT / f"{name}_qaoa_small_exact/rows.jsonl")):
    rows[(r["instance_id"], r["solver"])] = r


def ci(v, draws=10000):
    v = np.asarray(v, float)
    rng = np.random.default_rng(0)
    m = v[rng.integers(0, len(v), (draws, len(v)))].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


solvers = ["classical", "classical_restarts", "classical_exact", "sa_qubo", "qaoa", "random_qubo", "qaoa+classical"]
report = {}
for size in ("20", "50", "100", "all"):
    ids = sorted({i for (i, s), r in rows.items() if s == "qaoa" and (size == "all" or str(r["n_customers"]) == size)})
    block = {"graphs": len(ids)}
    for s in solvers:
        sel = [rows[(i, s)] for i in ids if (i, s) in rows]
        if len(sel) != len(ids):
            continue
        block[s] = {"gap_after": round(float(np.mean([r["gap_after"] for r in sel])), 2),
                    "solver_seconds": round(float(np.mean([r["solver_seconds"] for r in sel])), 1)}
    for a, b in (("qaoa", "random_qubo"), ("qaoa", "classical_exact"), ("random_qubo", "classical_exact"), ("qaoa", "classical_restarts"), ("classical_exact", "classical_restarts"), ("qaoa+classical", "classical_restarts")):
        if a in block and b in block:
            d = np.array([rows[(i, a)]["gap_after"] - rows[(i, b)]["gap_after"] for i in ids])
            block[f"{a}_minus_{b}"] = [round(float(d.mean()), 2), ci(d), f"{int((d < -1e-9).sum())} better / {int((d > 1e-9).sum())} worse"]
    report[size] = block
(EXT / f"qaoa_loop_{name}.json").write_text(json.dumps(report, indent=1) + "\n")
for size, block in report.items():
    print("== N" + size, json.dumps({k: v for k, v in block.items()}))
