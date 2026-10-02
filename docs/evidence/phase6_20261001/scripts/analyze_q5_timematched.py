"""Q5 time-matched control: SA + restarted polish vs restarted classical with 20 restarts."""
import json
import sys
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parent / "extension"
name = sys.argv[1]
r20 = {r["instance_id"]: r for r in map(json.loads, open(EXT / f"{name}_default_restarts20/rows.jsonl"))}
r10 = {r["instance_id"]: r for r in map(json.loads, open(EXT / f"{name}_default/rows.jsonl")) if r["solver"] == "classical_restarts"}
sa = {r["instance_id"]: r for r in map(json.loads, open(EXT / f"{name}_default/rows.jsonl")) if r["solver"] == "sa_qubo"}
pol = {r["instance_id"]: r for r in map(json.loads, open(EXT / f"{name}_sa_then_classical/rows.jsonl")) if r["solver"] == "classical_restarts"}
ids = sorted(r20)


def ci(v):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(0).integers(0, len(v), (10000, len(v)))].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


out = {
    "restarts10_gap": round(float(np.mean([r10[i]["gap_after"] for i in ids])), 2),
    "restarts20_gap": round(float(np.mean([r20[i]["gap_after"] for i in ids])), 2),
    "sa_then_restarts10_gap": round(float(np.mean([pol[i]["gap_after"] for i in ids])), 2),
    "seconds_restarts10": round(float(np.mean([r10[i]["solver_seconds"] for i in ids])), 1),
    "seconds_restarts20": round(float(np.mean([r20[i]["solver_seconds"] for i in ids])), 1),
    "seconds_sa_then_restarts10": round(float(np.mean([sa[i]["solver_seconds"] + pol[i]["solver_seconds"] for i in ids])), 1),
}
d = np.array([pol[i]["gap_after"] - r20[i]["gap_after"] for i in ids])
out["sa_then_restarts_minus_restarts20"] = [round(float(d.mean()), 2), ci(d)]
d2 = np.array([r20[i]["gap_after"] - r10[i]["gap_after"] for i in ids])
out["restarts20_minus_restarts10"] = [round(float(d2.mean()), 2), ci(d2)]
(EXT / f"q5_timematched_{name}.json").write_text(json.dumps(out, indent=1) + "\n")
print(name, out)
