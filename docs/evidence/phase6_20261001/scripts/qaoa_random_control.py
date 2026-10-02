"""Random-sampling control for the QAOA best-of-samples result.

For every exported QUBO: draw 1024 uniform random bitstrings (seeded), keep the 50 lowest-energy
unique ones (the same storage rule as the QAOA screening), decode/repair each and keep the best
true cost. Compares hit rate and kept excess with QAOA under the identical rule.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/phase6_20261001"
SET = sys.argv[1] if len(sys.argv) > 1 else "development"
spec = importlib.util.spec_from_file_location("export_qubo", ROOT / "scripts/export_qubo_instances.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

neighborhood_set = json.loads((P6 / f"neighborhoods_{SET}.json").read_text())
records = {r["instance_id"]: r for r in neighborhood_set["instances"]}
entries = {e["id"]: e for e in neighborhood_set["neighborhoods"]}
qubos = json.loads((P6 / f"qaoa_qubos_{SET}.json").read_text())["qubos"]
qaoa = {(r["id"], r["reps"]): r for r in json.loads((P6 / f"qaoa_analysis_{SET}.json").read_text())["rows"]}
experiment = {}
for line in (P6 / f"experiment_{SET}/rows.jsonl").read_text().splitlines():
    row = json.loads(line)
    experiment.setdefault(row["id"], {})[row["method"]] = row

rows = []
for index, q in enumerate(qubos):
    entry = entries[q["id"]]
    instance, sub, wrapper, optimum = module._subproblem_and_qubo(records[entry["instance_id"]], entry)
    matrix = np.asarray(q["matrix"])
    n = matrix.shape[0]
    rng = np.random.default_rng(1000 + index)
    shots = rng.integers(0, 2, (1024, n))
    unique = np.unique(shots, axis=0)
    energies = np.einsum("si,ij,sj->s", unique, matrix, unique) + q["offset"]
    kept = unique[np.argsort(energies)[:50]]
    costs = [module._score_sample(instance, sub, wrapper, [int(b) for b in x])[0] for x in kept]
    initial = experiment[q["id"]]["exact_reference"]["initial_cost"]
    rows.append({"id": q["id"], "kind": q["kind"], "n": n, "initial": initial, "optimum": optimum, "random_best": min(costs)})

report = {}
for kind in ("reorder", "exchange"):
    sel = [r for r in rows if r["kind"] == kind]
    for bucket, lo, hi in (("all", 0, 99), ("1-8", 0, 8), ("9-12", 9, 12), ("13-16", 13, 16)):
        part = [r for r in sel if lo <= r["n"] <= hi]
        if not part:
            continue
        opt = np.array([r["optimum"] for r in part])
        init = np.array([r["initial"] for r in part])
        block = {"n": len(part)}
        rand = np.minimum(np.array([r["random_best"] for r in part]), init)
        block["random"] = [round(float(np.mean(rand <= opt + 1e-9)), 3), round(float(np.mean(100 * (rand - opt) / opt)), 3)]
        for reps in (1, 2, 3):
            qa = np.minimum(np.array([qaoa[(r["id"], reps)]["best_of_samples_true_cost"] for r in part]), init)
            block[f"qaoa_p{reps}"] = [round(float(np.mean(qa <= opt + 1e-9)), 3), round(float(np.mean(100 * (qa - opt) / opt)), 3)]
        for method in ("classical_ls", "classical_ls_budget", "sa_qubo", "sa_qubo_bias"):
            c = np.minimum(np.array([experiment[r["id"]][method]["cost"] for r in part]), init)
            block[method] = [round(float(np.mean(c <= opt + 1e-9)), 3), round(float(np.mean(100 * (c - opt) / opt)), 3)]
        report[f"{kind}|q={bucket}"] = block
(P6 / f"qaoa_random_control_{SET}.json").write_text(json.dumps(report, indent=1) + "\n")
print("[hit rate, kept excess %]")
for key, block in report.items():
    print(key, json.dumps(block))
