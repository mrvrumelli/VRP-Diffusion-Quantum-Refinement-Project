"""Loop-level effect of the strong bias (alpha 2): sa_qubo_bias vs unbiased sa_qubo, default limits."""
import json
import sys
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parent / "extension"
name = sys.argv[1]
base = {r["instance_id"]: r for r in map(json.loads, open(EXT / f"{name}_default/rows.jsonl")) if r["solver"] == "sa_qubo"}
bias = {r["instance_id"]: r for r in map(json.loads, open(EXT / f"{name}_default_bias2/rows.jsonl"))}
out = {}
for size in ("20", "50", "100", "all"):
    ids = [i for i in bias if size == "all" or str(bias[i]["n_customers"]) == size]
    d = np.array([bias[i]["gap_after"] - base[i]["gap_after"] for i in ids])
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
    out[size] = {"unbiased": round(float(np.mean([base[i]["gap_after"] for i in ids])), 2),
                 "bias_alpha2": round(float(np.mean([bias[i]["gap_after"] for i in ids])), 2),
                 "diff": round(float(d.mean()), 2), "ci95": [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]}
    print(name, size, out[size])
(EXT / f"bias2_{name}.json").write_text(json.dumps(out, indent=1) + "\n")
