"""Seed robustness: classical restarts and SA at default limits, seeds 0 (default arm), 1 and 2."""

import json
import sys
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parent / "extension"
name = sys.argv[1] if len(sys.argv) > 1 else "development"
runs = {0: EXT / f"{name}_default", 1: EXT / f"{name}_default_seed1", 2: EXT / f"{name}_default_seed2"}
data = {}
for seed, folder in runs.items():
    if (folder / "rows.jsonl").exists():
        data[seed] = {(r["instance_id"], r["solver"]): r for r in map(json.loads, open(folder / "rows.jsonl"))}


def ci(values, draws=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


ids = sorted({i for (i, s) in data[0] if s == "sa_qubo"})
report: dict = {"seeds": sorted(data)}
for solver in ("classical_restarts", "sa_qubo"):
    per_seed = {s: float(np.mean([d[(i, solver)]["gap_after"] for i in ids])) for s, d in data.items()}
    report[solver] = {
        "per_seed": {str(k): round(v, 2) for k, v in per_seed.items()},
        "mean": round(float(np.mean(list(per_seed.values()))), 2),
        "sd_across_seeds": round(float(np.std(list(per_seed.values()), ddof=1)), 2) if len(per_seed) > 1 else None,
    }
diffs = {}
for s, d in data.items():
    diff = np.array([d[(i, "sa_qubo")]["gap_after"] - d[(i, "classical_restarts")]["gap_after"] for i in ids])
    diffs[s] = diff
    report[f"sa_minus_restarts_seed{s}"] = [round(float(diff.mean()), 2), ci(diff)]
averaged = np.mean(np.stack(list(diffs.values())), axis=0)
report["sa_minus_restarts_seed_averaged"] = [round(float(averaged.mean()), 2), ci(averaged)]
(EXT / f"seeds_{name}.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report, indent=1))
