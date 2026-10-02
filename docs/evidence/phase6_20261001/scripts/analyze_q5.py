"""Q5: does a QUBO (SA) stage add value once classical polishing runs to convergence?

CMD + LS:       extension/<set>_default, solver classical or classical_restarts (from baseline).
CMD + SA + LS:  extension/<set>_default sa_qubo routes, then extension/<set>_sa_then_classical with
                the same classical solver, both loops run until a round brings no improvement.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "outputs/phase6_20261001/extension"


def ci(values, seed=0, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def load(path):
    return {(r["instance_id"], r["solver"]): r for r in map(json.loads, open(path))}


name = sys.argv[1] if len(sys.argv) > 1 else "development"
base = load(EXT / f"{name}_default/rows.jsonl")
polish = load(EXT / f"{name}_sa_then_classical/rows.jsonl")
report = {}
for ls in ("classical", "classical_restarts"):
    for size in ("20", "50", "100", "all"):
        ids = sorted({i for (i, s), r in base.items() if s == ls and (size == "all" or str(r["n_customers"]) == size)})
        before = np.array([base[(i, ls)]["gap_before"] for i in ids])
        ls_only = np.array([base[(i, ls)]["gap_after"] for i in ids])
        sa_only = np.array([base[(i, "sa_qubo")]["gap_after"] for i in ids])
        sa_ls = np.array([polish[(i, ls)]["gap_after"] for i in ids])
        t_ls = np.array([base[(i, ls)]["solver_seconds"] for i in ids])
        t_sa_ls = np.array([base[(i, "sa_qubo")]["solver_seconds"] + polish[(i, ls)]["solver_seconds"] for i in ids])
        diff = sa_ls - ls_only
        report[f"{ls}|N{size}"] = {
            "instances": len(ids),
            "gap_before": round(float(before.mean()), 2),
            "cmd_ls": round(float(ls_only.mean()), 2),
            "cmd_sa": round(float(sa_only.mean()), 2),
            "cmd_sa_ls": round(float(sa_ls.mean()), 2),
            "sa_ls_minus_ls_pp": [round(float(diff.mean()), 2), ci(diff)],
            "sa_ls_better": int(np.sum(diff < -1e-9)),
            "ls_better": int(np.sum(diff > 1e-9)),
            "tied": int(np.sum(np.abs(diff) <= 1e-9)),
            "polish_gain_on_sa_pp": round(float((sa_only - sa_ls).mean()), 2),
            "solver_seconds_ls": round(float(t_ls.mean()), 1),
            "solver_seconds_sa_ls": round(float(t_sa_ls.mean()), 1),
        }
(EXT / f"q5_{name}.json").write_text(json.dumps(report, indent=1) + "\n")
for key, value in report.items():
    print(key, json.dumps(value))
