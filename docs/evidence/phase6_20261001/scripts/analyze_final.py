"""Final test analysis: the pre-declared comparisons of docs/phase6_final_test_protocol_2026-10-02.md."""

import json
import sys
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parent / "final"
CONFIG_SOURCES = {
    "T1": ("t1", "classical"),
    "T2": ("t2_t4_t5", "classical_restarts"),
    "T4": ("t2_t4_t5", "sa_qubo"),
    "T5": ("t2_t4_t5", "sa_qubo+classical_restarts"),
    "T3": ("t3", "classical_restarts"),
    "T8": ("t6_t9", "classical_restarts"),
    "T10": ("t6_t9", "classical_exact"),
    "T6": ("t6_t9", "qaoa"),
    "T7": ("t6_t9", "random_qubo"),
    "T9": ("t6_t9", "qaoa+classical_restarts"),
}
COMPARISONS = [
    ("T1", None, "refinement value: T1 minus baseline"),
    ("T3", None, "refinement value: T3 minus baseline"),
    ("T4", "T2", "SA minus classical restarts, limits 5/8/30"),
    ("T5", "T2", "Q5: SA + classical polish minus classical restarts"),
    ("T6", "T7", "QAOA minus random sampling"),
    ("T6", "T10", "QAOA minus classical exact"),
    ("T6", "T8", "QAOA minus classical restarts"),
    ("T9", "T8", "QAOA + classical polish minus classical restarts"),
]


def ci(values, draws=10000):
    values = np.asarray(values, float)
    means = values[np.random.default_rng(0).integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def main() -> None:
    name = sys.argv[1]
    tables = {}
    for config, (folder, solver) in CONFIG_SOURCES.items():
        path = FINAL / f"{name}_{folder}" / "rows.jsonl"
        if path.exists():
            rows = {r["instance_id"]: r for r in map(json.loads, open(path)) if r["solver"] == solver}
            if rows:
                tables[config] = rows
    any_table = next(iter(tables.values()))
    ids = sorted(any_table)
    report: dict = {"set": name, "graphs": len(ids), "configs": {}, "comparisons": {}}
    for size in ("20", "50", "100", "all"):
        sel = [i for i in ids if size == "all" or str(any_table[i]["n_customers"]) == size]
        report["configs"][size] = {
            "baseline": round(float(np.mean([any_table[i]["gap_before"] for i in sel])), 2),
            **{
                c: {
                    "gap": round(float(np.mean([t[i]["gap_after"] for i in sel])), 2),
                    "improved": int(sum(t[i]["gap_after"] < t[i]["gap_before"] - 1e-9 for i in sel)),
                    "routes": round(float(np.mean([t[i]["routes_after"] for i in sel])), 2),
                    "solver_seconds": round(float(np.mean([t[i]["solver_seconds"] for i in sel])), 1),
                    "skipped_too_large": int(sum(t[i].get("skipped_too_large", 0) for i in sel)),
                    "attempted": int(sum(t[i]["attempted_steps"] for i in sel)),
                }
                for c, t in tables.items()
            },
        }
        block = {}
        for first, second, label in COMPARISONS:
            if first not in tables or (second is not None and second not in tables):
                continue
            if second is None:
                diff = [tables[first][i]["gap_after"] - tables[first][i]["gap_before"] for i in sel]
            else:
                diff = [tables[first][i]["gap_after"] - tables[second][i]["gap_after"] for i in sel]
            block[label] = [round(float(np.mean(diff)), 2), ci(diff)]
        report["comparisons"][size] = block
    out = FINAL / f"analysis_{name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["configs"]["all"], indent=1))
    print(json.dumps(report["comparisons"]["all"], indent=1))


if __name__ == "__main__":
    main()
