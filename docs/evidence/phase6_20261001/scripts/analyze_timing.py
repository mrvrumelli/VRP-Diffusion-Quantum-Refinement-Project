"""Clean timing pass: wall-clock refinement time per graph by configuration and size.

Adds the baseline's measured end-to-end time (prior + policy on GPU, from the freeze record) to give
total time per graph.
"""

import json
from pathlib import Path

import numpy as np

P6 = Path(__file__).resolve().parent
TIMING = P6 / "timing"
BASELINE_SECONDS = {20: 0.78, 50: 0.86, 100: 1.06}  # freeze record, development panel
CONFIGS = [
    ("T1 classical, one pass, 5/8/10", "t1_t10", "classical"),
    ("T2 classical, 10 restarts, 5/8/30", "t2_t4_t5", "classical_restarts"),
    ("T3 classical, 10 restarts, 10/16/30", "t3", "classical_restarts"),
    ("T4 SA on QUBO, 5/8/30", "t2_t4_t5", "sa_qubo"),
    ("T5 SA then classical restarts, 5/8/30", "t2_t4_t5", "sa_qubo+classical_restarts"),
    ("T10 classical exact, 4/6/10", "exact_small", "classical_exact"),
    ("T6 QAOA depth 1, 4/6/10", "t6_t7", "qaoa"),
    ("T7 random QUBO sampling, 4/6/10", "t6_t7", "random_qubo"),
]


def main() -> None:
    report = {}
    for label, folder, solver in CONFIGS:
        path = TIMING / folder / "rows.jsonl"
        if not path.exists():
            continue
        rows = [r for r in map(json.loads, open(path)) if r["solver"] == solver]
        block = {}
        for n in (20, 50, 100):
            sel = [r for r in rows if r["n_customers"] == n]
            wall = float(np.mean([r["wall_seconds"] for r in sel]))
            block[str(n)] = {
                "graphs": len(sel),
                "refinement_wall_seconds": round(wall, 1),
                "end_to_end_seconds": round(wall + BASELINE_SECONDS[n], 1),
                "gap_before": round(float(np.mean([r["gap_before"] for r in sel])), 2),
                "gap_after": round(float(np.mean([r["gap_after"] for r in sel])), 2),
            }
        report[label] = block
    (TIMING / "analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    print("| Configuration | N20 s | N50 s | N100 s |")
    print("|---|---:|---:|---:|")
    for label, block in report.items():
        cells = " | ".join(f"{block[n]['refinement_wall_seconds']:.1f}" for n in ("20", "50", "100"))
        print(f"| {label} | {cells} |")


if __name__ == "__main__":
    main()
