"""QAOA screening analysis against uniform random sampling at the same shot count.

Reads the screening results and the decoded analysis written by export_qubo_instances.py --analyze.
Per kind, depth and qubit bucket it reports the optimal-state probability of the optimised QAOA
state, the same probability for uniform random guessing, their ratio, and the chance that the best
of ``shots`` samples is QUBO-optimal for QAOA and for uniform sampling.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SHOTS = 1024


def bucket(n: int) -> str:
    return "1-4" if n <= 4 else "5-8" if n <= 8 else "9-12" if n <= 12 else "13-16"


def main() -> None:
    results = json.loads(Path(sys.argv[1]).read_text())["results"]
    analysis = {(r["id"], r["reps"]): r for r in json.loads(Path(sys.argv[2]).read_text())["rows"]}
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in results:
        decoded = analysis[(row["id"], row["reps"])]
        merged = {**row, **decoded}
        groups[(row["kind"], row["reps"], bucket(row["num_qubits"]))].append(merged)
        groups[(row["kind"], row["reps"], "all")].append(merged)
    report = {}
    for (kind, reps, size), rows in sorted(groups.items()):
        p_opt = np.array([r["optimal_probability"] for r in rows])
        p_rand = np.array([r["random_guess_optimal_probability"] for r in rows])
        report[f"{kind}|p={reps}|q={size}"] = {
            "count": len(rows),
            "mean_qubits": round(float(np.mean([r["num_qubits"] for r in rows])), 2),
            "mean_p_opt": round(float(p_opt.mean()), 4),
            "mean_p_random": round(float(p_rand.mean()), 5),
            "median_amplification": round(float(np.median(p_opt / p_rand)), 2),
            "share_below_random": round(float(np.mean(p_opt < p_rand)), 3),
            "best_of_shots_optimal_qaoa": round(float(np.mean([r["best_sample_is_qubo_optimal"] for r in rows])), 3),
            "best_of_shots_optimal_random_expected": round(float(np.mean(1 - (1 - p_rand) ** SHOTS)), 3),
            "best_sample_true_optimum_rate": round(float(np.mean([r["matches_classical_optimum"] for r in rows])), 3),
            "best_sample_valid_without_repair": round(float(np.mean([r["best_sample_valid_without_repair"] for r in rows])), 3),
            "mean_runtime_seconds": round(float(np.mean([r["runtime_seconds"] for r in rows])), 2),
            "mean_energy_evaluations": round(float(np.mean([r["energy_evaluations"] for r in rows])), 1),
        }
    Path(sys.argv[3]).write_text(json.dumps(report, indent=1) + "\n")
    for key, value in report.items():
        print(key, json.dumps(value))


if __name__ == "__main__":
    main()
