"""Q1/Q2: QAOA best shot vs classical local search and SA on the identical small subproblems.

Joins the QAOA screening analysis (decoded best shot per QUBO and depth) with the matched
single-subproblem experiment rows (same neighborhood ids) on the development set.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/phase6_20261001"

qaoa = json.loads((P6 / "qaoa_analysis_development.json").read_text())["rows"]
results = {(r["id"], r["reps"]): r for r in json.loads((P6 / "qaoa_results_development.json").read_text())["results"]}
experiment = defaultdict(dict)
for line in (P6 / "experiment_development/rows.jsonl").read_text().splitlines():
    row = json.loads(line)
    experiment[row["id"]][row["method"]] = row

report = {}
for kind in ("reorder", "exchange"):
    for reps in (1, 2, 3):
        rows = [r for r in qaoa if r["kind"] == kind and r["reps"] == reps and r["id"] in experiment]
        if not rows:
            continue
        block = {"subproblems": len(rows), "mean_qubits": round(float(np.mean([r["num_qubits"] for r in rows])), 1)}
        initial = np.array([experiment[r["id"]]["exact_reference"]["initial_cost"] for r in rows])
        optimum = np.array([r["classical_optimum_cost"] for r in rows])
        qaoa_cost = np.array([r["best_sample_true_cost"] for r in rows])
        qaoa_kept = np.minimum(qaoa_cost, initial)
        block["qaoa"] = {
            "hit_rate": round(float(np.mean(qaoa_cost <= optimum + 1e-9)), 3),
            "kept_excess_percent": round(float(np.mean(100 * (qaoa_kept - optimum) / optimum)), 3),
            "mean_delta_cost_percent": round(float(np.mean(100 * (initial - qaoa_kept) / initial)), 3),
            "worse_than_start_before_acceptance": round(float(np.mean(qaoa_cost > initial + 1e-9)), 3),
            "valid_without_repair": round(float(np.mean([r["best_sample_valid_without_repair"] for r in rows])), 3),
            "shots": 1024,
            "mean_energy_evaluations": round(float(np.mean([results[(r["id"], reps)]["energy_evaluations"] for r in rows])), 0),
            "mean_simulation_seconds": round(float(np.mean([results[(r["id"], reps)]["runtime_seconds"] for r in rows])), 2),
        }
        for method in ("classical_ls", "classical_ls_budget", "sa_qubo", "sa_qubo_bias"):
            cost = np.array([experiment[r["id"]][method]["cost"] for r in rows])
            kept = np.minimum(cost, initial)
            block[method] = {
                "hit_rate": round(float(np.mean(cost <= optimum + 1e-9)), 3),
                "kept_excess_percent": round(float(np.mean(100 * (kept - optimum) / optimum)), 3),
                "mean_delta_cost_percent": round(float(np.mean(100 * (initial - kept) / initial)), 3),
                "mean_seconds": round(float(np.mean([experiment[r["id"]][method]["seconds"] for r in rows])), 4),
            }
        block["optimum_mean_delta_cost_percent"] = round(float(np.mean(100 * (initial - optimum) / initial)), 3)
        report[f"{kind}|p={reps}"] = block
(P6 / "qaoa_vs_classical_development.json").write_text(json.dumps(report, indent=1) + "\n")
for key, block in report.items():
    print(f"== {key} n={block['subproblems']} qubits={block['mean_qubits']} optimum delta={block['optimum_mean_delta_cost_percent']}%")
    for method in ("qaoa", "classical_ls", "classical_ls_budget", "sa_qubo", "sa_qubo_bias"):
        print(f"   {method:20s} {json.dumps(block[method])}")
