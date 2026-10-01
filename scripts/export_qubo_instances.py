"""Export small refinement QUBOs from a frozen neighborhood set for QAOA screening.

Every neighborhood whose (unbiased) QUBO has at most ``--max-qubits`` variables is exported with
its matrix, offset, labels, exact minimum energy and the classical exact optimum of the
subproblem, so the quantum environment needs nothing from this package. ``--analyze`` instead
reads a QAOA result file, decodes and repairs the best sampled bitstrings, and scores their true
cost against the classical optimum.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.local_search.baselines import solve_exchange, solve_reorder
from vrp_diffusion_quantum.quantum.neighborhoods import (
    Neighborhood,
    evaluate_exchange,
    extract_exchange_subproblem,
    extract_reorder_subproblem,
)
from vrp_diffusion_quantum.quantum.qubo import solve_exact
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    build_exchange_qubo,
    decode_exchange,
    repair_exchange,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import (
    build_reorder_qubo,
    decode_reorder,
    repair_reorder,
)

ROOT = Path(__file__).resolve().parents[1]


def _neighborhood(entry: dict[str, Any]) -> Neighborhood:
    return Neighborhood(
        entry["neighborhood_type"],
        entry["kind"],
        tuple(entry["route_indices"]),
        tuple(entry["customers"]),
        float(entry["score"]),
        tuple(entry["segment"]) if entry["segment"] is not None else None,
    )


def _subproblem_and_qubo(
    record: dict[str, Any], entry: dict[str, Any]
) -> tuple[Any, Any, Any, float]:
    """Instance, subproblem, built QUBO wrapper and classical exact optimum cost."""
    instance = load_example(ROOT / record["file"]).instance
    neighborhood = _neighborhood(entry)
    if neighborhood.kind == "reorder":
        sub = extract_reorder_subproblem(instance, record["routes"], neighborhood)
        return instance, sub, build_reorder_qubo(sub), solve_reorder(sub, "exact").cost_after
    exchange = extract_exchange_subproblem(instance, record["routes"], neighborhood)
    optimum = solve_exchange(instance, exchange, "exhaustive").cost_after
    return instance, exchange, build_exchange_qubo(instance, exchange), optimum


def export(set_path: Path, output: Path, max_qubits: int) -> None:
    neighborhood_set = json.loads(set_path.read_text())
    records = {r["instance_id"]: r for r in neighborhood_set["instances"]}
    qubos = []
    for entry in neighborhood_set["neighborhoods"]:
        if entry["kind"] == "reorder" and len(entry["customers"]) ** 2 > max_qubits:
            continue
        record = records[entry["instance_id"]]
        _, _, wrapper, optimum = _subproblem_and_qubo(record, entry)
        if wrapper.qubo.num_variables > max_qubits:
            continue
        exact = solve_exact(wrapper.qubo)
        qubos.append(
            {
                "id": entry["id"],
                "kind": entry["kind"],
                "neighborhood_type": entry["neighborhood_type"],
                "matrix": wrapper.qubo.matrix.tolist(),
                "offset": wrapper.qubo.offset,
                "labels": list(wrapper.qubo.labels),
                "exact_energy": exact.energy,
                "classical_optimum_cost": optimum,
            }
        )
    output.write_text(json.dumps({"set": str(set_path), "max_qubits": max_qubits, "qubos": qubos}))
    kinds: dict[str, int] = defaultdict(int)
    for item in qubos:
        kinds[f"{item['kind']}:{len(item['labels'])}q"] += 1
    print(json.dumps({"exported": len(qubos), **dict(sorted(kinds.items()))}))


def analyze(set_path: Path, qubo_path: Path, results_path: Path, output: Path) -> None:
    neighborhood_set = json.loads(set_path.read_text())
    records = {r["instance_id"]: r for r in neighborhood_set["instances"]}
    entries = {e["id"]: e for e in neighborhood_set["neighborhoods"]}
    exported = {q["id"]: q for q in json.loads(qubo_path.read_text())["qubos"]}
    rows = []
    for result in json.loads(results_path.read_text())["results"]:
        entry = entries[result["id"]]
        instance, sub, wrapper, optimum = _subproblem_and_qubo(records[entry["instance_id"]], entry)
        best = result["samples"][0]  # lowest sampled energy
        if entry["kind"] == "reorder":
            decoded = decode_reorder(wrapper, best["x"])
            order = decoded if decoded is not None else repair_reorder(wrapper, best["x"])
            cost, valid = sub.path_cost(order), decoded is not None
        else:
            raw = decode_exchange(wrapper, best["x"])
            assignment = repair_exchange(wrapper, raw)
            valid = sub.is_feasible(raw)
            cost = (
                evaluate_exchange(instance, sub, assignment)[0]
                if assignment is not None
                else float("inf")
            )
        rows.append(
            {
                "id": result["id"],
                "kind": entry["kind"],
                "reps": result["reps"],
                "num_qubits": result["num_qubits"],
                "optimal_probability": result["optimal_probability"],
                "random_guess_optimal_probability": result["random_guess_optimal_probability"],
                "best_sample_is_qubo_optimal": result["best_sample_is_optimal"],
                "best_sample_valid_without_repair": bool(valid),
                "best_sample_true_cost": cost,
                "classical_optimum_cost": optimum,
                "matches_classical_optimum": bool(cost <= optimum + 1e-9),
                "exported_classical_optimum": exported[result["id"]]["classical_optimum_cost"],
            }
        )
    summary = {}
    for kind in ("reorder", "exchange"):
        for reps in sorted({r["reps"] for r in rows}):
            sel = [r for r in rows if r["kind"] == kind and r["reps"] == reps]
            if not sel:
                continue
            summary[f"{kind}|p={reps}"] = {
                "count": len(sel),
                "mean_qubits": float(np.mean([r["num_qubits"] for r in sel])),
                "mean_optimal_probability": float(np.mean([r["optimal_probability"] for r in sel])),
                "mean_random_guess_probability": float(
                    np.mean([r["random_guess_optimal_probability"] for r in sel])
                ),
                "best_sample_qubo_optimal_rate": float(
                    np.mean([r["best_sample_is_qubo_optimal"] for r in sel])
                ),
                "best_sample_matches_classical_optimum_rate": float(
                    np.mean([r["matches_classical_optimum"] for r in sel])
                ),
                "best_sample_valid_without_repair_rate": float(
                    np.mean([r["best_sample_valid_without_repair"] for r in sel])
                ),
            }
    output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--qubos", type=Path, required=True)
    parser.add_argument("--max-qubits", type=int, default=16)
    parser.add_argument("--analyze", type=Path, help="QAOA result file to decode and score")
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args()
    set_path = args.set if args.set.is_absolute() else ROOT / args.set
    qubo_path = args.qubos if args.qubos.is_absolute() else ROOT / args.qubos
    if args.analyze is None:
        export(set_path, qubo_path, args.max_qubits)
        return
    if args.analysis_output is None:
        parser.error("--analyze needs --analysis-output")
    analyze(set_path, qubo_path, ROOT / args.analyze, ROOT / args.analysis_output)


if __name__ == "__main__":
    main()
