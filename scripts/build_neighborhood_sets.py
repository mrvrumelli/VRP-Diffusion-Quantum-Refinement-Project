"""Freeze a refinement neighborhood set from baseline solutions.

Input is a directory written by ``solve_with_baseline.py`` (one ``<instance>.json`` with the
baseline routes and one ``<instance>.m_prob.npz`` with the prior per instance). For each instance
the four selectors run on the baseline routes with the default ``NeighborhoodConfig``, and the
``--per-type`` highest-scoring neighborhoods of each type are kept. The output JSON stores the
instance file, the starting routes, the prior path and every neighborhood, together with SHA-256
hashes of all inputs, so experiments on the set are reproducible and the set cannot drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.quantum.neighborhoods import NeighborhoodConfig, select_neighborhoods
from vrp_diffusion_quantum.utils.feasibility import validate_routes

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--name", required=True, help="set name, e.g. development or validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=3)
    args = parser.parse_args()
    solutions = args.solutions if args.solutions.is_absolute() else ROOT / args.solutions
    config = NeighborhoodConfig()
    entries = []
    instances = []
    for record_path in sorted(solutions.glob("*.json")):
        if record_path.name == "summary.json":
            continue
        record = json.loads(record_path.read_text())
        example = load_example(ROOT / record["file"])
        routes = [list(route) for route in record["routes"] if route]
        if not validate_routes(example.instance, routes).feasible:
            raise ValueError(f"baseline routes infeasible for {record['instance_id']}")
        prior_path = record_path.with_name(f"{record['instance_id']}.m_prob.npz")
        m_prob = np.load(prior_path)["m_prob"].astype(np.float64)
        m_prob = np.clip((m_prob + m_prob.T) / 2.0, 0.0, 1.0)
        np.fill_diagonal(m_prob, 0.0)
        neighborhoods = select_neighborhoods(example.instance, routes, m_prob=m_prob, config=config)
        kept = []
        for neighborhood_type in (
            "high_cost_route",
            "uncertain_m",
            "low_confidence_edges",
            "two_route_exchange",
        ):
            of_type = [n for n in neighborhoods if n.neighborhood_type == neighborhood_type]
            kept += of_type[: args.per_type]
        instances.append(
            {
                "instance_id": record["instance_id"],
                "file": record["file"],
                "file_sha256": _sha256(ROOT / record["file"]),
                "n_customers": example.instance.n_customers,
                "routes": routes,
                "baseline_cost": record["cost"],
                "reference_cost": record["reference_cost"],
                "m_prob": str(prior_path.relative_to(ROOT).as_posix()),
                "m_prob_sha256": _sha256(prior_path),
            }
        )
        for index, neighborhood in enumerate(kept):
            entries.append(
                {
                    "id": f"{record['instance_id']}#{index}",
                    "instance_id": record["instance_id"],
                    **asdict(neighborhood),
                }
            )
    payload = {
        "name": args.name,
        "solutions": str(args.solutions),
        "neighborhood_config": asdict(config),
        "per_type": args.per_type,
        "instances": instances,
        "neighborhoods": entries,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1) + "\n")
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["neighborhood_type"]] = counts.get(entry["neighborhood_type"], 0) + 1
    print(json.dumps({"instances": len(instances), "neighborhoods": len(entries), **counts}))
    print("sha256", _sha256(output))


if __name__ == "__main__":
    main()
