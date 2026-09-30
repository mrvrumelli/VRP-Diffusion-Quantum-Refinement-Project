"""Predetermined-seed HGS labels with explicit, verifiable provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from vrp_diffusion_quantum.data.dataset import load_example, make_example, save_example
from vrp_diffusion_quantum.data.types import LabeledSolution
from vrp_diffusion_quantum.utils.feasibility import validate_routes


def materialize_single_hgs_reference(
    source: Path, candidate_path: Path, destination: Path, *, base_seed: int
) -> dict[str, Any]:
    """Use exactly the requested solve, regardless of competing seed costs or stability."""
    candidate = json.loads(candidate_path.read_text())
    example = load_example(source)
    if (
        candidate["solver_name"] != "pyvrp"
        or candidate["base_seed"] != base_seed
        or candidate["instance_id"] != example.instance.instance_id
    ):
        raise ValueError("candidate solver, seed or source mismatch")
    if not candidate["feasible"] or candidate.get("violations"):
        raise ValueError("predetermined candidate is infeasible; do not substitute another seed")
    routes = candidate["routes"]
    feasibility = validate_routes(example.instance, routes)
    if not feasibility.feasible:
        raise ValueError("candidate fails independent feasibility validation")
    policy = {
        "mode": "single_hgs_route_partition",
        "base_seed": base_seed,
        "source_file": source.name,
        "candidate_route_hash": candidate["route_hash"],
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "reference_count": 1,
        "reference_index": 0,
        "stability_filter": False,
    }
    instance = replace(
        example.instance,
        generator_settings={**example.instance.generator_settings, "training_label_policy": policy},
    )
    solution = LabeledSolution(
        routes=routes,
        cost=float(candidate["cost"]),
        num_vehicles=len(routes),
        feasible=True,
        solver_name="single_reference_pyvrp",
        time_budget=float(candidate["time_budget"]),
        seed=int(candidate["derived_seed"]),
        runtime_seconds=float(candidate["runtime_seconds"]),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_example(make_example(instance, solution), destination)
    return {
        "file": destination.name,
        "instance_id": instance.instance_id,
        "n_customers": instance.n_customers,
        "base_seed": base_seed,
        "candidate_sha256": policy["candidate_sha256"],
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def validate_single_reference_dataset(directory: Path) -> None:
    """Reject metadata-only label claims before a new paper-mode training job starts."""
    manifest_path = directory / "training_label_manifest.json"
    if not manifest_path.exists():
        raise ValueError("single-HGS training requires a verified training_label_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("label_policy") != "single_hgs_route_partition"
        or manifest.get("stability_filter") is not False
    ):
        raise ValueError("dataset does not establish unfiltered single-HGS provenance")
    identities: set[str] = set()
    listed: set[str] = set()
    for entry in manifest["examples"]:
        path = directory / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"label hash mismatch: {path.name}")
        example = load_example(path)
        identity = example.instance.instance_id
        policy = example.instance.generator_settings.get("training_label_policy", {})
        if identity in identities or identity != entry["instance_id"]:
            raise ValueError("duplicate or mismatched source identity")
        if (
            policy.get("mode") != "single_hgs_route_partition"
            or policy.get("base_seed") != manifest["base_seed"]
            or policy.get("stability_filter") is not False
            or policy.get("reference_count") != 1
            or entry.get("base_seed") != manifest["base_seed"]
            or policy.get("candidate_sha256") != entry.get("candidate_sha256")
        ):
            raise ValueError("per-example label provenance mismatch")
        identities.add(identity)
        listed.add(path.name)
    actual = {
        path.name for path in directory.glob("*.json") if not path.name.endswith("manifest.json")
    }
    if listed != actual or not listed:
        raise ValueError("manifest does not cover exactly the nonempty dataset")
