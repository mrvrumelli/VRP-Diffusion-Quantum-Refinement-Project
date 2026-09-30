"""Preserve compact results and provenance outside ignored experiment directories."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from run_corrective_evaluation import dataset_ids


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    destination = Path("docs/evidence/corrective_20260930")
    destination.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {}
    hashes = {}
    for root in sorted(Path("outputs").glob("corrective*_20260930")):
        for path in sorted(root.rglob("*.json")):
            if path.stat().st_size > 25_000_000:
                continue
            bundle[path.as_posix()] = json.loads(path.read_text())
            hashes[path.as_posix()] = digest(path)
        for path in sorted(root.rglob("summary.csv")):
            bundle[path.as_posix()] = path.read_text()
            hashes[path.as_posix()] = digest(path)
    data_root = Path("data/processed/corrective_20260930")
    for path in sorted(data_root.rglob("*manifest.json")):
        bundle[path.as_posix()] = json.loads(path.read_text())
        hashes[path.as_posix()] = digest(path)
    assignment = data_root / "single_hgs/source_assignments.json"
    bundle[assignment.as_posix()] = json.loads(assignment.read_text())
    hashes[assignment.as_posix()] = digest(assignment)
    reserved = bundle["outputs/corrective_20260930/reserved_test.json"]
    reserved_ids = {entry["instance_id"] for entry in reserved["examples"]}
    audited = {}
    for key, payload in bundle.items():
        if not isinstance(payload, dict):
            continue
        if key.endswith("panel.json"):
            for name, inventory in payload.get("training_inventories", {}).items():
                audited[name] = set(
                    inventory["source_ids"] if isinstance(inventory, dict) else inventory
                )
        if key.endswith("training_label_manifest.json") or key.endswith("subset_manifest.json"):
            audited[key] = {entry["instance_id"] for entry in payload["examples"]}
        if key.endswith("_training_inventory.json"):
            audited[key] = set(payload["source_ids"])
    for name in ("s7799_test20_policy_v2_canonical", "s7799_test20_policy_v2_source"):
        audited[name] = dataset_ids(Path("data/processed") / name)
    overlaps = {name: sorted(reserved_ids & ids) for name, ids in audited.items()}
    if any(overlaps.values()):
        raise ValueError("reserved test overlaps an audited training or previously scored source")
    bundle["reserved_test_audit"] = {
        "num_reserved": len(reserved_ids),
        "audited_source_counts": {name: len(ids) for name, ids in audited.items()},
        "overlaps": overlaps,
        "prediction_status": "reserved test not evaluated",
    }
    artifact = destination / "graph_results_and_manifests.json.gz"
    artifact.write_bytes(gzip.compress(json.dumps(bundle, separators=(",", ":")).encode(), mtime=0))
    code_paths = sorted(
        {
            *Path("src").rglob("*.py"),
            *Path("scripts").glob("*corrective*.py"),
            Path("eval/evaluate_cvrplib.py"),
            Path("scripts/pretrain_gat_encoder.py"),
            *Path("configs/train").glob("corrective*.yaml"),
        }
    )
    manifest = {
        "date": "2026-09-30",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_snapshot_scope": "end-of-execution working tree; not a clean committed start state",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "pyvrp", "ortools", "scipy", "pytest")
            },
            "training_device": "NVIDIA GeForce RTX 3060 Ti",
        },
        "code_sha256": {path.as_posix(): digest(path) for path in code_paths},
        "bundle": artifact.name,
        "bundle_sha256": digest(artifact),
        "bundle_bytes": artifact.stat().st_size,
        "original_artifact_sha256": hashes,
        "excluded": (
            "large checkpoints and prediction arrays remain in local outputs; no test scores"
        ),
    }
    (destination / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# Corrective execution evidence\n\n"
        "`summary.json` contains readable aggregate tables. "
        "`graph_results_and_manifests.json.gz` maps original paths to per-graph counts, "
        "costs, timings, source/panel manifests, diagnostic configurations and summaries. "
        "The bundle supports recalculating paired graph-bootstrap results without the models.\n\n"
        "`artifact_manifest.json` hashes these artifacts and the final code/configuration "
        "snapshot. "
        "It explicitly records a dirty working-tree snapshot, not an asserted clean start commit. "
        "Large probability arrays and checkpoints remain in the corresponding local outputs. "
        "The bundle does not include predictions for the reserved untouched test.\n\n"
        "`candidate_baseline.json` records the freeze decision and retained candidates.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
