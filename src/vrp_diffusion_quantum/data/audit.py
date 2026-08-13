"""Read-only audits for materialized CVRP train/validation/test splits."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from vrp_diffusion_quantum.utils.experiment import hash_dataset

__all__ = ["audit_dataset_splits"]

_SPLIT_NAMES = ("train", "val", "test")
_SIZE_PATTERN = re.compile(r"^cvrp(?P<size>\d+)_.*\.json$")


def _scan_split(split_dir: Path) -> tuple[set[str], dict[str, int], int]:
    files: set[str] = set()
    counts_by_size: dict[str, int] = defaultdict(int)
    logical_bytes = 0
    for path in split_dir.glob("*.json"):
        files.add(path.name)
        match = _SIZE_PATTERN.match(path.name)
        size = match.group("size") if match else "unknown"
        counts_by_size[size] += 1
        logical_bytes += path.stat().st_size
    return files, dict(sorted(counts_by_size.items())), logical_bytes


def audit_dataset_splits(
    splits_dir: str | Path,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Audit split existence, counts, overlap, manifest metadata, and logical byte size.

    Hash verification reads every byte and is therefore optional. Without it, the report includes
    the hashes recorded when the split was created and still validates counts and membership.
    """
    root = Path(splits_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"split directory does not exist: {root}")

    manifest_path = root / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    errors: list[str] = []
    file_sets: dict[str, set[str]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    manifest_splits = manifest.get("splits") or {}
    for split_name in _SPLIT_NAMES:
        split_dir = root / split_name
        if not split_dir.is_dir():
            errors.append(f"missing split directory: {split_name}")
            file_sets[split_name] = set()
            continue

        files, counts_by_size, logical_bytes = _scan_split(split_dir)
        file_sets[split_name] = files
        recorded = manifest_splits.get(split_name) or {}
        report: dict[str, Any] = {
            "count": len(files),
            "counts_by_size": counts_by_size,
            "logical_bytes": logical_bytes,
            "recorded_sha256": recorded.get("sha256"),
        }
        if verify_hashes:
            report["computed_sha256"] = hash_dataset(split_dir)
            if report["computed_sha256"] != report["recorded_sha256"]:
                errors.append(f"{split_name} hash does not match manifest")
        if len(files) != recorded.get("count"):
            errors.append(f"{split_name} count does not match manifest")
        if counts_by_size != recorded.get("counts_by_size"):
            errors.append(f"{split_name} per-size counts do not match manifest")
        split_reports[split_name] = report

    overlaps: dict[str, int] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        count = len(file_sets[left] & file_sets[right])
        overlaps[f"{left}_{right}"] = count
        if count:
            errors.append(f"{left} and {right} overlap by {count} file(s)")

    source_dir_value = manifest.get("source_dir")
    source_dir = Path(source_dir_value) if source_dir_value else None
    source_exists = bool(source_dir and source_dir.is_dir())
    if not source_exists:
        errors.append("manifest source directory does not exist")

    return {
        "valid": not errors,
        "splits_dir": str(root),
        "materialization": manifest.get("materialization", "legacy-copy"),
        "source_dir": str(source_dir) if source_dir else None,
        "source_exists": source_exists,
        "source_sha256": manifest.get("source_sha256"),
        "verify_hashes": verify_hashes,
        "splits": split_reports,
        "overlaps": overlaps,
        "total_examples": sum(report["count"] for report in split_reports.values()),
        "total_logical_bytes": sum(report["logical_bytes"] for report in split_reports.values()),
        "errors": errors,
    }
