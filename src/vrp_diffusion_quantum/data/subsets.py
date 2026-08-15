"""Deterministic, size-stratified subsets of modeling CVRPExample files."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vrp_diffusion_quantum.data.splits import SplitMaterialization, materialize_example
from vrp_diffusion_quantum.utils.experiment import hash_dataset

__all__ = ["select_example_subset"]

_SIZE_PATTERN = re.compile(r"^cvrp(?P<size>\d+)_.*\.json$")
_METADATA_FILENAMES = {"subset_manifest.json", "training_label_manifest.json"}


def _source_hash(source: Path) -> tuple[str, str]:
    parent_manifest = source.parent / "split_manifest.json"
    if parent_manifest.is_file():
        payload = json.loads(parent_manifest.read_text())
        split = (payload.get("splits") or {}).get(source.name) or {}
        recorded_hash = split.get("sha256")
        if recorded_hash:
            return str(recorded_hash), str(parent_manifest)
    return hash_dataset(source), "computed"


def select_example_subset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    sizes: list[int],
    per_size: int,
    seed: int,
    materialization: SplitMaterialization = "hardlink",
) -> dict[str, Any]:
    """Select a deterministic number of examples per customer size and write a manifest."""
    if per_size < 1:
        raise ValueError(f"per_size must be >= 1, got {per_size}")
    if not sizes or len(set(sizes)) != len(sizes) or any(size < 1 for size in sizes):
        raise ValueError("sizes must contain unique positive integers")
    if materialization not in ("copy", "hardlink"):
        raise ValueError("materialization must be 'copy' or 'hardlink'")

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"source directory does not exist: {source}")
    if source == output or source in output.parents:
        raise ValueError("output_dir must not be the source directory or one of its descendants")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    requested_sizes = set(sizes)
    candidates: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(source.glob("*.json")):
        if path.name in _METADATA_FILENAMES:
            continue
        match = _SIZE_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"cannot determine customer size from filename: {path.name}")
        size = int(match.group("size"))
        if size in requested_sizes:
            candidates[size].append(path)

    selected: list[tuple[int, Path]] = []
    for size in sorted(requested_sizes):
        available = candidates[size]
        if len(available) < per_size:
            raise ValueError(
                f"requested {per_size} CVRP{size} examples but only {len(available)} are available"
            )
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), size]))
        indices = sorted(int(index) for index in generator.permutation(len(available))[:per_size])
        selected.extend((size, available[index]) for index in indices)

    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for size, source_path in selected:
        target = output / source_path.name
        materialize_example(source_path, target, materialization)
        entries.append({"file": source_path.name, "n_customers": size})

    source_sha256, source_hash_origin = _source_hash(source)
    examples_sha256 = hash_dataset(output)
    manifest: dict[str, Any] = {
        "seed": int(seed),
        "source_dir": str(source),
        "source_sha256": source_sha256,
        "source_hash_origin": source_hash_origin,
        "sizes": sorted(requested_sizes),
        "per_size": int(per_size),
        "count": len(entries),
        "counts_by_size": {str(size): per_size for size in sorted(requested_sizes)},
        "materialization": materialization,
        "examples_sha256": examples_sha256,
        "examples": entries,
    }
    (output / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
