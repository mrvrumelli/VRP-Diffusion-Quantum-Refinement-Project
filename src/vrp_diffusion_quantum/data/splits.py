"""Deterministic, size-stratified train/validation/test splits for CVRP examples."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vrp_diffusion_quantum.utils.experiment import hash_dataset

__all__ = ["SplitMaterialization", "make_dataset_splits", "materialize_example"]

_SPLIT_NAMES = ("train", "val", "test")
SplitMaterialization = Literal["copy", "hardlink"]
_MATERIALIZATIONS = ("copy", "hardlink")


def _validate_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    fractions = (train_fraction, val_fraction, test_fraction)
    if any(not math.isfinite(value) or value <= 0.0 for value in fractions):
        raise ValueError(f"split fractions must be finite and positive, got {fractions}")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split fractions must sum to 1.0, got {sum(fractions):.12g}")


def _allocate_counts(total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate all examples with deterministic largest-remainder rounding."""
    raw = [total * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remainder_order = sorted(
        range(len(fractions)),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[: total - sum(counts)]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _example_identity(path: Path) -> tuple[int, str]:
    try:
        instance = json.loads(path.read_text())["instance"]
        return int(instance["n_customers"]), str(instance["instance_id"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid CVRPExample JSON: {path}") from exc


def _ensure_safe_output(source_dir: Path, output_dir: Path) -> None:
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("output_dir must not be the source directory or one of its descendants")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def materialize_example(source: Path, target: Path, method: SplitMaterialization) -> None:
    if method == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError as exc:
        raise OSError(
            f"could not hard-link {source} to {target}; use materialization='copy' when source "
            "and output are on different filesystems"
        ) from exc


def make_dataset_splits(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    train_fraction: float = 0.9,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    materialization: SplitMaterialization = "hardlink",
) -> dict[str, Any]:
    """Materialize example JSONs into deterministic, size-stratified split directories.

    The source directory must contain one modeling ``CVRPExample`` per top-level JSON file.
    Existing non-empty output directories are rejected to prevent accidental split mixing or
    overwrite. ``hardlink`` avoids duplicating file contents and requires source and output to be
    on the same filesystem; split files must then be treated as read-only. The returned manifest
    is also written to ``split_manifest.json``.
    """
    _validate_fractions(train_fraction, val_fraction, test_fraction)
    if materialization not in _MATERIALIZATIONS:
        raise ValueError(
            f"materialization must be one of {_MATERIALIZATIONS}, got {materialization!r}"
        )
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"source directory does not exist: {source}")
    _ensure_safe_output(source, output)

    paths = sorted(source.glob("*.json"))
    if not paths:
        raise ValueError(f"no CVRPExample JSON files found in {source}")

    by_size: dict[int, list[tuple[Path, str]]] = defaultdict(list)
    seen_instance_ids: set[str] = set()
    for path in paths:
        n_customers, instance_id = _example_identity(path)
        if instance_id in seen_instance_ids:
            raise ValueError(f"duplicate instance_id in source dataset: {instance_id}")
        seen_instance_ids.add(instance_id)
        by_size[n_customers].append((path, instance_id))

    output.mkdir(parents=True, exist_ok=True)
    split_entries: dict[str, list[dict[str, Any]]] = {name: [] for name in _SPLIT_NAMES}
    fractions = (train_fraction, val_fraction, test_fraction)
    for n_customers in sorted(by_size):
        size_examples = by_size[n_customers]
        generator = np.random.default_rng(np.random.SeedSequence([int(seed), n_customers]))
        order = generator.permutation(len(size_examples)).tolist()
        shuffled = [size_examples[int(index)] for index in order]
        train_count, val_count, _ = _allocate_counts(len(shuffled), fractions)
        boundaries = (0, train_count, train_count + val_count, len(shuffled))
        for split_index, split_name in enumerate(_SPLIT_NAMES):
            for source_path, instance_id in shuffled[
                boundaries[split_index] : boundaries[split_index + 1]
            ]:
                split_entries[split_name].append(
                    {
                        "file": source_path.name,
                        "instance_id": instance_id,
                        "n_customers": n_customers,
                    }
                )

    for split_name, entries in split_entries.items():
        split_dir = output / split_name
        split_dir.mkdir()
        for entry in entries:
            materialize_example(
                source / str(entry["file"]),
                split_dir / str(entry["file"]),
                materialization,
            )

    manifest: dict[str, Any] = {
        "seed": int(seed),
        "source_dir": str(source),
        "source_sha256": hash_dataset(source),
        "fractions": dict(zip(_SPLIT_NAMES, fractions, strict=True)),
        "materialization": materialization,
        "splits": {},
    }
    split_manifest = manifest["splits"]
    assert isinstance(split_manifest, dict)
    for split_name, entries in split_entries.items():
        counts_by_size: dict[str, int] = defaultdict(int)
        for entry in entries:
            counts_by_size[str(entry["n_customers"])] += 1
        split_manifest[split_name] = {
            "count": len(entries),
            "counts_by_size": dict(sorted(counts_by_size.items(), key=lambda item: int(item[0]))),
            "sha256": hash_dataset(output / split_name),
            "examples": entries,
        }

    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
