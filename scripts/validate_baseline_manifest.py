"""Validate the files and dataset hashes frozen by a classical-baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from vrp_diffusion_quantum.utils.experiment import hash_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "eval" / "classical_baseline_v1_candidate.yaml"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _expected_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, *, field: str) -> Mapping[object, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def validate_manifest(manifest_path: Path, *, root: Path = ROOT) -> list[str]:
    """Return human-readable verification rows or raise on a missing/mismatched artifact."""
    raw = yaml.safe_load(manifest_path.read_text())
    manifest = _mapping(raw, field="manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must be 1")

    rows: list[str] = []
    artifact_entries: list[tuple[str, Mapping[object, Any]]] = [
        ("shared_gat", _mapping(manifest.get("shared_gat"), field="shared_gat"))
    ]
    sizes = _mapping(manifest.get("sizes"), field="sizes")
    for size in (20, 50, 100):
        size_entry = sizes.get(size, sizes.get(str(size)))
        size_config = _mapping(size_entry, field=f"sizes.{size}")
        selected_solver = size_config.get("selected_solver")
        if selected_solver not in {"diffusion", "policy"}:
            raise ValueError(f"sizes.{size}.selected_solver must be diffusion or policy")
        artifact_entries.append(
            (
                f"sizes.{size}.diffusion",
                _mapping(size_config.get("diffusion"), field=f"sizes.{size}.diffusion"),
            )
        )
        policy = size_config.get("policy")
        if policy is not None:
            artifact_entries.append(
                (f"sizes.{size}.policy", _mapping(policy, field=f"sizes.{size}.policy"))
            )

    for label, entry in artifact_entries:
        path = _resolve(root, entry.get("path"), field=f"{label}.path")
        expected = _expected_hash(entry.get("sha256"), field=f"{label}.sha256")
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact not found: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
        rows.append(f"artifact {label}: ok ({actual})")

    datasets = _mapping(manifest.get("datasets"), field="datasets")
    if not datasets:
        raise ValueError("datasets must not be empty")
    for name, value in datasets.items():
        entry = _mapping(value, field=f"datasets.{name}")
        path = _resolve(root, entry.get("path"), field=f"datasets.{name}.path")
        expected = _expected_hash(entry.get("hash"), field=f"datasets.{name}.hash")
        actual = hash_dataset(path)
        if actual != expected:
            raise ValueError(f"datasets.{name} hash mismatch: expected {expected}, got {actual}")
        rows.append(f"dataset {name}: ok ({actual})")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    rows = validate_manifest(manifest, root=args.root.resolve())
    print("\n".join(rows))
    print(f"validated {len(rows)} frozen entries")


if __name__ == "__main__":
    main()
