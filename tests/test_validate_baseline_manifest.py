"""Tests for the frozen classical-baseline manifest validator."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from vrp_diffusion_quantum.utils.experiment import hash_dataset

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "validate_baseline_manifest",
    root / "scripts" / "validate_baseline_manifest.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load validate_baseline_manifest.py")
module = importlib.util.module_from_spec(spec)
sys.modules["validate_baseline_manifest"] = module
spec.loader.exec_module(module)

from validate_baseline_manifest import validate_manifest  # noqa: E402


def _write_manifest(tmp_path: Path, *, bad_artifact_hash: bool = False) -> Path:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"checkpoint")
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "example.json").write_text("{}")
    artifact_entry = {"path": artifact.name, "sha256": sha256}
    manifest = {
        "schema_version": 1,
        "shared_gat": {
            **artifact_entry,
            "sha256": "0" * 64 if bad_artifact_hash else sha256,
        },
        "sizes": {
            size: {
                "selected_solver": "diffusion",
                "diffusion": artifact_entry,
            }
            for size in (20, 50, 100)
        },
        "datasets": {
            "test": {
                "path": dataset.name,
                "hash": hash_dataset(dataset),
            }
        },
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def test_validate_manifest_accepts_matching_files_and_dataset(tmp_path: Path) -> None:
    rows = validate_manifest(_write_manifest(tmp_path), root=tmp_path)
    assert len(rows) == 5


def test_validate_manifest_rejects_changed_artifact(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, bad_artifact_hash=True)
    with pytest.raises(ValueError, match="shared_gat SHA-256 mismatch"):
        validate_manifest(manifest, root=tmp_path)
