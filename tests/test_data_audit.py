"""Tests for the read-only dataset split audit."""

from __future__ import annotations

import json
from pathlib import Path

from test_splits import _write_examples
from vrp_diffusion_quantum.data.audit import audit_dataset_splits
from vrp_diffusion_quantum.data.splits import make_dataset_splits


def test_audit_dataset_splits_accepts_valid_manifest(tmp_path: Path) -> None:
    source = tmp_path / "examples"
    source.mkdir()
    _write_examples(source, n_customers=2, count=10)
    _write_examples(source, n_customers=3, count=10)
    splits = tmp_path / "splits"
    make_dataset_splits(
        source,
        splits,
        seed=42,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    report = audit_dataset_splits(splits, verify_hashes=True)

    assert report["valid"] is True
    assert report["total_examples"] == 20
    assert report["overlaps"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert report["splits"]["train"]["counts_by_size"] == {"2": 6, "3": 6}
    assert report["splits"]["train"]["computed_sha256"]
    assert report["total_logical_bytes"] > 0


def test_audit_dataset_splits_reports_manifest_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "examples"
    source.mkdir()
    _write_examples(source, n_customers=2, count=5)
    splits = tmp_path / "splits"
    make_dataset_splits(source, splits, seed=0)

    manifest_path = splits / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"]["count"] += 1
    manifest_path.write_text(json.dumps(manifest))

    report = audit_dataset_splits(splits)

    assert report["valid"] is False
    assert "train count does not match manifest" in report["errors"]
