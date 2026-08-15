"""Tests for deterministic size-stratified subset selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_splits import _write_examples
from vrp_diffusion_quantum.data.subsets import select_example_subset


def test_select_example_subset_is_stratified_reproducible_and_hardlinked(
    tmp_path: Path,
) -> None:
    source = tmp_path / "examples"
    source.mkdir()
    _write_examples(source, n_customers=2, count=10)
    _write_examples(source, n_customers=3, count=10)

    first = select_example_subset(source, tmp_path / "first", sizes=[2, 3], per_size=3, seed=42)
    second = select_example_subset(source, tmp_path / "second", sizes=[2, 3], per_size=3, seed=42)

    assert first["count"] == 6
    assert first["counts_by_size"] == {"2": 3, "3": 3}
    assert first["examples"] == second["examples"]
    assert first["examples_sha256"] == second["examples_sha256"]
    selected_file = tmp_path / "first" / first["examples"][0]["file"]
    assert selected_file.stat().st_ino == (source / selected_file.name).stat().st_ino


def test_select_example_subset_rejects_insufficient_or_occupied_output(tmp_path: Path) -> None:
    source = tmp_path / "examples"
    source.mkdir()
    _write_examples(source, n_customers=2, count=2)

    with pytest.raises(ValueError, match="only 2 are available"):
        select_example_subset(source, tmp_path / "too_many", sizes=[2], per_size=3, seed=0)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not empty"):
        select_example_subset(source, occupied, sizes=[2], per_size=1, seed=0)


def test_select_example_subset_ignores_dataset_manifests(tmp_path: Path) -> None:
    source = tmp_path / "examples"
    source.mkdir()
    _write_examples(source, n_customers=2, count=2)
    (source / "subset_manifest.json").write_text("{}")
    (source / "training_label_manifest.json").write_text("{}")

    result = select_example_subset(source, tmp_path / "selected", sizes=[2], per_size=1, seed=0)

    assert result["count"] == 1
