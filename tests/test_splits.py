"""Tests for deterministic CVRP train/validation/test split creation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vrp_diffusion_quantum.data.dataset import make_example, save_example
from vrp_diffusion_quantum.data.splits import make_dataset_splits
from vrp_diffusion_quantum.data.types import CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.utils.feasibility import route_cost


def _write_examples(source: Path, *, n_customers: int, count: int) -> None:
    for index in range(count):
        coords = np.column_stack(
            [
                np.linspace(0.1, 0.9, n_customers + 1),
                np.linspace(0.9, 0.1, n_customers + 1),
            ]
        )
        instance = CVRPInstance(
            coords=coords,
            demands=np.concatenate([[0.0], np.ones(n_customers)]),
            capacity=float(n_customers),
            depot_index=0,
            instance_id=f"n{n_customers}_{index}",
            n_customers=n_customers,
            seed=index,
            generator_settings={"fixture": True},
        )
        routes = [list(range(n_customers))]
        solution = LabeledSolution(
            routes=routes,
            cost=route_cost(instance, routes),
            num_vehicles=1,
            feasible=True,
            solver_name="fixture",
            time_budget=None,
            seed=index,
            runtime_seconds=0.0,
        )
        save_example(
            make_example(instance, solution),
            source / f"cvrp{n_customers}_{index:04d}.json",
        )


def _split_files(manifest: dict[str, object], split_name: str) -> set[str]:
    splits = manifest["splits"]
    assert isinstance(splits, dict)
    split = splits[split_name]
    assert isinstance(split, dict)
    examples = split["examples"]
    assert isinstance(examples, list)
    return {str(entry["file"]) for entry in examples}


def test_make_dataset_splits_is_stratified_disjoint_and_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "examples"
    source.mkdir()
    _write_examples(source, n_customers=2, count=10)
    _write_examples(source, n_customers=3, count=10)

    first = make_dataset_splits(
        source,
        tmp_path / "splits_a",
        seed=42,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
    )
    second = make_dataset_splits(
        source,
        tmp_path / "splits_b",
        seed=42,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    train = _split_files(first, "train")
    val = _split_files(first, "val")
    test = _split_files(first, "test")
    assert len(train) == 12
    assert len(val) == 4
    assert len(test) == 4
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == {path.name for path in source.glob("*.json")}
    assert train == _split_files(second, "train")
    assert val == _split_files(second, "val")
    assert test == _split_files(second, "test")

    splits = first["splits"]
    assert isinstance(splits, dict)
    assert splits["train"]["counts_by_size"] == {"2": 6, "3": 6}
    second_splits = second["splits"]
    assert isinstance(second_splits, dict)
    for split_name in ("train", "val", "test"):
        assert splits[split_name]["sha256"] == second_splits[split_name]["sha256"]
    assert (tmp_path / "splits_a" / "split_manifest.json").is_file()
    assert first["materialization"] == "hardlink"

    first_train_file = next((tmp_path / "splits_a" / "train").glob("*.json"))
    source_file = source / first_train_file.name
    assert first_train_file.stat().st_ino == source_file.stat().st_ino


def test_make_dataset_splits_can_copy_independent_files(tmp_path: Path) -> None:
    source = tmp_path / "examples"
    source.mkdir()
    _write_examples(source, n_customers=2, count=3)

    manifest = make_dataset_splits(
        source,
        tmp_path / "copied_splits",
        seed=0,
        materialization="copy",
    )

    copied_file = next((tmp_path / "copied_splits" / "train").glob("*.json"))
    assert copied_file.read_bytes() == (source / copied_file.name).read_bytes()
    assert copied_file.stat().st_ino != (source / copied_file.name).stat().st_ino
    assert manifest["materialization"] == "copy"


def test_make_dataset_splits_rejects_bad_or_destructive_targets(tmp_path: Path) -> None:
    source = tmp_path / "examples"
    source.mkdir()
    _write_examples(source, n_customers=2, count=2)

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        make_dataset_splits(source, tmp_path / "bad_fraction", seed=0, train_fraction=0.8)
    with pytest.raises(ValueError, match="descendants"):
        make_dataset_splits(source, source / "splits", seed=0)
    with pytest.raises(ValueError, match="materialization"):
        make_dataset_splits(  # type: ignore[arg-type]
            source, tmp_path / "bad_materialization", seed=0, materialization="invalid"
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="not empty"):
        make_dataset_splits(source, occupied, seed=0)
