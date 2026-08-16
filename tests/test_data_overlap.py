from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vrp_diffusion_quantum.data.overlap import dataset_content_overlap, instance_content_hashes


def _write_dataset(root: Path, x: float, *, nodes: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for size in (2,):
        with (root / f"cvrp{size}_instances.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["instance", "n_customers", "capacity"])
            writer.writerow([0, size, 10.0])
        with (root / f"cvrp{size}_nodes.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["instance", "node_id", "is_depot", "x", "y", "demand"])
            for node in range(nodes):
                writer.writerow([0, node, int(node == 0), x + node, 0.0, int(node > 0)])


def test_dataset_content_overlap_uses_optimization_inputs(tmp_path: Path) -> None:
    left, same, different = tmp_path / "left", tmp_path / "same", tmp_path / "different"
    _write_dataset(left, 0.1)
    _write_dataset(same, 0.1)
    _write_dataset(different, 0.2)
    assert dataset_content_overlap(left, same, sizes=(2,)) == {2: 1}
    assert dataset_content_overlap(left, different, sizes=(2,)) == {2: 0}


def test_instance_content_hashes_rejects_wrong_node_count(tmp_path: Path) -> None:
    _write_dataset(tmp_path, 0.1, nodes=2)
    with pytest.raises(ValueError, match="has 2 nodes"):
        instance_content_hashes(tmp_path, 2)
