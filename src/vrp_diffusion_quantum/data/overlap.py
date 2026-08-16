"""Content-based overlap checks for generated CVRP CSV datasets."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def instance_content_hashes(dataset_root: Path, n_customers: int) -> set[str]:
    """Return hashes of capacity, coordinates, and demands for every CSV instance.

    Generator metadata such as seed and distribution names is intentionally excluded: two
    instances overlap when their actual optimization inputs are identical.
    """
    prefix = dataset_root / f"cvrp{n_customers}"
    capacities: dict[str, str] = {}
    with prefix.with_name(f"{prefix.name}_instances.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            capacities[row["instance"]] = row["capacity"]

    hashes: set[str] = set()
    current_id: str | None = None
    digest: hashlib._Hash | None = None
    node_count = 0
    with prefix.with_name(f"{prefix.name}_nodes.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            instance_id = row["instance"]
            if instance_id != current_id:
                if digest is not None:
                    if node_count != n_customers + 1:
                        raise ValueError(f"instance {current_id} has {node_count} nodes")
                    hashes.add(digest.hexdigest())
                if instance_id not in capacities:
                    raise ValueError(f"missing metadata for instance {instance_id}")
                current_id = instance_id
                node_count = 0
                digest = hashlib.sha256()
                digest.update(f"capacity={capacities[instance_id]}\n".encode())
            assert digest is not None
            digest.update(
                f"{row['node_id']},{row['is_depot']},{row['x']},{row['y']},{row['demand']}\n".encode()
            )
            node_count += 1
    if digest is not None:
        if node_count != n_customers + 1:
            raise ValueError(f"instance {current_id} has {node_count} nodes")
        hashes.add(digest.hexdigest())
    if len(hashes) != len(capacities):
        raise ValueError(
            f"expected {len(capacities)} unique CVRP{n_customers} instances, got {len(hashes)}"
        )
    return hashes


def dataset_content_overlap(
    left_root: Path, right_root: Path, sizes: tuple[int, ...] = (20, 50, 100)
) -> dict[int, int]:
    """Count exact optimization-input overlaps between two generated datasets by size."""
    return {
        size: len(
            instance_content_hashes(left_root, size) & instance_content_hashes(right_root, size)
        )
        for size in sizes
    }
