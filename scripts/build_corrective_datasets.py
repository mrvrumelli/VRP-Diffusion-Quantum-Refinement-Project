"""Rebuild source-count curves and materialize already-solved, single-seed HGS labels."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.data.single_reference import (
    materialize_single_hgs_reference,
    validate_single_reference_dataset,
)
from vrp_diffusion_quantum.data.subsets import select_source_subset

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-hgs", action="store_true")
    args = parser.parse_args()
    root = Path("data/processed/corrective_20260930")
    root.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(root / "run.log")],
    )
    excluded = {
        load_example(path).instance.instance_id
        for directory in ("s7799_val100_policy_v1", "s7799_test20_policy_v2_canonical")
        for path in (Path("data/processed") / directory).glob("*.json")
        if not path.name.endswith("manifest.json")
    }
    for size in (20, 50, 100):
        previous: set[str] = set()
        for count in (500, 1000, 2000):
            output = root / f"curve_n{size}_{count}"
            manifest_path = output / "subset_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text())
                if manifest_path.exists()
                else select_source_subset(
                    "data/processed/s7799_task5_labels_combined",
                    output,
                    sizes=[size],
                    per_size=count,
                    seed=74001,
                    excluded_source_ids=excluded,
                )
            )
            identities = {entry["instance_id"] for entry in manifest["examples"]}
            assert len(identities) == count and previous <= identities and not identities & excluded
            previous = identities
            LOGGER.info(
                "CVRP%d: %d sources, %d references", size, len(identities), manifest["count"]
            )
    if not args.single_hgs:
        return
    source = Path("data/processed/paper_cmd_hgs_full_50k")
    audit = Path("outputs/label_audit/paper_cmd_full_50k/candidates")
    assignments = []
    # Declare assignments using source filenames before reading any candidate or audit result.
    for size, train_count in ((20, 16667), (50, 16667), (100, 16666)):
        paths = sorted(source.glob(f"cvrp{size}_*.json"))
        rng = np.random.default_rng(np.random.SeedSequence([93017, size]))
        paths = [paths[int(index)] for index in rng.permutation(len(paths))]
        remainder = len(paths) - train_count
        if remainder < 2:
            raise ValueError("insufficient sources for declared split")
        for index, path in enumerate(paths):
            split = (
                "train"
                if index < train_count
                else ("val" if index < train_count + remainder // 2 else "test")
            )
            assignments.append((split, path))
    split_root = root / "single_hgs"
    split_root.mkdir(exist_ok=True)
    assignment_path = split_root / "source_assignments.json"
    frozen = {
        "seed": 93017,
        "base_seed": 50937,
        "assignments": [(split, path.name) for split, path in assignments],
    }
    if assignment_path.exists() and json.loads(assignment_path.read_text()) != json.loads(
        json.dumps(frozen)
    ):
        raise ValueError("source assignments changed")
    assignment_path.write_text(json.dumps(frozen, indent=2) + "\n")
    for split in ("train", "val", "test"):
        output = split_root / split
        manifest_path = output / "training_label_manifest.json"
        if manifest_path.exists():
            validate_single_reference_dataset(output)
            continue
        entries = []
        for name, path in assignments:
            if name != split:
                continue
            entry = materialize_single_hgs_reference(
                path,
                audit / path.stem / "pyvrp_seed_50937.json",
                output / path.name,
                base_seed=50937,
            )
            entries.append(entry)
            if len(entries) % 1000 == 0:
                LOGGER.info("single-HGS %s: %d", split, len(entries))
        manifest = {
            "schema_version": 2,
            "label_policy": "single_hgs_route_partition",
            "stability_filter": False,
            "base_seed": 50937,
            "split_seed": 93017,
            "examples": entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        validate_single_reference_dataset(output)
        LOGGER.info("Verified single-HGS %s: %d", split, len(entries))


if __name__ == "__main__":
    main()
