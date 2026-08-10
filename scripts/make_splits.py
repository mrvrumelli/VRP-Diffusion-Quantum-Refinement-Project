"""Create reproducible, size-stratified train/validation/test CVRP example splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from vrp_diffusion_quantum.data.splits import make_dataset_splits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="directory of CVRPExample JSONs")
    parser.add_argument("--output", type=Path, required=True, help="new/empty splits directory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = make_dataset_splits(
        args.source,
        args.output,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    split_summary = manifest["splits"]
    print(f"source_sha256={manifest['source_sha256']}")
    for split_name in ("train", "val", "test"):
        split = split_summary[split_name]
        print(
            f"{split_name}: count={split['count']} "
            f"counts_by_size={split['counts_by_size']} sha256={split['sha256']}"
        )
    print(f"manifest={args.output.resolve() / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
