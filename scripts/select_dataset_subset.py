"""Select a deterministic size-stratified subset of CVRPExample JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

from vrp_diffusion_quantum.data.subsets import select_example_subset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--per-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--materialization",
        choices=("hardlink", "copy"),
        default="hardlink",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = select_example_subset(
        args.source,
        args.output,
        sizes=args.sizes,
        per_size=args.per_size,
        seed=args.seed,
        materialization=args.materialization,
    )
    print(f"count={manifest['count']}")
    print(f"counts_by_size={manifest['counts_by_size']}")
    print(f"source_sha256={manifest['source_sha256']}")
    print(f"examples_sha256={manifest['examples_sha256']}")
    print(f"manifest={args.output.resolve() / 'subset_manifest.json'}")


if __name__ == "__main__":
    main()
