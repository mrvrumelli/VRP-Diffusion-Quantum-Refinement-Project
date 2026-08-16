"""Check exact CVRP instance overlap between two generated CSV datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrp_diffusion_quantum.data.overlap import dataset_content_overlap, instance_content_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    overlap = dataset_content_overlap(args.left, args.right)
    report = {
        "schema_version": 1,
        "left": args.left.as_posix(),
        "right": args.right.as_posix(),
        "instance_counts": {
            "left": {str(size): len(instance_content_hashes(args.left, size)) for size in overlap},
            "right": {
                str(size): len(instance_content_hashes(args.right, size)) for size in overlap
            },
        },
        "exact_content_overlap": {str(size): count for size, count in overlap.items()},
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if any(overlap.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
