"""Audit CVRP split counts, overlap, manifest metadata, hashes, and logical size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrp_diffusion_quantum.data.audit import audit_dataset_splits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="read all split bytes and compare computed hashes with the manifest",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = audit_dataset_splits(args.splits, verify_hashes=args.verify_hashes)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
