"""Materialize a mixed per-size training-label policy from a strong audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrp_diffusion_quantum.data.training_labels import (
    TrainingLabelPolicy,
    materialize_training_labels,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--competitive-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--profile", choices=("train", "canonical", "validation"), default="train"
    )
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = _parse_args()
    if args.profile == "validation":
        modes = {20: "original", 50: "accepted_canonical", 100: "accepted_canonical"}
    elif args.profile == "canonical":
        modes = {20: "multi_reference", 50: "multi_reference", 100: "multi_reference"}
    else:
        modes = {20: "original", 50: "canonical_else_multi", 100: "multi_reference"}
    tolerance = 0.0 if args.profile == "canonical" else args.competitive_tolerance
    manifest = materialize_training_labels(
        _rooted(args.source),
        _rooted(args.audit),
        _rooted(args.output),
        policy=TrainingLabelPolicy(
            modes_by_size=modes,
            competitive_relative_tolerance=tolerance,
        ),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
