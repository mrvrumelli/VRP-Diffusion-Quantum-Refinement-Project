"""Build a deterministic, versioned follow-up subset from label-audit outcomes."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

from vrp_diffusion_quantum.utils.experiment import hash_dataset

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--both", type=int, default=20)
    parser.add_argument("--cost-only", type=int, default=10)
    parser.add_argument("--matrix-only", type=int, default=10)
    parser.add_argument("--accepted", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4301)
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _category(raw_reasons: str) -> str:
    reasons = set(ast.literal_eval(raw_reasons))
    cost = "pyvrp_cost_not_converged" in reasons
    matrix = "route_membership_ambiguous" in reasons
    if cost and matrix:
        return "both"
    if cost:
        return "cost_only"
    if matrix:
        return "matrix_only"
    return "accepted"


def prepare_followup(
    summary: Path,
    source: Path,
    output: Path,
    *,
    size: int,
    requested: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """Materialize a deterministic stratified subset without changing source files."""
    with summary.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["n_customers"]) == size]

    pools: dict[str, list[str]] = {name: [] for name in requested}
    for row in rows:
        category = _category(row["acceptance_reasons"])
        if category in pools:
            pools[category].append(row["source_file"])

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for category, count in requested.items():
        available = sorted(pools[category])
        if len(available) < count:
            raise ValueError(
                f"requested {count} {category} CVRP{size} examples, only {len(available)} exist"
            )
        for filename in sorted(rng.sample(available, count)):
            selected.append({"file": filename, "category": category})

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"follow-up output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    materialization = "hardlink"
    for item in selected:
        src = source / item["file"]
        dst = output / item["file"]
        if not src.is_file():
            raise FileNotFoundError(src)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
            materialization = "copy"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "summary": str(summary.resolve()),
        "source": str(source.resolve()),
        "source_sha256": hash_dataset(source),
        "size": size,
        "seed": seed,
        "requested_by_category": requested,
        "count": len(selected),
        "materialization": materialization,
        "examples": sorted(selected, key=lambda item: item["file"]),
    }
    (output / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = prepare_followup(
        _rooted(args.summary),
        _rooted(args.source),
        _rooted(args.output),
        size=args.size,
        requested={
            "both": args.both,
            "cost_only": args.cost_only,
            "matrix_only": args.matrix_only,
            "accepted": args.accepted,
        },
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
