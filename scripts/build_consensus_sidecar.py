"""Build a versioned training-only consensus target/confidence sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrp_diffusion_quantum.data.consensus_targets import (
    build_consensus_targets,
    save_consensus_sidecar,
)
from vrp_diffusion_quantum.data.dataset import load_dataset
from vrp_diffusion_quantum.utils.experiment import hash_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    examples = load_dataset(args.dataset)
    targets = build_consensus_targets(examples)
    manifest = save_consensus_sidecar(targets, args.output)
    manifest["source_dataset"] = str(args.dataset.resolve())
    manifest["source_dataset_sha256"] = hash_dataset(args.dataset)
    manifest_path = args.output / "consensus_manifest.json"
    persisted = json.loads(manifest_path.read_text())
    persisted.update(
        {
            "source_dataset": manifest["source_dataset"],
            "source_dataset_sha256": manifest["source_dataset_sha256"],
        }
    )
    manifest_path.write_text(json.dumps(persisted, indent=2) + "\n")
    print(
        json.dumps(
            {
                "target_count": manifest["target_count"],
                "source_dataset_sha256": manifest["source_dataset_sha256"],
                "sidecar_sha256_before_source_metadata": manifest["sidecar_sha256"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
