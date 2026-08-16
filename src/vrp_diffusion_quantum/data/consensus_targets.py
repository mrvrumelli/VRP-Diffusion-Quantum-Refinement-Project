"""Training-only consensus targets without changing binary ``CVRPExample`` labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.training_labels import training_source_id
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.utils.experiment import hash_dataset


@dataclass(frozen=True)
class ConsensusTarget:
    """Mean route-membership target and unanimity-derived pair confidence."""

    source_id: str
    target_probability: npt.NDArray[np.float32]
    target_confidence: npt.NDArray[np.float32]
    reference_count: int

    def __post_init__(self) -> None:
        probability = self.target_probability
        confidence = self.target_confidence
        if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
            raise ValueError("target_probability must be square")
        if confidence.shape != probability.shape:
            raise ValueError("target_confidence shape must match target_probability")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("target_probability must contain finite values in [0, 1]")
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("target_confidence must contain finite values in [0, 1]")
        if not np.allclose(probability, probability.T) or not np.allclose(confidence, confidence.T):
            raise ValueError("consensus matrices must be symmetric")
        if not np.allclose(np.diag(probability), 0) or not np.allclose(np.diag(confidence), 0):
            raise ValueError("consensus matrices must have zero diagonals")
        if self.reference_count < 1:
            raise ValueError("reference_count must be positive")


def build_consensus_targets(examples: list[CVRPExample]) -> dict[str, ConsensusTarget]:
    """Group preloaded candidate examples and average their binary matrices."""
    if not examples:
        raise ValueError("cannot build consensus targets from an empty list")
    grouped: dict[str, list[CVRPExample]] = {}
    for example in examples:
        grouped.setdefault(training_source_id(example), []).append(example)
    targets: dict[str, ConsensusTarget] = {}
    for source_id, references in sorted(grouped.items()):
        shapes = {reference.constraint_matrix.shape for reference in references}
        if len(shapes) != 1:
            raise ValueError(f"reference shapes disagree for source {source_id}")
        matrices = np.stack([reference.constraint_matrix for reference in references]).astype(
            np.float32
        )
        probability = matrices.mean(axis=0, dtype=np.float32)
        confidence = np.abs(2.0 * probability - 1.0).astype(np.float32)
        np.fill_diagonal(probability, 0.0)
        np.fill_diagonal(confidence, 0.0)
        targets[source_id] = ConsensusTarget(
            source_id=source_id,
            target_probability=probability,
            target_confidence=confidence,
            reference_count=len(references),
        )
    return targets


def save_consensus_sidecar(
    targets: dict[str, ConsensusTarget], output_dir: str | Path
) -> dict[str, object]:
    """Save versioned compressed per-source arrays and a hashable manifest."""
    if not targets:
        raise ValueError("cannot save an empty consensus sidecar")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for index, source_id in enumerate(sorted(targets)):
        target = targets[source_id]
        file_name = f"target_{index:06d}.npz"
        np.savez_compressed(
            output / file_name,
            target_probability=target.target_probability,
            target_confidence=target.target_confidence,
        )
        entries.append(
            {
                "source_id": source_id,
                "file": file_name,
                "n_customers": target.target_probability.shape[0],
                "reference_count": target.reference_count,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "confidence_definition": "abs(2 * mean_binary_membership - 1)",
        "target_count": len(entries),
        "entries": entries,
    }
    (output / "consensus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["sidecar_sha256"] = hash_dataset(output)
    return manifest


def load_consensus_sidecar(input_dir: str | Path) -> dict[str, ConsensusTarget]:
    """Load and validate a sidecar without enabling pickle-backed arrays."""
    root = Path(input_dir)
    manifest = json.loads((root / "consensus_manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported consensus schema: {manifest.get('schema_version')}")
    targets: dict[str, ConsensusTarget] = {}
    for entry in manifest["entries"]:
        source_id = str(entry["source_id"])
        with np.load(root / entry["file"], allow_pickle=False) as arrays:
            target = ConsensusTarget(
                source_id=source_id,
                target_probability=np.asarray(arrays["target_probability"], dtype=np.float32),
                target_confidence=np.asarray(arrays["target_confidence"], dtype=np.float32),
                reference_count=int(entry["reference_count"]),
            )
        targets[source_id] = target
    if len(targets) != int(manifest["target_count"]):
        raise ValueError("consensus manifest contains duplicate source ids")
    return targets
