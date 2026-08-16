"""Materialize versioned training labels from strong-audit solver candidates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from vrp_diffusion_quantum.data.dataset import load_example, make_example, save_example
from vrp_diffusion_quantum.data.label_audit import SolveCandidate
from vrp_diffusion_quantum.data.types import CVRPExample, LabeledSolution
from vrp_diffusion_quantum.utils.experiment import git_commit_hash, hash_dataset

LabelMode = Literal["original", "canonical_else_multi", "multi_reference", "accepted_canonical"]


def training_source_id(example: CVRPExample) -> str:
    """Return the stable source identity shared by materialized candidate references."""
    policy = example.instance.generator_settings.get("training_label_policy")
    if isinstance(policy, dict) and policy.get("source_file"):
        return str(policy["source_file"])
    return example.instance.instance_id


def select_stochastic_references(
    examples: list[CVRPExample], *, seed: int, epoch: int
) -> list[CVRPExample]:
    """Select exactly one preloaded hard reference per source for this epoch.

    Selection is invariant to input ordering and derives solely from ``(seed, epoch, source_id)``.
    The returned examples remain ordinary binary ``CVRPExample`` objects, so both forward noising
    and loss use the same selected hard target.
    """
    grouped: dict[str, list[CVRPExample]] = {}
    for example in examples:
        grouped.setdefault(training_source_id(example), []).append(example)
    selected: list[CVRPExample] = []
    for source_id in sorted(grouped):
        candidates = sorted(
            grouped[source_id],
            key=lambda item: (
                str(
                    item.instance.generator_settings.get("training_label_policy", {}).get(
                        "candidate_route_hash", ""
                    )
                ),
                item.solution.cost,
                item.solution.seed if item.solution.seed is not None else -1,
            ),
        )
        digest = hashlib.sha256(f"{seed}:{epoch}:{source_id}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(candidates)
        selected.append(candidates[index])
    return selected


@dataclass(frozen=True)
class TrainingLabelPolicy:
    """Per-size materialization policy for audited route labels."""

    modes_by_size: dict[int, LabelMode]
    competitive_relative_tolerance: float = 0.005

    def __post_init__(self) -> None:
        if not self.modes_by_size:
            raise ValueError("modes_by_size must not be empty")
        if any(size <= 0 for size in self.modes_by_size):
            raise ValueError("customer sizes must be positive")
        if self.competitive_relative_tolerance < 0:
            raise ValueError("competitive_relative_tolerance must be non-negative")


def _load_candidates(audit_dir: Path, source_file: str) -> list[SolveCandidate]:
    candidate_dir = audit_dir / "candidates" / Path(source_file).stem
    candidates: list[SolveCandidate] = []
    for path in sorted(candidate_dir.glob("pyvrp_seed_*.json")):
        candidate: SolveCandidate = json.loads(path.read_text())
        if candidate.get("feasible") and not candidate.get("violations"):
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: (float(item["cost"]), item["route_hash"]))


def _unique_competitive(
    candidates: list[SolveCandidate], relative_tolerance: float
) -> list[SolveCandidate]:
    if not candidates:
        return []
    best_cost = float(candidates[0]["cost"])
    limit = best_cost * (1.0 + relative_tolerance)
    unique: dict[str, SolveCandidate] = {}
    for candidate in candidates:
        if float(candidate["cost"]) <= limit:
            unique.setdefault(candidate["route_hash"], candidate)
    return sorted(unique.values(), key=lambda item: (float(item["cost"]), item["route_hash"]))


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_candidate_example(
    source_path: Path,
    output_path: Path,
    candidate: SolveCandidate,
    *,
    policy_mode: LabelMode,
    reference_index: int,
    reference_count: int,
) -> None:
    example = load_example(source_path)
    generator_settings = dict(example.instance.generator_settings)
    generator_settings["training_label_policy"] = {
        "mode": policy_mode,
        "source_file": source_path.name,
        "candidate_route_hash": candidate["route_hash"],
        "reference_index": reference_index,
        "reference_count": reference_count,
    }
    instance = replace(example.instance, generator_settings=generator_settings)
    solution = LabeledSolution(
        routes=[list(route) for route in candidate["routes"]],
        cost=float(candidate["cost"]),
        num_vehicles=int(candidate["num_vehicles"]),
        feasible=True,
        solver_name="training_reference_pyvrp",
        time_budget=float(candidate["time_budget"]),
        seed=int(candidate["derived_seed"]),
        runtime_seconds=float(candidate["runtime_seconds"]),
    )
    save_example(make_example(instance, solution), output_path)


def materialize_training_labels(
    source_dir: str | Path,
    audit_dir: str | Path,
    output_dir: str | Path,
    *,
    policy: TrainingLabelPolicy,
) -> dict[str, Any]:
    """Create a new dataset while preserving every source example unchanged."""
    source = Path(source_dir)
    audit = Path(audit_dir)
    output = Path(output_dir)
    source_paths = sorted(
        path
        for path in source.glob("*.json")
        if path.name not in {"subset_manifest.json", "training_label_manifest.json"}
    )
    if not source_paths:
        raise ValueError(f"no source examples found under {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    audit_results: dict[str, dict[str, Any]] = {}
    result_dir = audit / "instance_results"
    for path in result_dir.glob("*.json"):
        audit_results[path.name] = json.loads(path.read_text())

    entries: list[dict[str, Any]] = []
    materializations: set[str] = set()
    source_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for source_path in source_paths:
        example = load_example(source_path)
        size = example.instance.n_customers
        if size not in policy.modes_by_size:
            raise ValueError(f"no training-label mode configured for CVRP{size}")
        mode = policy.modes_by_size[size]
        source_counts[str(size)] = source_counts.get(str(size), 0) + 1

        if mode == "original":
            output_path = output / source_path.name
            materializations.add(_link_or_copy(source_path, output_path))
            entries.append(
                {
                    "file": output_path.name,
                    "source_file": source_path.name,
                    "n_customers": size,
                    "mode": mode,
                    "route_hash": None,
                }
            )
            label_counts[str(size)] = label_counts.get(str(size), 0) + 1
            continue

        candidates = _load_candidates(audit, source_path.name)
        if not candidates:
            raise ValueError(f"no feasible PyVRP candidates for {source_path.name}")
        selected = _unique_competitive(candidates, policy.competitive_relative_tolerance)
        if mode in {"canonical_else_multi", "accepted_canonical"}:
            result = audit_results.get(source_path.name)
            if result is None:
                raise ValueError(f"missing audit result for {source_path.name}")
            if mode == "accepted_canonical" and not bool(result.get("matrix_target_accepted")):
                continue
            if bool(result.get("matrix_target_accepted")):
                selected = selected[:1]
        if not selected:
            raise ValueError(f"no competitive candidates for {source_path.name}")

        for index, candidate in enumerate(selected):
            output_name = f"{source_path.stem}__ref{index:02d}.json"
            _write_candidate_example(
                source_path,
                output / output_name,
                candidate,
                policy_mode=mode,
                reference_index=index,
                reference_count=len(selected),
            )
            entries.append(
                {
                    "file": output_name,
                    "source_file": source_path.name,
                    "n_customers": size,
                    "mode": mode,
                    "route_hash": candidate["route_hash"],
                    "cost": float(candidate["cost"]),
                }
            )
            label_counts[str(size)] = label_counts.get(str(size), 0) + 1

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_dir": str(source.resolve()),
        "source_sha256": hash_dataset(source),
        "audit_dir": str(audit.resolve()),
        "audit_input_sha256": json.loads((audit / "metrics.json").read_text()).get("input_sha256"),
        "policy": asdict(policy),
        "source_counts_by_size": source_counts,
        "label_counts_by_size": label_counts,
        "source_count": len(source_paths),
        "label_count": len(entries),
        "materialization": sorted(materializations) or ["generated"],
        "git_commit": git_commit_hash(),
        "examples": entries,
    }
    (output / "training_label_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
