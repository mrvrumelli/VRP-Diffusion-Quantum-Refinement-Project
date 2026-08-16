from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vrp_diffusion_quantum.data.dataset import load_dataset, make_example, save_example
from vrp_diffusion_quantum.data.training_labels import (
    TrainingLabelPolicy,
    materialize_training_labels,
    select_stochastic_references,
    training_source_id,
)
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution


def _source_example() -> CVRPExample:
    instance = CVRPInstance(
        coords=np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]),
        demands=np.array([0.0, 1.0, 1.0, 1.0]),
        capacity=2.0,
        depot_index=0,
        instance_id="tiny_3",
        n_customers=3,
        seed=1,
        generator_settings={"kind": "test"},
    )
    solution = LabeledSolution(
        routes=[[0, 1], [2]],
        cost=10.0,
        num_vehicles=2,
        feasible=True,
        solver_name="original",
        time_budget=1.0,
        seed=1,
        runtime_seconds=1.0,
    )
    return make_example(instance, solution)


def _candidate(routes: list[list[int]], cost: float, route_hash: str, seed: int) -> dict:
    return {
        "schema_version": 1,
        "source_file": "cvrp3_0.json",
        "instance_id": "tiny_3",
        "n_customers": 3,
        "solver_name": "pyvrp",
        "base_seed": seed,
        "derived_seed": seed,
        "time_budget": 2.0,
        "routes": routes,
        "cost": cost,
        "num_vehicles": len(routes),
        "feasible": True,
        "violations": [],
        "runtime_seconds": 2.0,
        "route_hash": route_hash,
    }


def _write_audit(tmp_path: Path, *, accepted: bool) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    save_example(_source_example(), source / "cvrp3_0.json")
    audit = tmp_path / "audit"
    candidates = audit / "candidates" / "cvrp3_0"
    candidates.mkdir(parents=True)
    for index, candidate in enumerate(
        [
            _candidate([[0, 1], [2]], 10.0, "a", 11),
            _candidate([[0], [1, 2]], 10.04, "b", 12),
            _candidate([[0], [1, 2]], 10.04, "b", 13),
            _candidate([[0, 2], [1]], 10.2, "c", 14),
        ]
    ):
        (candidates / f"pyvrp_seed_{index}.json").write_text(json.dumps(candidate))
    results = audit / "instance_results"
    results.mkdir()
    (results / "cvrp3_0.json").write_text(json.dumps({"matrix_target_accepted": accepted}))
    (audit / "metrics.json").write_text(json.dumps({"input_sha256": "test"}))
    return source, audit


def test_multi_reference_materializes_unique_competitive_candidates(tmp_path: Path) -> None:
    source, audit = _write_audit(tmp_path, accepted=False)
    output = tmp_path / "multi"

    manifest = materialize_training_labels(
        source,
        audit,
        output,
        policy=TrainingLabelPolicy(
            modes_by_size={3: "multi_reference"}, competitive_relative_tolerance=0.005
        ),
    )

    assert manifest["source_count"] == 1
    assert manifest["label_count"] == 2
    examples = load_dataset(output)
    assert len(examples) == 2
    assert {example.solution.seed for example in examples} == {11, 12}
    assert all(
        example.instance.generator_settings["training_label_policy"]["reference_count"] == 2
        for example in examples
    )


def test_canonical_else_multi_uses_one_label_for_stable_target(tmp_path: Path) -> None:
    source, audit = _write_audit(tmp_path, accepted=True)
    output = tmp_path / "canonical"

    manifest = materialize_training_labels(
        source,
        audit,
        output,
        policy=TrainingLabelPolicy(modes_by_size={3: "canonical_else_multi"}),
    )

    assert manifest["label_count"] == 1
    assert load_dataset(output)[0].solution.seed == 11


def test_accepted_canonical_skips_ambiguous_target(tmp_path: Path) -> None:
    source, audit = _write_audit(tmp_path, accepted=False)
    output = tmp_path / "accepted"

    manifest = materialize_training_labels(
        source,
        audit,
        output,
        policy=TrainingLabelPolicy(modes_by_size={3: "accepted_canonical"}),
    )

    assert manifest["source_count"] == 1
    assert manifest["label_count"] == 0
    assert load_dataset(output) == []


def test_stochastic_reference_selection_is_compute_matched_and_deterministic(
    tmp_path: Path,
) -> None:
    source, audit = _write_audit(tmp_path, accepted=False)
    output = tmp_path / "multi"
    materialize_training_labels(
        source,
        audit,
        output,
        policy=TrainingLabelPolicy(modes_by_size={3: "multi_reference"}),
    )
    candidates = load_dataset(output)
    first = select_stochastic_references(candidates, seed=10, epoch=3)
    reordered = select_stochastic_references(list(reversed(candidates)), seed=10, epoch=3)
    assert len(first) == 1
    assert training_source_id(first[0]) == "cvrp3_0.json"
    assert first[0].solution.seed == reordered[0].solution.seed


def test_stochastic_reference_selection_keeps_one_per_distinct_source() -> None:
    first = _source_example()
    second = _source_example()
    second.instance.instance_id = "tiny_other"
    selected = select_stochastic_references([first, second], seed=0, epoch=0)
    assert len(selected) == 2
