from pathlib import Path

import numpy as np

from vrp_diffusion_quantum.data.consensus_targets import (
    build_consensus_targets,
    load_consensus_sidecar,
    save_consensus_sidecar,
)
from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution


def _example(routes: list[list[int]], route_hash: str) -> CVRPExample:
    instance = CVRPInstance(
        coords=np.zeros((4, 2)),
        demands=np.array([0.0, 1.0, 1.0, 1.0]),
        capacity=3.0,
        depot_index=0,
        instance_id="shared",
        n_customers=3,
        seed=0,
        generator_settings={
            "training_label_policy": {
                "source_file": "shared.json",
                "candidate_route_hash": route_hash,
            }
        },
    )
    solution = LabeledSolution(
        routes=routes,
        cost=1.0,
        num_vehicles=len(routes),
        feasible=True,
        solver_name="test",
        time_budget=1.0,
        seed=0,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_consensus_probability_and_confidence() -> None:
    targets = build_consensus_targets([_example([[0, 1], [2]], "a"), _example([[0], [1, 2]], "b")])
    target = targets["shared.json"]
    assert target.reference_count == 2
    assert target.target_probability[0, 1] == 0.5
    assert target.target_confidence[0, 1] == 0.0
    assert target.target_probability[0, 2] == 0.0
    assert target.target_confidence[0, 2] == 1.0
    assert np.all(np.diag(target.target_probability) == 0)


def test_consensus_sidecar_roundtrip(tmp_path: Path) -> None:
    targets = build_consensus_targets([_example([[0, 1, 2]], "a")])
    manifest = save_consensus_sidecar(targets, tmp_path / "sidecar")
    loaded = load_consensus_sidecar(tmp_path / "sidecar")
    assert manifest["target_count"] == 1
    assert len(str(manifest["sidecar_sha256"])) == 64
    np.testing.assert_array_equal(
        loaded["shared.json"].target_probability,
        targets["shared.json"].target_probability,
    )
