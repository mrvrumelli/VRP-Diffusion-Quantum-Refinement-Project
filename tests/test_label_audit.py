"""Tests for the resumable strong-reference labeling audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vrp_diffusion_quantum.data.dataset import make_example, save_example
from vrp_diffusion_quantum.data.label_audit import (
    AcceptancePolicy,
    AuditPolicy,
    SolveCandidate,
    analyse_instance_candidates,
    run_reference_label_audit,
)
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.utils.feasibility import route_cost


def _example(seed: int = 0) -> CVRPExample:
    rng = np.random.default_rng(seed)
    instance = CVRPInstance(
        coords=np.vstack(([0.5, 0.5], rng.random((6, 2)))),
        demands=np.array([0.0, 1, 1, 1, 1, 1, 1]),
        capacity=3.0,
        depot_index=0,
        instance_id=f"audit_{seed}",
        n_customers=6,
        seed=seed,
        generator_settings={"kind": "unit"},
    )
    routes = [[0, 1, 2], [3, 4, 5]]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=2,
        feasible=True,
        solver_name="original",
        time_budget=1.0,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def _candidate(
    routes: list[list[int]],
    cost: float,
    *,
    seed: int,
    solver: str = "pyvrp",
) -> SolveCandidate:
    return SolveCandidate(
        schema_version=1,
        source_file="example.json",
        instance_id="audit_0",
        n_customers=6,
        solver_name=solver,
        base_seed=seed,
        derived_seed=seed + 100,
        time_budget=1.0,
        routes=routes,
        cost=cost,
        num_vehicles=len(routes),
        feasible=True,
        violations=[],
        runtime_seconds=0.1,
        route_hash=f"hash-{solver}-{seed}",
    )


def test_acceptance_separates_cost_convergence_from_matrix_stability() -> None:
    example = _example()
    first_routes = [[0, 1, 2], [3, 4, 5]]
    other_routes = [[0, 1, 3], [2, 4, 5]]
    candidates = [
        _candidate(first_routes, 10.0, seed=1),
        _candidate(first_routes, 10.001, seed=2),
        _candidate(other_routes, 10.002, seed=3),
        _candidate(other_routes, 10.003, seed=4),
    ]

    row = analyse_instance_candidates(
        example,
        candidates,
        acceptance=AcceptancePolicy(matrix_disagreement_tolerance=0.01),
        expected_pyvrp_runs=4,
        source_file="example.json",
    )

    assert row["cost_converged"] is True
    assert row["reference_accepted"] is True
    assert row["matrix_stable"] is False
    assert row["matrix_target_accepted"] is False
    assert "route_membership_ambiguous" in row["acceptance_reasons"]


def test_ortools_challenger_can_reject_pyvrp_reference() -> None:
    example = _example()
    routes = [[0, 1, 2], [3, 4, 5]]
    pyvrp = [_candidate(routes, 10.0 + index * 0.001, seed=index) for index in range(4)]
    ortools = _candidate(routes, 9.5, seed=99, solver="ortools")

    row = analyse_instance_candidates(
        example,
        pyvrp,
        acceptance=AcceptancePolicy(),
        ortools_candidate=ortools,
        challenger_selection_reason="stable_control",
        expected_pyvrp_runs=4,
        source_file="example.json",
    )

    assert row["ortools_beats_pyvrp"] is True
    assert row["reference_solver"] == "ortools"
    assert row["reference_accepted"] is False
    assert row["needs_review"] is True


def test_ortools_fallback_is_preserved_for_review_when_pyvrp_fails() -> None:
    example = _example()
    routes = [[0, 1, 2], [3, 4, 5]]
    ortools = _candidate(routes, 9.5, seed=99, solver="ortools")

    row = analyse_instance_candidates(
        example,
        [],
        acceptance=AcceptancePolicy(),
        ortools_candidate=ortools,
        challenger_selection_reason="unstable_cost",
        expected_pyvrp_runs=4,
        source_file="example.json",
    )

    assert row["reference_solver"] == "ortools"
    assert row["reference_accepted"] is False
    assert row["needs_review"] is True
    assert row["reference_routes"] == routes


def test_tiny_audit_runs_and_resumes_from_candidate_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    save_example(_example(), source / "cvrp6_0.json")
    output = tmp_path / "audit"
    policy = AuditPolicy(
        pyvrp_base_seeds=(11, 12),
        time_budgets_by_size={6: 0.05},
        ortools_stable_sample_per_size=1,
        workers=1,
        acceptance=AcceptancePolicy(
            near_best_relative_tolerance=0.05,
            seed_spread_relative_tolerance=0.10,
            matrix_disagreement_tolerance=1.0,
            minimum_near_best_seeds=2,
        ),
    )

    first = run_reference_label_audit(
        source,
        output,
        policy=policy,
        expected_counts_by_size={6: 1},
    )
    candidate_paths = sorted((output / "candidates").rglob("*.json"))
    mtimes = {path: path.stat().st_mtime_ns for path in candidate_paths}
    second = run_reference_label_audit(
        source,
        output,
        policy=policy,
        expected_counts_by_size={6: 1},
    )

    assert first == second
    assert first["pyvrp_runs_completed"] == 2
    assert first["ortools_runs_selected"] == 1
    assert len(candidate_paths) == 3
    assert {path: path.stat().st_mtime_ns for path in candidate_paths} == mtimes
    assert (output / "summary.csv").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "config.yaml").is_file()
    assert (output / "dataset_hash.txt").is_file()
    assert (output / "run.log").is_file()
    assert (output / "reference_examples" / "cvrp6_0.json").is_file()
