"""Resumable multi-solver reference-label audit for CVRP examples."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import yaml
from tqdm import tqdm  # type: ignore[import-untyped]

from vrp_diffusion_quantum.data.dataset import load_example, make_example, save_example
from vrp_diffusion_quantum.data.generate_cvrp import CVRPInstance as SolverInstance
from vrp_diffusion_quantum.data.solve_cvrp import solve_instance
from vrp_diffusion_quantum.data.types import CVRPExample, LabeledSolution
from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix
from vrp_diffusion_quantum.utils.experiment import git_commit_hash, hash_dataset
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

__all__ = [
    "AcceptancePolicy",
    "AuditPolicy",
    "SolveCandidate",
    "analyse_instance_candidates",
    "run_reference_label_audit",
]

SolverName = Literal["pyvrp", "ortools"]
_SCHEMA_VERSION = 1


class SolveCandidate(TypedDict):
    schema_version: int
    source_file: str
    instance_id: str
    n_customers: int
    solver_name: str
    base_seed: int
    derived_seed: int
    time_budget: float
    routes: list[list[int]]
    cost: float
    num_vehicles: int
    feasible: bool
    violations: list[str]
    runtime_seconds: float
    route_hash: str


@dataclass(frozen=True)
class AcceptancePolicy:
    """Thresholds used to classify strong references and stable matrix targets."""

    near_best_relative_tolerance: float = 0.001
    seed_spread_relative_tolerance: float = 0.005
    matrix_disagreement_tolerance: float = 0.05
    challenger_improvement_relative_tolerance: float = 1e-6
    minimum_near_best_seeds: int = 2

    def __post_init__(self) -> None:
        tolerances = (
            self.near_best_relative_tolerance,
            self.seed_spread_relative_tolerance,
            self.matrix_disagreement_tolerance,
            self.challenger_improvement_relative_tolerance,
        )
        if any(value < 0 for value in tolerances):
            raise ValueError("acceptance tolerances must be non-negative")
        if self.minimum_near_best_seeds < 2:
            raise ValueError("minimum_near_best_seeds must be >= 2")


@dataclass(frozen=True)
class AuditPolicy:
    """Complete reproducible solver policy for one audit run."""

    pyvrp_base_seeds: tuple[int, ...]
    time_budgets_by_size: dict[int, float]
    ortools_stable_sample_per_size: int = 50
    workers: int | None = None
    acceptance: AcceptancePolicy = AcceptancePolicy()

    def __post_init__(self) -> None:
        if not self.pyvrp_base_seeds or len(set(self.pyvrp_base_seeds)) != len(
            self.pyvrp_base_seeds
        ):
            raise ValueError("PyVRP base seeds must be non-empty and unique")
        if not self.time_budgets_by_size or any(
            size <= 0 or budget <= 0 for size, budget in self.time_budgets_by_size.items()
        ):
            raise ValueError("time budgets require positive sizes and seconds")
        if self.ortools_stable_sample_per_size < 0:
            raise ValueError("ortools_stable_sample_per_size must be >= 0")
        if self.workers is not None and self.workers < 1:
            raise ValueError("workers must be >= 1 when set")


@dataclass(frozen=True)
class _SolveTask:
    source_path: str
    cache_path: str
    solver_name: SolverName
    base_seed: int
    time_budget: float


def _relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


def _canonical_route_hash(routes: list[list[int]]) -> str:
    canonical_routes: list[tuple[int, ...]] = []
    for route in routes:
        direct = tuple(route)
        reverse = tuple(reversed(route))
        canonical_routes.append(min(direct, reverse))
    canonical_routes.sort()
    payload = json.dumps(canonical_routes, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _derived_seed(base_seed: int, instance_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{instance_id}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _node_routes_to_customer_routes(
    example: CVRPExample, routes: list[list[int]]
) -> list[list[int]]:
    node_to_customer = {
        node_id: customer_id
        for customer_id, node_id in enumerate(example.instance.customer_node_indices())
    }
    try:
        return [[node_to_customer[node_id] for node_id in route] for route in routes]
    except KeyError as exc:
        raise ValueError(f"solver route contains depot or unknown node {exc.args[0]}") from exc


def _solve_task(task: _SolveTask) -> SolveCandidate:
    example = load_example(task.source_path)
    instance = example.instance
    solver_instance = SolverInstance(
        coords=instance.coords,
        demands=instance.demands,
        capacity=instance.capacity,
        depot_index=instance.depot_index,
    )
    derived_seed = _derived_seed(task.base_seed, instance.instance_id)
    solved = solve_instance(
        solver_instance,
        solver=task.solver_name,
        time_limit=task.time_budget,
        seed=derived_seed,
        instance_id=0,
    )
    routes = _node_routes_to_customer_routes(example, solved.routes)
    report = validate_routes(instance, routes)
    cost = route_cost(instance, routes)
    feasible = bool(solved.feasible and report.feasible and math.isfinite(cost))
    return SolveCandidate(
        schema_version=_SCHEMA_VERSION,
        source_file=Path(task.source_path).name,
        instance_id=instance.instance_id,
        n_customers=instance.n_customers,
        solver_name=task.solver_name,
        base_seed=task.base_seed,
        derived_seed=derived_seed,
        time_budget=float(task.time_budget),
        routes=routes,
        cost=float(cost),
        num_vehicles=len(routes),
        feasible=feasible,
        violations=list(report.violations),
        runtime_seconds=float(solved.runtime_seconds),
        route_hash=_canonical_route_hash(routes),
    )


def _candidate_matches_task(candidate: SolveCandidate, task: _SolveTask) -> bool:
    return bool(
        candidate.get("schema_version") == _SCHEMA_VERSION
        and candidate.get("source_file") == Path(task.source_path).name
        and candidate.get("solver_name") == task.solver_name
        and int(candidate.get("base_seed", -1)) == task.base_seed
        and math.isclose(float(candidate.get("time_budget", -1)), task.time_budget)
    )


def _read_cached_candidate(task: _SolveTask) -> SolveCandidate | None:
    path = Path(task.cache_path)
    if not path.is_file():
        return None
    try:
        candidate: SolveCandidate = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return candidate if _candidate_matches_task(candidate, task) else None


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _execute_tasks(
    tasks: list[_SolveTask], *, workers: int
) -> tuple[list[SolveCandidate], list[str]]:
    candidates: list[SolveCandidate] = []
    pending: list[_SolveTask] = []
    for task in tasks:
        cached = _read_cached_candidate(task)
        if cached is None:
            pending.append(task)
        else:
            candidates.append(cached)

    errors: list[str] = []
    if pending and workers == 1:
        for task in tqdm(pending, desc="solver candidates", unit="run"):
            try:
                candidate = _solve_task(task)
            except Exception as exc:
                errors.append(
                    f"{Path(task.source_path).name} {task.solver_name} "
                    f"seed={task.base_seed}: {type(exc).__name__}: {exc}"
                )
                continue
            _atomic_write_json(Path(task.cache_path), candidate)
            candidates.append(candidate)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_solve_task, task): task for task in pending}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="solver candidates",
                unit="run",
            ):
                task = futures[future]
                try:
                    candidate = future.result()
                except Exception as exc:
                    errors.append(
                        f"{Path(task.source_path).name} {task.solver_name} "
                        f"seed={task.base_seed}: {type(exc).__name__}: {exc}"
                    )
                    continue
                _atomic_write_json(Path(task.cache_path), candidate)
                candidates.append(candidate)
    return candidates, errors


def _matrix_disagreement(first: SolveCandidate, second: SolveCandidate) -> float:
    n_customers = first["n_customers"]
    first_m = build_constraint_matrix(first["routes"], n_customers)
    second_m = build_constraint_matrix(second["routes"], n_customers)
    if n_customers < 2:
        return 0.0
    upper = np.triu_indices(n_customers, k=1)
    return float(np.mean(first_m[upper] != second_m[upper]))


def _matrix_disagreement_from_routes(
    first_routes: list[list[int]], second_routes: list[list[int]], n_customers: int
) -> float:
    first = build_constraint_matrix(first_routes, n_customers)
    second = build_constraint_matrix(second_routes, n_customers)
    if n_customers < 2:
        return 0.0
    upper = np.triu_indices(n_customers, k=1)
    return float(np.mean(first[upper] != second[upper]))


def analyse_instance_candidates(
    example: CVRPExample,
    pyvrp_candidates: list[SolveCandidate],
    *,
    acceptance: AcceptancePolicy,
    ortools_candidate: SolveCandidate | None = None,
    challenger_selection_reason: str = "not_selected",
    expected_pyvrp_runs: int | None = None,
    source_file: str = "",
) -> dict[str, Any]:
    """Analyse convergence, matrix ambiguity, challenger outcome, and reference acceptance."""
    valid_pyvrp = [candidate for candidate in pyvrp_candidates if candidate["feasible"]]
    valid_pyvrp.sort(key=lambda item: (item["cost"], item["route_hash"]))
    expected_runs = expected_pyvrp_runs or len(pyvrp_candidates)
    all_pyvrp_feasible = expected_runs > 0 and len(valid_pyvrp) == expected_runs
    original_cost = route_cost(example.instance, example.solution.routes)

    if not valid_pyvrp:
        fallback: dict[str, Any] = {
            "instance_id": example.instance.instance_id,
            "source_file": source_file,
            "n_customers": example.instance.n_customers,
            "original_cost": original_cost,
            "pyvrp_candidate_count": expected_runs,
            "pyvrp_feasible_count": 0,
            "cost_converged": False,
            "matrix_stable": False,
            "challenger_selection_reason": challenger_selection_reason,
            "ortools_checked": ortools_candidate is not None,
            "ortools_feasible": bool(ortools_candidate and ortools_candidate["feasible"]),
            "reference_accepted": False,
            "matrix_target_accepted": False,
            "needs_review": True,
            "acceptance_reasons": ["no_feasible_pyvrp_candidate"],
        }
        if ortools_candidate is not None and ortools_candidate["feasible"]:
            fallback.update(
                {
                    "reference_cost": ortools_candidate["cost"],
                    "reference_solver": "ortools",
                    "reference_base_seed": ortools_candidate["base_seed"],
                    "reference_derived_seed": ortools_candidate["derived_seed"],
                    "reference_time_budget": ortools_candidate["time_budget"],
                    "reference_runtime_seconds": ortools_candidate["runtime_seconds"],
                    "reference_routes": ortools_candidate["routes"],
                    "reference_route_hash": ortools_candidate["route_hash"],
                    "reference_num_vehicles": ortools_candidate["num_vehicles"],
                    "vehicle_count_delta_from_original": (
                        ortools_candidate["num_vehicles"] - example.solution.num_vehicles
                    ),
                    "improvement_over_original": original_cost - ortools_candidate["cost"],
                    "relative_improvement_over_original": (
                        (original_cost - ortools_candidate["cost"]) / max(abs(original_cost), 1e-12)
                    ),
                    "matrix_disagreement_from_original": _matrix_disagreement_from_routes(
                        example.solution.routes,
                        ortools_candidate["routes"],
                        example.instance.n_customers,
                    ),
                }
            )
        return fallback

    pyvrp_best = valid_pyvrp[0]
    near_best = [
        candidate
        for candidate in valid_pyvrp
        if _relative_difference(candidate["cost"], pyvrp_best["cost"])
        <= acceptance.near_best_relative_tolerance
    ]
    seed_spread = _relative_difference(valid_pyvrp[-1]["cost"], pyvrp_best["cost"])
    cost_converged = (
        len(near_best) >= acceptance.minimum_near_best_seeds
        and seed_spread <= acceptance.seed_spread_relative_tolerance
    )
    disagreements = [
        _matrix_disagreement(first, second)
        for index, first in enumerate(near_best)
        for second in near_best[index + 1 :]
    ]
    max_matrix_disagreement = max(disagreements, default=0.0)
    mean_matrix_disagreement = float(np.mean(disagreements)) if disagreements else 0.0
    customer_instabilities: list[float] = []
    for index, first in enumerate(near_best):
        first_m = build_constraint_matrix(first["routes"], example.instance.n_customers)
        for second in near_best[index + 1 :]:
            second_m = build_constraint_matrix(second["routes"], example.instance.n_customers)
            customer_instabilities.extend(
                np.mean(first_m != second_m, axis=1).astype(float).tolist()
            )
    max_customer_instability = max(customer_instabilities, default=0.0)
    matrix_stable = (
        len(near_best) >= acceptance.minimum_near_best_seeds
        and max_matrix_disagreement <= acceptance.matrix_disagreement_tolerance
    )

    ortools_feasible = bool(ortools_candidate and ortools_candidate["feasible"])
    ortools_beats_pyvrp = bool(
        ortools_feasible
        and ortools_candidate is not None
        and ortools_candidate["cost"]
        < pyvrp_best["cost"] * (1.0 - acceptance.challenger_improvement_relative_tolerance)
    )
    valid_all = [*valid_pyvrp]
    if ortools_feasible and ortools_candidate is not None:
        valid_all.append(ortools_candidate)
    valid_all.sort(key=lambda item: (item["cost"], item["route_hash"]))
    reference = valid_all[0]

    challenger_required = challenger_selection_reason != "not_selected"
    challenger_passed = not challenger_required or (ortools_feasible and not ortools_beats_pyvrp)
    reference_accepted = all_pyvrp_feasible and cost_converged and challenger_passed
    matrix_target_accepted = reference_accepted and matrix_stable
    reasons: list[str] = []
    if not all_pyvrp_feasible:
        reasons.append("missing_or_infeasible_pyvrp_candidate")
    if not cost_converged:
        reasons.append("pyvrp_cost_not_converged")
    if challenger_required and not ortools_feasible:
        reasons.append("ortools_challenger_failed")
    if ortools_beats_pyvrp:
        reasons.append("ortools_beats_pyvrp")
    if not matrix_stable:
        reasons.append("route_membership_ambiguous")
    if not reasons:
        reasons.append("accepted")

    return {
        "instance_id": example.instance.instance_id,
        "source_file": pyvrp_best["source_file"],
        "n_customers": example.instance.n_customers,
        "original_cost": original_cost,
        "reference_cost": reference["cost"],
        "reference_solver": reference["solver_name"],
        "reference_base_seed": reference["base_seed"],
        "reference_derived_seed": reference["derived_seed"],
        "reference_time_budget": reference["time_budget"],
        "reference_runtime_seconds": reference["runtime_seconds"],
        "reference_routes": reference["routes"],
        "reference_route_hash": reference["route_hash"],
        "reference_num_vehicles": reference["num_vehicles"],
        "vehicle_count_delta_from_original": (
            reference["num_vehicles"] - example.solution.num_vehicles
        ),
        "improvement_over_original": original_cost - reference["cost"],
        "relative_improvement_over_original": (
            (original_cost - reference["cost"]) / max(abs(original_cost), 1e-12)
        ),
        "pyvrp_candidate_count": expected_runs,
        "pyvrp_feasible_count": len(valid_pyvrp),
        "pyvrp_best_cost": pyvrp_best["cost"],
        "pyvrp_worst_cost": valid_pyvrp[-1]["cost"],
        "pyvrp_seed_spread_relative": seed_spread,
        "near_best_seed_count": len(near_best),
        "cost_converged": cost_converged,
        "max_near_best_matrix_disagreement": max_matrix_disagreement,
        "mean_near_best_matrix_disagreement": mean_matrix_disagreement,
        "max_customer_membership_instability": max_customer_instability,
        "matrix_disagreement_from_original": _matrix_disagreement_from_routes(
            example.solution.routes,
            reference["routes"],
            example.instance.n_customers,
        ),
        "matrix_stable": matrix_stable,
        "challenger_selection_reason": challenger_selection_reason,
        "ortools_checked": ortools_candidate is not None,
        "ortools_feasible": ortools_feasible,
        "ortools_cost": (
            ortools_candidate["cost"]
            if ortools_candidate is not None and ortools_feasible
            else None
        ),
        "ortools_beats_pyvrp": ortools_beats_pyvrp,
        "reference_accepted": reference_accepted,
        "matrix_target_accepted": matrix_target_accepted,
        "needs_review": not matrix_target_accepted,
        "acceptance_reasons": reasons,
    }


def _candidate_cache_path(
    output_dir: Path,
    source_path: Path,
    solver_name: SolverName,
    base_seed: int,
) -> Path:
    return output_dir / "candidates" / source_path.stem / f"{solver_name}_seed_{base_seed}.json"


def _task_for(
    source_path: Path,
    output_dir: Path,
    *,
    solver_name: SolverName,
    base_seed: int,
    time_budget: float,
) -> _SolveTask:
    return _SolveTask(
        source_path=str(source_path.resolve()),
        cache_path=str(
            _candidate_cache_path(output_dir, source_path, solver_name, base_seed).resolve()
        ),
        solver_name=solver_name,
        base_seed=base_seed,
        time_budget=time_budget,
    )


def _group_candidates(candidates: list[SolveCandidate]) -> dict[str, list[SolveCandidate]]:
    grouped: dict[str, list[SolveCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["source_file"], []).append(candidate)
    return grouped


def _select_ortools_challengers(
    provisional: list[dict[str, Any]], *, stable_sample_per_size: int
) -> dict[str, str]:
    selected: dict[str, str] = {}
    stable_by_size: dict[int, list[dict[str, Any]]] = {}
    for row in provisional:
        source_file = str(row["source_file"])
        if not row.get("cost_converged", False):
            selected[source_file] = "unstable_cost"
        elif not row.get("matrix_stable", False):
            selected[source_file] = "unstable_matrix"
        else:
            stable_by_size.setdefault(int(row["n_customers"]), []).append(row)

    for size, rows in stable_by_size.items():
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"ortools-control:{size}:{row['instance_id']}".encode()
            ).hexdigest(),
        )
        for row in ranked[:stable_sample_per_size]:
            selected[str(row["source_file"])] = "stable_control"
    return selected


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row if key != "reference_routes"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value) if isinstance(value, list | dict) else value
                for key, value in row.items()
                if key != "reference_routes"
            }
            writer.writerow(serializable)


def _aggregate_metrics(rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "num_instances": len(rows),
        "reference_accepted": sum(bool(row.get("reference_accepted")) for row in rows),
        "matrix_target_accepted": sum(bool(row.get("matrix_target_accepted")) for row in rows),
        "needs_review": sum(bool(row.get("needs_review")) for row in rows),
        "ortools_checked": sum(bool(row.get("ortools_checked")) for row in rows),
        "ortools_beats_pyvrp": sum(bool(row.get("ortools_beats_pyvrp")) for row in rows),
        "solver_errors": len(errors),
        "errors": errors,
        "by_size": {},
    }
    by_size: dict[str, Any] = {}
    for size in sorted({int(row["n_customers"]) for row in rows}):
        subset = [row for row in rows if int(row["n_customers"]) == size]
        improvements = [
            float(row["relative_improvement_over_original"])
            for row in subset
            if row.get("relative_improvement_over_original") is not None
        ]
        original_matrix_disagreements = [
            float(row["matrix_disagreement_from_original"])
            for row in subset
            if row.get("matrix_disagreement_from_original") is not None
        ]
        by_size[str(size)] = {
            "count": len(subset),
            "reference_accepted": sum(bool(row.get("reference_accepted")) for row in subset),
            "matrix_target_accepted": sum(
                bool(row.get("matrix_target_accepted")) for row in subset
            ),
            "needs_review": sum(bool(row.get("needs_review")) for row in subset),
            "ortools_checked": sum(bool(row.get("ortools_checked")) for row in subset),
            "ortools_beats_pyvrp": sum(bool(row.get("ortools_beats_pyvrp")) for row in subset),
            "mean_relative_improvement_over_original": (
                float(np.mean(improvements)) if improvements else None
            ),
            "mean_matrix_disagreement_from_original": (
                float(np.mean(original_matrix_disagreements))
                if original_matrix_disagreements
                else None
            ),
        }
    metrics["by_size"] = by_size
    return metrics


def _write_reference_example(source_path: Path, output_path: Path, row: dict[str, Any]) -> None:
    example = load_example(source_path)
    generator_settings = dict(example.instance.generator_settings)
    generator_settings["strong_reference_audit"] = {
        "accepted": bool(row["reference_accepted"]),
        "matrix_target_accepted": bool(row["matrix_target_accepted"]),
        "route_hash": row["reference_route_hash"],
    }
    instance = replace(example.instance, generator_settings=generator_settings)
    solution = LabeledSolution(
        routes=row["reference_routes"],
        cost=float(row["reference_cost"]),
        num_vehicles=int(row["reference_num_vehicles"]),
        feasible=True,
        solver_name=f"strong_reference_{row['reference_solver']}",
        time_budget=float(row["reference_time_budget"]),
        seed=int(row["reference_derived_seed"]),
        runtime_seconds=float(row["reference_runtime_seconds"]),
    )
    save_example(make_example(instance, solution), output_path)


def run_reference_label_audit(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    policy: AuditPolicy,
    expected_counts_by_size: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Run or resume the complete multi-start reference-label audit."""
    source_dir = Path(input_dir)
    destination = Path(output_dir)
    source_paths = sorted(
        path for path in source_dir.glob("*.json") if path.name != "subset_manifest.json"
    )
    if not source_paths:
        raise ValueError(f"no CVRP example JSON files found under {source_dir}")

    examples = {path.name: load_example(path) for path in source_paths}
    counts_by_size: dict[int, int] = {}
    for example in examples.values():
        size = example.instance.n_customers
        counts_by_size[size] = counts_by_size.get(size, 0) + 1
        if size not in policy.time_budgets_by_size:
            raise ValueError(f"no solver time budget configured for CVRP{size}")
    if expected_counts_by_size is not None and counts_by_size != expected_counts_by_size:
        raise ValueError(
            f"audit counts {counts_by_size} do not match expected {expected_counts_by_size}"
        )
    if len(policy.pyvrp_base_seeds) < policy.acceptance.minimum_near_best_seeds:
        raise ValueError("not enough PyVRP seeds for the minimum near-best acceptance count")
    workers = policy.workers or min(max((os.cpu_count() or 2) - 1, 1), 16)
    destination.mkdir(parents=True, exist_ok=True)

    run_config: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "input_dir": str(source_dir.resolve()),
        "input_sha256": hash_dataset(source_dir),
        "counts_by_size": counts_by_size,
        "policy": asdict(policy),
        "workers_resolved": workers,
    }
    # Normalize integer-keyed maps to their on-disk JSON representation before resume comparison.
    run_config = json.loads(json.dumps(run_config))
    config_path = destination / "config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text())
        comparable_previous = dict(previous)
        comparable_previous.pop("workers_resolved", None)
        comparable_current = dict(run_config)
        comparable_current.pop("workers_resolved", None)
        if comparable_previous != comparable_current:
            raise ValueError(
                f"existing audit config differs at {config_path}; choose a new output directory"
            )
    _atomic_write_json(config_path, run_config)
    _atomic_write_json(destination / "acceptance_policy.json", asdict(policy.acceptance))
    (destination / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    (destination / "seed.txt").write_text(
        "\n".join(str(seed) for seed in policy.pyvrp_base_seeds) + "\n"
    )
    (destination / "dataset_hash.txt").write_text(f"{run_config['input_sha256']}\n")
    commit = git_commit_hash()
    if commit is not None:
        (destination / "commit_hash.txt").write_text(f"{commit}\n")

    pyvrp_tasks = [
        _task_for(
            source_path,
            destination,
            solver_name="pyvrp",
            base_seed=base_seed,
            time_budget=policy.time_budgets_by_size[
                examples[source_path.name].instance.n_customers
            ],
        )
        for source_path in source_paths
        for base_seed in policy.pyvrp_base_seeds
    ]
    pyvrp_candidates, pyvrp_errors = _execute_tasks(pyvrp_tasks, workers=workers)
    pyvrp_by_file = _group_candidates(pyvrp_candidates)
    provisional = [
        analyse_instance_candidates(
            examples[path.name],
            pyvrp_by_file.get(path.name, []),
            acceptance=policy.acceptance,
            expected_pyvrp_runs=len(policy.pyvrp_base_seeds),
            source_file=path.name,
        )
        for path in source_paths
    ]
    challenger_selection = _select_ortools_challengers(
        provisional,
        stable_sample_per_size=policy.ortools_stable_sample_per_size,
    )

    ortools_tasks = [
        _task_for(
            source_path,
            destination,
            solver_name="ortools",
            base_seed=policy.pyvrp_base_seeds[0],
            time_budget=policy.time_budgets_by_size[
                examples[source_path.name].instance.n_customers
            ],
        )
        for source_path in source_paths
        if source_path.name in challenger_selection
    ]
    ortools_candidates, ortools_errors = _execute_tasks(ortools_tasks, workers=workers)
    ortools_by_file = _group_candidates(ortools_candidates)

    final_rows: list[dict[str, Any]] = []
    results_dir = destination / "instance_results"
    references_dir = destination / "reference_examples"
    accepted_matrix_dir = destination / "accepted_matrix_examples"
    for source_path in source_paths:
        ortools_for_file = ortools_by_file.get(source_path.name, [])
        row = analyse_instance_candidates(
            examples[source_path.name],
            pyvrp_by_file.get(source_path.name, []),
            acceptance=policy.acceptance,
            ortools_candidate=ortools_for_file[0] if ortools_for_file else None,
            challenger_selection_reason=challenger_selection.get(source_path.name, "not_selected"),
            expected_pyvrp_runs=len(policy.pyvrp_base_seeds),
            source_file=source_path.name,
        )
        final_rows.append(row)
        _atomic_write_json(results_dir / source_path.name, row)
        if row.get("reference_routes") is not None:
            _write_reference_example(source_path, references_dir / source_path.name, row)
            if row.get("matrix_target_accepted"):
                _write_reference_example(source_path, accepted_matrix_dir / source_path.name, row)

    errors = [*pyvrp_errors, *ortools_errors]
    _write_summary_csv(destination / "summary.csv", final_rows)
    metrics = _aggregate_metrics(final_rows, errors)
    metrics.update(
        {
            "input_sha256": run_config["input_sha256"],
            "counts_by_size": {str(key): value for key, value in counts_by_size.items()},
            "pyvrp_runs_expected": len(pyvrp_tasks),
            "pyvrp_runs_completed": len(pyvrp_candidates),
            "ortools_runs_selected": len(ortools_tasks),
            "ortools_runs_completed": len(ortools_candidates),
            "pyvrp_solver_runtime_seconds": sum(
                candidate["runtime_seconds"] for candidate in pyvrp_candidates
            ),
            "ortools_solver_runtime_seconds": sum(
                candidate["runtime_seconds"] for candidate in ortools_candidates
            ),
        }
    )
    _atomic_write_json(destination / "metrics.json", metrics)
    log_lines = [
        "strong reference label audit complete",
        f"input={source_dir.resolve()}",
        f"input_sha256={run_config['input_sha256']}",
        f"instances={len(final_rows)}",
        f"reference_accepted={metrics['reference_accepted']}",
        f"matrix_target_accepted={metrics['matrix_target_accepted']}",
        f"needs_review={metrics['needs_review']}",
        f"ortools_beats_pyvrp={metrics['ortools_beats_pyvrp']}",
        f"solver_errors={len(errors)}",
        *errors,
    ]
    (destination / "run.log").write_text("\n".join(log_lines) + "\n")
    return metrics
