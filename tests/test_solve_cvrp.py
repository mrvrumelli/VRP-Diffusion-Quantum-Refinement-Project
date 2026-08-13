"""P1.1: a batch of instances receives feasible routes and costs from an OR solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vrp_diffusion_quantum.data.generate_cvrp import (
    CVRPDataset,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from vrp_diffusion_quantum.data.solve_cvrp import (
    is_feasible_solution,
    route_cost,
    save_labels,
    solve_dataset,
    solve_instance,
)


@pytest.fixture
def small_dataset() -> CVRPDataset:
    return generate_dataset(20, num_instances=3, seed=7, demand_mode="uniform")


def test_route_cost_empty_is_zero() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert route_cost([], coords) == 0.0


def test_route_cost_single_customer() -> None:
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    # depot -> customer (5) -> depot (5)
    assert route_cost([[1]], coords) == pytest.approx(10.0)


def test_pyvrp_solves_instance_feasibly(small_dataset: CVRPDataset) -> None:
    instance = small_dataset[0]
    solution = solve_instance(instance, solver="pyvrp", time_limit=0.5, seed=0)
    assert solution.feasible
    assert solution.solver_name == "pyvrp"
    assert solution.num_vehicles >= 1
    assert solution.cost > 0.0
    assert solution.time_budget == pytest.approx(0.5)
    assert is_feasible_solution(instance, solution.routes)
    assert solution.cost == pytest.approx(route_cost(solution.routes, instance.coords), rel=1e-9)


def test_ortools_solves_instance_feasibly(small_dataset: CVRPDataset) -> None:
    instance = small_dataset[1]
    solution = solve_instance(instance, solver="ortools", time_limit=0.5, seed=1)
    assert solution.feasible
    assert solution.solver_name == "ortools"
    assert is_feasible_solution(instance, solution.routes)


def test_solve_dataset_labels_batch(small_dataset: CVRPDataset) -> None:
    solutions = solve_dataset(small_dataset, solver="pyvrp", time_limit=0.5, seed=11)
    assert len(solutions) == len(small_dataset)
    assert all(solution.feasible for solution in solutions)
    assert all(solution.cost > 0.0 for solution in solutions)
    # Per-instance seeds must differ so the batch is not a single shared stream.
    assert len({solution.seed for solution in solutions}) == len(solutions)


def test_save_labels_round_trip(tmp_path: Path, small_dataset: CVRPDataset) -> None:
    solutions = solve_dataset(
        small_dataset,
        solver="pyvrp",
        time_limit=0.5,
        seed=3,
        fleet_mode="up_to",
        fleet_size=8,
    )
    path = save_labels(solutions, tmp_path / "labels.json")
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert len(payload) == len(solutions)
    assert payload[0]["routes"] == solutions[0].routes
    assert payload[0]["feasible"] is True
    assert payload[0]["fleet_mode"] == "up_to"
    assert payload[0]["fleet_size"] == 8
    assert payload[0]["time_budget"] == pytest.approx(0.5)


def test_pyvrp_no_improvement_stop(small_dataset: CVRPDataset) -> None:
    solution = solve_instance(
        small_dataset[0],
        solver="pyvrp",
        time_limit=None,
        no_improvement_seconds=0.3,
        seed=0,
    )
    assert solution.feasible
    assert solution.time_budget is None
    assert solution.runtime_seconds < 5.0


def test_fleet_up_to_respects_cap(small_dataset: CVRPDataset) -> None:
    solution = solve_instance(
        small_dataset[0],
        solver="pyvrp",
        time_limit=0.5,
        fleet_mode="up_to",
        fleet_size=8,
        seed=0,
    )
    assert solution.feasible
    assert solution.num_vehicles <= 8


def test_pyvrp_exact_fleet_does_not_reject_fixed_cost(small_dataset: CVRPDataset) -> None:
    # PyVRP requires fixed_cost >= 0; exact mode must not pass a negative cost.
    solution = solve_instance(
        small_dataset[0],
        solver="pyvrp",
        time_limit=0.5,
        fleet_mode="exact",
        fleet_size=8,
        seed=0,
    )
    assert solution.feasible
    assert solution.fleet_mode == "exact"
    assert solution.fleet_size == 8
    assert solution.num_vehicles <= 8


def test_solve_normalized_csv_round_trip(tmp_path: Path) -> None:
    # Samples under data/samples/ are not a required input — generate into tmp and reload.
    generated = generate_dataset(20, num_instances=2, seed=13)
    stem = tmp_path / "cvrp20"
    save_dataset(generated, stem)
    dataset = load_dataset(stem)
    solution = solve_instance(dataset[0], solver="pyvrp", time_limit=0.5, seed=0)
    assert solution.feasible
    assert is_feasible_solution(dataset[0], solution.routes)
