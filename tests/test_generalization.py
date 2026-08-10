"""Tests for P3.6 size/distribution stress evaluation."""

from __future__ import annotations

import numpy as np
import torch

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.eval.generalization import (
    GeneralizationCase,
    evaluate_generalization_cases,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.utils.feasibility import route_cost


def _example(n_customers: int, instance_id: str, seed: int) -> CVRPExample:
    rng = np.random.default_rng(seed)
    instance = CVRPInstance(
        coords=rng.random((n_customers + 1, 2)),
        demands=np.concatenate([[0.0], np.ones(n_customers)]),
        capacity=float(n_customers),
        depot_index=0,
        instance_id=instance_id,
        n_customers=n_customers,
        seed=seed,
        generator_settings={"demand_mode": "unit"},
    )
    split = max(n_customers // 2, 1)
    routes = [list(range(split)), list(range(split, n_customers))]
    solution = LabeledSolution(
        routes=routes,
        cost=route_cost(instance, routes),
        num_vehicles=len(routes),
        feasible=True,
        solver_name="test",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_generalization_report_has_drops_and_failure_cases() -> None:
    torch.manual_seed(0)
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    schedule = BernoulliDiffusionSchedule(num_timesteps=4)
    selection = [_example(4, "selection", 0)]
    cases = [
        GeneralizationCase("reference", "uniform", [_example(4, "reference", 1)]),
        GeneralizationCase("size_shift", "uniform", [_example(6, "size-shift", 2)]),
    ]

    report = evaluate_generalization_cases(
        model,
        schedule,
        selection,
        cases,
        reference_case="reference",
        mode="one_shot",
        device="cpu",
        seed=7,
        failure_limit=1,
    )

    assert report["reference_case"] == "reference"
    assert len(report["rows"]) == 2
    assert report["rows"][0]["f1_drop_from_reference"] == 0.0
    assert "f1_drop_from_reference" in report["rows"][1]
    assert len(report["failure_cases"]["reference"]) == 1
    assert report["failure_cases"]["size_shift"][0]["instance_id"] == "size-shift"
