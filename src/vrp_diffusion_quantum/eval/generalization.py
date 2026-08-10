"""P3.6 size- and demand-shift evaluation for predicted constraint matrices."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.eval.matrix_ablation import (
    score_matrix_probabilities,
    validate_disjoint_examples,
)
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    predict_matrix_one_shot,
    sample_constraint_matrix,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule

PredictionMode = Literal["one_shot", "full_chain"]


@dataclass(frozen=True)
class GeneralizationCase:
    """One named test distribution evaluated without threshold or checkpoint tuning."""

    name: str
    distribution: str
    examples: Sequence[CVRPExample]


def _predict_examples(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    examples: Sequence[CVRPExample],
    *,
    mode: PredictionMode,
    device: torch.device | str,
    seed: int,
    step_stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    probabilities: list[np.ndarray] = []
    hard_matrices: list[np.ndarray] = []
    for index, example in enumerate(examples):
        coords, demands, capacity, _, mask = example_to_model_inputs(example, device=device)
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        if mode == "one_shot":
            prediction = predict_matrix_one_shot(
                model,
                schedule,
                coords=coords,
                demands=demands,
                capacity=capacity,
                customer_mask=mask,
                generator=generator,
            )
        else:
            prediction = sample_constraint_matrix(
                model,
                schedule,
                coords=coords,
                demands=demands,
                capacity=capacity,
                customer_mask=mask,
                generator=generator,
                step_stride=step_stride,
            )
        probabilities.append(prediction.m_prob)
        hard_matrices.append(prediction.m_hat)
    return probabilities, hard_matrices


def select_validation_threshold(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    selection_examples: Sequence[CVRPExample],
    *,
    mode: PredictionMode,
    device: torch.device | str,
    seed: int,
    step_stride: int = 1,
) -> float:
    """Select a validation threshold; full-chain sampling keeps its fixed hard output."""
    if not selection_examples:
        raise ValueError("selection_examples must not be empty")
    if mode == "full_chain":
        return 0.5
    probabilities, _ = _predict_examples(
        model,
        schedule,
        selection_examples,
        mode=mode,
        device=device,
        seed=seed,
        step_stride=step_stride,
    )
    metrics = score_matrix_probabilities(selection_examples, probabilities)
    return float(metrics["threshold"])


def evaluate_generalization_case(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    case: GeneralizationCase,
    *,
    mode: PredictionMode,
    threshold: float,
    device: torch.device | str,
    seed: int,
    step_stride: int = 1,
    failure_limit: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one untouched test case and return aggregate metrics plus worst examples."""
    if not case.examples:
        raise ValueError(f"generalization case {case.name!r} has no examples")
    if failure_limit < 0:
        raise ValueError(f"failure_limit must be non-negative, got {failure_limit}")

    started = time.perf_counter()
    probabilities, hard_matrices = _predict_examples(
        model,
        schedule,
        case.examples,
        mode=mode,
        device=device,
        seed=seed,
        step_stride=step_stride,
    )
    metrics = score_matrix_probabilities(
        case.examples,
        probabilities,
        m_hats=hard_matrices,
        hard_from_hats=mode == "full_chain",
        threshold=threshold,
        adaptive_threshold=False,
    )
    runtime_seconds = time.perf_counter() - started
    row: dict[str, Any] = {
        "case": case.name,
        "distribution": case.distribution,
        "mode": mode,
        **metrics,
        "runtime_seconds": runtime_seconds,
        "seed": seed,
    }

    failures: list[dict[str, Any]] = []
    for example, probability, hard_matrix in zip(
        case.examples, probabilities, hard_matrices, strict=True
    ):
        single = score_matrix_probabilities(
            [example],
            [probability],
            m_hats=[hard_matrix],
            hard_from_hats=mode == "full_chain",
            threshold=threshold,
            adaptive_threshold=False,
        )
        failures.append(
            {
                "instance_id": example.instance.instance_id,
                "n_customers": example.instance.n_customers,
                "f1": single["f1"],
                "precision": single["precision"],
                "recall": single["recall"],
                "capacity_consistency": single["capacity_consistency"],
            }
        )
    failures.sort(key=lambda item: (float(item["f1"]), str(item["instance_id"])))
    return row, failures[:failure_limit]


def evaluate_generalization_cases(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    selection_examples: Sequence[CVRPExample],
    cases: Sequence[GeneralizationCase],
    *,
    reference_case: str,
    mode: PredictionMode,
    device: torch.device | str,
    seed: int,
    step_stride: int = 1,
    failure_limit: int = 10,
) -> dict[str, Any]:
    """Evaluate all shifts and report F1 drop relative to a named in-distribution case."""
    if not cases:
        raise ValueError("cases must not be empty")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("generalization case names must be unique")
    if reference_case not in names:
        raise ValueError(f"reference_case {reference_case!r} is not in {names}")
    for case in cases:
        validate_disjoint_examples(selection_examples, case.examples)

    threshold = select_validation_threshold(
        model,
        schedule,
        selection_examples,
        mode=mode,
        device=device,
        seed=seed,
        step_stride=step_stride,
    )
    rows: list[dict[str, Any]] = []
    failure_cases: dict[str, list[dict[str, Any]]] = {}
    for case_index, case in enumerate(cases):
        row, failures = evaluate_generalization_case(
            model,
            schedule,
            case,
            mode=mode,
            threshold=threshold,
            device=device,
            seed=seed + 100_000 * case_index,
            step_stride=step_stride,
            failure_limit=failure_limit,
        )
        rows.append(row)
        failure_cases[case.name] = failures

    reference_f1 = float(next(row for row in rows if row["case"] == reference_case)["f1"])
    for row in rows:
        row["f1_drop_from_reference"] = reference_f1 - float(row["f1"])
    return {
        "reference_case": reference_case,
        "selection_threshold": threshold,
        "rows": rows,
        "failure_cases": failure_cases,
    }


__all__ = [
    "GeneralizationCase",
    "PredictionMode",
    "evaluate_generalization_case",
    "evaluate_generalization_cases",
    "select_validation_threshold",
]
