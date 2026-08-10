"""Leakage-safe helpers shared by matrix-predictor evaluation scripts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.metrics.matrix_metrics import MatrixPrediction, compute_matrix_metrics


def validate_disjoint_examples(
    selection_examples: Sequence[CVRPExample], test_examples: Sequence[CVRPExample]
) -> None:
    """Reject validation/test leakage by stable instance id."""
    selection_ids = {example.instance.instance_id for example in selection_examples}
    test_ids = {example.instance.instance_id for example in test_examples}
    overlap = selection_ids & test_ids
    if overlap:
        raise ValueError(f"selection/test instance overlap detected: {sorted(overlap)[:5]}")


def score_matrix_probabilities(
    examples: Sequence[CVRPExample],
    m_probs: Sequence[npt.NDArray[np.float64]],
    *,
    m_hats: Sequence[npt.NDArray[np.float64]] | None = None,
    hard_from_hats: bool = False,
    threshold: float | None = None,
    adaptive_threshold: bool = True,
) -> dict[str, float]:
    """Score soft probabilities with an optional fixed hard matrix readout."""
    soft_predictions = [
        MatrixPrediction.from_example(example, probability)
        for example, probability in zip(examples, m_probs, strict=True)
    ]
    soft_metrics = compute_matrix_metrics(
        soft_predictions,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
    )

    if hard_from_hats:
        if m_hats is None:
            raise ValueError("hard_from_hats requires m_hats")
        hard_predictions = [
            MatrixPrediction.from_example(example, matrix)
            for example, matrix in zip(examples, m_hats, strict=True)
        ]
        hard_metrics = compute_matrix_metrics(
            hard_predictions,
            threshold=0.5,
            adaptive_threshold=False,
        )
        output: dict[str, float] = {
            "bce": float(soft_metrics.bce),
            "auc": (float(soft_metrics.auc) if soft_metrics.auc is not None else float("nan")),
            "precision": float(hard_metrics.precision),
            "recall": float(hard_metrics.recall),
            "f1": float(hard_metrics.f1),
            "threshold": 0.5,
            "calibration_error": float(soft_metrics.calibration_error),
            "capacity_consistency": float(soft_metrics.capacity_consistency),
            "num_examples": float(len(examples)),
        }
        predictions_for_size = hard_predictions
        threshold_for_size = 0.5
    else:
        output = {
            "bce": float(soft_metrics.bce),
            "auc": (float(soft_metrics.auc) if soft_metrics.auc is not None else float("nan")),
            "precision": float(soft_metrics.precision),
            "recall": float(soft_metrics.recall),
            "f1": float(soft_metrics.f1),
            "threshold": float(soft_metrics.threshold),
            "calibration_error": float(soft_metrics.calibration_error),
            "capacity_consistency": float(soft_metrics.capacity_consistency),
            "num_examples": float(len(examples)),
        }
        predictions_for_size = soft_predictions
        threshold_for_size = float(soft_metrics.threshold)

    by_size: dict[int, list[MatrixPrediction]] = {}
    for example, prediction in zip(examples, predictions_for_size, strict=True):
        by_size.setdefault(example.instance.n_customers, []).append(prediction)
    for size, size_predictions in sorted(by_size.items()):
        output[f"f1_n{size}"] = float(
            compute_matrix_metrics(
                size_predictions,
                threshold=threshold_for_size,
                adaptive_threshold=False,
            ).f1
        )
    return output


__all__ = ["score_matrix_probabilities", "validate_disjoint_examples"]
