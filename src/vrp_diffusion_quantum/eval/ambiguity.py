"""Partition-invariant matrix evaluation against a frozen candidate set."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.metrics.matrix_metrics import (
    binary_cross_entropy,
    off_diagonal_pairs,
    precision_recall_f1,
    roc_auc,
)


@dataclass(frozen=True)
class ReferenceAgreement:
    """Metrics against one declared reference selected by a single criterion."""

    reference_index: int
    selection_criterion: str
    f1: float
    precision: float
    recall: float
    accuracy: float
    bce: float
    auc: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OracleAgreement:
    """Canonical and oracle best-known-reference agreement for one prediction."""

    canonical: ReferenceAgreement
    oracle: ReferenceAgreement
    num_frozen_references: int
    threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_reference(
    reference: npt.ArrayLike, expected_shape: tuple[int, int]
) -> npt.NDArray[np.float64]:
    matrix = np.asarray(reference, dtype=np.float64)
    if matrix.shape != expected_shape:
        raise ValueError(f"reference shape {matrix.shape} does not match {expected_shape}")
    if not np.all(np.isin(matrix, [0.0, 1.0])):
        raise ValueError("references must be binary")
    if not np.array_equal(matrix, matrix.T) or not np.all(np.diag(matrix) == 0):
        raise ValueError("references must be symmetric with a zero diagonal")
    return matrix


def _agreement(
    prediction: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
    *,
    reference_index: int,
    threshold: float,
    selection_criterion: str,
) -> ReferenceAgreement:
    probability = off_diagonal_pairs(prediction)
    truth = off_diagonal_pairs(reference)
    hard = precision_recall_f1(probability, truth, threshold=threshold)
    return ReferenceAgreement(
        reference_index=reference_index,
        selection_criterion=selection_criterion,
        f1=hard.f1,
        precision=hard.precision,
        recall=hard.recall,
        accuracy=hard.accuracy,
        bce=binary_cross_entropy(probability, truth),
        auc=roc_auc(probability, truth),
    )


def evaluate_oracle_reference_agreement(
    prediction: npt.ArrayLike,
    canonical_reference: npt.ArrayLike,
    frozen_references: list[npt.ArrayLike],
    *,
    threshold: float = 0.5,
) -> OracleAgreement:
    """Report canonical and oracle agreement, choosing the oracle once by hard F1.

    F1 is the frozen matching criterion. Ties resolve by lowest candidate index. Every oracle
    metric is then reported against that one reference; metrics never choose separate winners.
    This is best-known-reference agreement, not an optimality or feasibility claim.
    """
    probability = np.asarray(prediction, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
        raise ValueError("prediction must be a square matrix")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("prediction must contain finite probabilities in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    canonical = _validate_reference(canonical_reference, probability.shape)
    references = [_validate_reference(item, probability.shape) for item in frozen_references]
    if not references:
        raise ValueError("frozen_references must be non-empty")
    canonical_metrics = _agreement(
        probability,
        canonical,
        reference_index=0,
        threshold=threshold,
        selection_criterion="canonical",
    )
    candidate_metrics = [
        _agreement(
            probability,
            reference,
            reference_index=index,
            threshold=threshold,
            selection_criterion="maximum_hard_f1_then_lowest_index",
        )
        for index, reference in enumerate(references)
    ]
    oracle = max(candidate_metrics, key=lambda item: (item.f1, -item.reference_index))
    return OracleAgreement(
        canonical=canonical_metrics,
        oracle=oracle,
        num_frozen_references=len(references),
        threshold=threshold,
    )
