import numpy as np
import pytest

from vrp_diffusion_quantum.eval.ambiguity import evaluate_oracle_reference_agreement
from vrp_diffusion_quantum.utils.constraint_matrix import build_constraint_matrix


def test_oracle_selects_one_reference_for_all_metrics() -> None:
    canonical = build_constraint_matrix([[0, 1], [2, 3]], 4)
    alternative = build_constraint_matrix([[0, 2], [1, 3]], 4)
    prediction = alternative.astype(float) * 0.9 + (1 - alternative) * 0.1
    result = evaluate_oracle_reference_agreement(
        prediction, canonical, [canonical, alternative], threshold=0.5
    )
    assert result.canonical.f1 == 0.0
    assert result.oracle.reference_index == 1
    assert result.oracle.f1 == 1.0
    assert result.oracle.bce < result.canonical.bce
    assert result.num_frozen_references == 2


def test_oracle_ties_choose_lowest_reference_index() -> None:
    reference = build_constraint_matrix([[0, 1], [2]], 3)
    result = evaluate_oracle_reference_agreement(
        reference.astype(float), reference, [reference, reference]
    )
    assert result.oracle.reference_index == 0


def test_oracle_rejects_invalid_or_empty_references() -> None:
    reference = build_constraint_matrix([[0, 1]], 2)
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_oracle_reference_agreement(reference, reference, [])
    with pytest.raises(ValueError, match="symmetric"):
        evaluate_oracle_reference_agreement(reference, reference, [np.array([[0, 1], [0, 0]])])
