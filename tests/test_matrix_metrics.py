import numpy as np
import pytest

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.metrics.matrix_metrics import (
    MatrixPrediction,
    binary_cross_entropy,
    capacity_consistency_proxy,
    compute_matrix_metrics,
    expected_calibration_error,
    off_diagonal_pairs,
    precision_recall_f1,
    roc_auc,
)


def test_off_diagonal_pairs_excludes_diagonal() -> None:
    matrix = np.array([[9, 1, 2], [3, 9, 4], [5, 6, 9]])
    pairs = off_diagonal_pairs(matrix)
    assert sorted(pairs.tolist()) == [1, 2, 3, 4, 5, 6]


def test_binary_cross_entropy_zero_for_perfect_predictions() -> None:
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    assert binary_cross_entropy(y_prob, y_true) == pytest.approx(0.0, abs=1e-5)


def test_binary_cross_entropy_higher_for_wrong_predictions() -> None:
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    correct = np.array([0.99, 0.01, 0.99, 0.01])
    wrong = np.array([0.01, 0.99, 0.01, 0.99])
    assert binary_cross_entropy(correct, y_true) < binary_cross_entropy(wrong, y_true)


def test_roc_auc_perfect_separation() -> None:
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])
    assert roc_auc(y_prob, y_true) == pytest.approx(1.0)


def test_roc_auc_returns_none_for_single_class() -> None:
    y_prob = np.array([0.1, 0.2, 0.3])
    y_true = np.array([0, 0, 0])
    assert roc_auc(y_prob, y_true) is None


def test_precision_recall_f1_hand_checked() -> None:
    # true positives: idx 0, 2; false positive: idx 1; false negative: idx 3
    y_true = np.array([1, 0, 1, 1])
    y_prob = np.array([0.9, 0.9, 0.9, 0.1])

    result = precision_recall_f1(y_prob, y_true, threshold=0.5)

    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)


def test_select_best_f1_threshold_prefers_higher_f1() -> None:
    from vrp_diffusion_quantum.metrics.matrix_metrics import select_best_f1_threshold

    # At 0.5: many FPs; at 0.8: cleaner positives
    y_true = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
    y_prob = np.array([0.9, 0.6, 0.55, 0.52, 0.85, 0.1], dtype=np.float64)
    thr, pr = select_best_f1_threshold(y_prob, y_true, thresholds=(0.5, 0.8))
    assert thr == 0.8
    assert pr.f1 >= precision_recall_f1(y_prob, y_true, threshold=0.5).f1


def test_default_f1_threshold_grid_span() -> None:
    from vrp_diffusion_quantum.metrics.matrix_metrics import DEFAULT_F1_THRESHOLD_GRID

    assert DEFAULT_F1_THRESHOLD_GRID[0] == pytest.approx(0.05)
    assert DEFAULT_F1_THRESHOLD_GRID[-1] == pytest.approx(0.95)
    assert len(DEFAULT_F1_THRESHOLD_GRID) == 19


def test_expected_calibration_error_zero_when_perfectly_calibrated() -> None:
    # 10 samples at p=0.3 with exactly 3 positives -> confidence matches observed frequency
    y_prob = np.full(10, 0.3)
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    assert expected_calibration_error(y_prob, y_true, num_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_positive_when_miscalibrated() -> None:
    y_prob = np.full(10, 0.9)
    y_true = np.zeros(10)
    assert expected_calibration_error(y_prob, y_true, num_bins=10) == pytest.approx(0.9)


def test_capacity_consistency_proxy_all_feasible() -> None:
    # two disjoint pairs, each pair's demand fits under capacity
    m_prob = np.array(
        [
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9],
            [0.0, 0.0, 0.9, 0.0],
        ]
    )
    customer_demands = np.array([2.0, 3.0, 2.0, 3.0])
    proxy = capacity_consistency_proxy(m_prob, customer_demands, capacity=10.0, threshold=0.5)
    assert proxy == pytest.approx(1.0)


def test_capacity_consistency_proxy_partial_violation() -> None:
    # cluster {0, 1} demand = 9 <= 10 (feasible); cluster {2, 3} demand = 12 > 10 (infeasible)
    m_prob = np.array(
        [
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9],
            [0.0, 0.0, 0.9, 0.0],
        ]
    )
    customer_demands = np.array([4.0, 5.0, 6.0, 6.0])
    proxy = capacity_consistency_proxy(m_prob, customer_demands, capacity=10.0, threshold=0.5)
    assert proxy == pytest.approx(0.5)


def test_capacity_consistency_proxy_empty_instance_returns_one() -> None:
    m_prob = np.zeros((0, 0))
    customer_demands = np.zeros(0)
    assert capacity_consistency_proxy(m_prob, customer_demands, capacity=10.0) == 1.0


def _tiny_example_prediction(m_prob_value: float) -> MatrixPrediction:
    instance = CVRPInstance(
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        demands=np.array([0.0, 3.0, 4.0, 2.0]),
        capacity=10.0,
        depot_index=0,
        instance_id="toy",
        n_customers=3,
        seed=0,
        generator_settings={"kind": "hand_crafted"},
    )
    solution = LabeledSolution(
        routes=[[0, 1], [2]],
        cost=12.5,
        num_vehicles=2,
        feasible=True,
        solver_name="hand_checked",
        time_budget=None,
        seed=0,
        runtime_seconds=0.001,
    )
    example = make_example(instance, solution)
    m_prob = np.where(example.constraint_matrix == 1, m_prob_value, 1 - m_prob_value)
    np.fill_diagonal(m_prob, 0.0)
    return MatrixPrediction.from_example(example, m_prob)


def test_compute_matrix_metrics_rejects_empty_predictions() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_matrix_metrics([])


def test_compute_matrix_metrics_on_near_perfect_predictions() -> None:
    predictions = [_tiny_example_prediction(0.95), _tiny_example_prediction(0.9)]

    metrics = compute_matrix_metrics(predictions)

    assert metrics.bce < 0.5
    assert metrics.auc == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.num_pairs == 3 * 2 * 2  # 3x3 matrix, 6 off-diagonal pairs, 2 examples
    assert metrics.num_positive_pairs == 2 * 2  # customers 0,1 share a route -> 2 pairs (i,j)+(j,i)
