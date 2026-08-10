"""Matrix metrics for evaluating predicted constraint matrices (task P2.2).

Metrics score `m_prob` (predicted route-membership probabilities) against `m_true` (ground
truth from `vrp_diffusion_quantum.utils.constraint_matrix.build_constraint_matrix`), restricted
to off-diagonal customer pairs since the diagonal is fixed at 0 by construction (`AGENTS.md`
section 7). `compute_matrix_metrics` is the entry point for scoring a validation batch: build one
`MatrixPrediction` per example (`MatrixPrediction.from_example` bridges a `CVRPExample` and a
predicted `m_prob`) and pass the list in.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from vrp_diffusion_quantum.data.types import CVRPExample

_BCE_EPS = 1e-7


def off_diagonal_pairs(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Flatten the off-diagonal entries of a square matrix into a 1D array."""
    n = matrix.shape[0]
    off_diagonal_mask = ~np.eye(n, dtype=bool)
    return matrix[off_diagonal_mask]


def binary_cross_entropy(y_prob: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]) -> float:
    """Mean binary cross-entropy between predicted probabilities and binary labels."""
    clamped_prob = np.clip(y_prob, _BCE_EPS, 1.0 - _BCE_EPS)
    label = y_true.astype(np.float64)
    return float(-np.mean(label * np.log(clamped_prob) + (1 - label) * np.log(1 - clamped_prob)))


def roc_auc(y_prob: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]) -> float | None:
    """ROC-AUC, or `None` if `y_true` has only one class (AUC is undefined)."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


@dataclass(frozen=True)
class PrecisionRecallF1:
    """Precision, recall, F1, and accuracy at a fixed decision threshold."""

    precision: float
    recall: float
    f1: float
    accuracy: float


def precision_recall_f1(
    y_prob: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64], threshold: float = 0.5
) -> PrecisionRecallF1:
    """Precision, recall, F1, and accuracy after thresholding `y_prob` at `threshold`."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_bin = y_true.astype(np.int64)
    return PrecisionRecallF1(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(np.mean(y_pred == y_bin)),
    )


DEFAULT_F1_THRESHOLD_GRID = tuple(float(t) for t in np.linspace(0.05, 0.95, 19))


def select_best_f1_threshold(
    y_prob: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.float64],
    *,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_F1_THRESHOLD_GRID,
) -> tuple[float, PrecisionRecallF1]:
    """Pick the decision threshold that maximizes F1 on ``(y_prob, y_true)``."""
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    best_t = float(thresholds[0])
    best_pr = precision_recall_f1(y_prob, y_true, threshold=best_t)
    for t in thresholds[1:]:
        pr = precision_recall_f1(y_prob, y_true, threshold=float(t))
        if pr.f1 > best_pr.f1:
            best_t = float(t)
            best_pr = pr
    return best_t, best_pr


def expected_calibration_error(
    y_prob: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64], num_bins: int = 10
) -> float:
    """Weighted mean gap between predicted confidence and observed frequency, per probability bin.

    Splits `[0, 1]` into `num_bins` equal-width bins. For each non-empty bin, compares the mean
    predicted probability (confidence) to the mean true label (observed positive frequency), and
    returns the bin-count-weighted mean absolute gap. 0 means perfectly calibrated.
    """
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, num_bins - 1)
    total = len(y_prob)

    error = 0.0
    for bin_index in range(num_bins):
        in_bin = bin_indices == bin_index
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        confidence = float(np.mean(y_prob[in_bin]))
        observed_frequency = float(np.mean(y_true[in_bin]))
        error += (count / total) * abs(confidence - observed_frequency)

    return error


def capacity_consistency_proxy(
    m_prob: npt.NDArray[np.float64],
    customer_demands: npt.NDArray[np.float64],
    capacity: float,
    threshold: float = 0.5,
) -> float:
    """Fraction of customers whose predicted same-route cluster fits within vehicle capacity.

    Thresholds `m_prob` at `threshold` to build a same-route adjacency graph, takes its connected
    components as candidate clusters, and returns the fraction of customers whose cluster's total
    demand does not exceed `capacity`. This is a proxy on the raw predicted matrix, not a
    feasibility guarantee: the decoder, not `M`, is responsible for final route feasibility
    (`AGENTS.md` section 2).
    """
    n_customers = m_prob.shape[0]
    if n_customers == 0:
        return 1.0

    graph = nx.Graph()
    graph.add_nodes_from(range(n_customers))
    off_diagonal_mask = ~np.eye(n_customers, dtype=bool)
    above_threshold = np.argwhere((m_prob >= threshold) & off_diagonal_mask)
    graph.add_edges_from(above_threshold.tolist())

    feasible_customers = 0
    for component in nx.connected_components(graph):
        component_demand = float(np.sum(customer_demands[list(component)]))
        if component_demand <= capacity:
            feasible_customers += len(component)

    return float(feasible_customers / n_customers)


@dataclass(frozen=True)
class MatrixPrediction:
    """One example's predicted vs. true constraint matrix, for metric computation."""

    m_prob: npt.NDArray[np.float64]  # [n_customers, n_customers]
    m_true: npt.NDArray[np.float64]  # [n_customers, n_customers]
    customer_demands: npt.NDArray[np.float64]  # [n_customers]
    capacity: float

    @classmethod
    def from_example(
        cls, example: CVRPExample, m_prob: npt.NDArray[np.float64]
    ) -> MatrixPrediction:
        """Build a `MatrixPrediction` from a labeled `CVRPExample` and a predicted `m_prob`.

        `m_prob` must already be a numpy array; convert a torch tensor with
        `m_prob.detach().cpu().numpy()` first.
        """
        return cls(
            m_prob=np.asarray(m_prob, dtype=np.float64),
            m_true=example.constraint_matrix.astype(np.float64),
            customer_demands=example.instance.customer_demands(),
            capacity=example.instance.capacity,
        )


@dataclass(frozen=True)
class MatrixMetrics:
    """Matrix metrics computed over a validation batch of `MatrixPrediction`s."""

    bce: float
    auc: float | None
    precision: float
    recall: float
    f1: float
    accuracy: float
    calibration_error: float
    capacity_consistency: float
    num_pairs: int
    num_positive_pairs: int
    threshold: float


def compute_matrix_metrics(
    predictions: list[MatrixPrediction],
    *,
    threshold: float | None = None,
    adaptive_threshold: bool = True,
    threshold_grid: tuple[float, ...] | list[float] = DEFAULT_F1_THRESHOLD_GRID,
    num_calibration_bins: int = 10,
) -> MatrixMetrics:
    """Compute matrix metrics for a validation batch of `MatrixPrediction`s.

    BCE/AUC/calibration pool off-diagonal pairs (threshold-free). Hard metrics
    (precision/recall/F1/accuracy/capacity) use ``threshold``. When
    ``adaptive_threshold`` is true (default), sweep ``threshold_grid`` and pick the
    value that maximizes F1; pass ``adaptive_threshold=False`` with an explicit
    ``threshold`` to fix binarization.
    """
    if not predictions:
        raise ValueError("cannot compute matrix metrics for an empty list of predictions")

    y_prob = np.concatenate([off_diagonal_pairs(p.m_prob) for p in predictions])
    y_true = np.concatenate([off_diagonal_pairs(p.m_true) for p in predictions])
    if y_prob.size == 0:
        raise ValueError(
            "no off-diagonal customer pairs to score (every example has <= 1 customer)"
        )

    if adaptive_threshold or threshold is None:
        used_threshold, precision_recall = select_best_f1_threshold(
            y_prob, y_true, thresholds=threshold_grid
        )
    else:
        used_threshold = float(threshold)
        precision_recall = precision_recall_f1(y_prob, y_true, threshold=used_threshold)

    capacity_consistency = float(
        np.mean(
            [
                capacity_consistency_proxy(
                    p.m_prob, p.customer_demands, p.capacity, threshold=used_threshold
                )
                for p in predictions
            ]
        )
    )

    return MatrixMetrics(
        bce=binary_cross_entropy(y_prob, y_true),
        auc=roc_auc(y_prob, y_true),
        precision=precision_recall.precision,
        recall=precision_recall.recall,
        f1=precision_recall.f1,
        accuracy=precision_recall.accuracy,
        calibration_error=expected_calibration_error(y_prob, y_true, num_bins=num_calibration_bins),
        capacity_consistency=capacity_consistency,
        num_pairs=int(y_prob.size),
        num_positive_pairs=int(np.sum(y_true)),
        threshold=float(used_threshold),
    )
