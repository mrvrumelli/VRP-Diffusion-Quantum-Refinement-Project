"""Common graph-level metrics and paired uncertainty for corrective comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vrp_diffusion_quantum.eval.confidence import BootstrapInterval


@dataclass(frozen=True)
class InstanceResult:
    """One generated graph, with counts over ordered off-diagonal customer pairs."""

    instance_id: str
    n_customers: int
    true_positive: int
    false_positive: int
    false_negative: int
    decoded_cost: float
    reference_cost: float
    feasible: bool
    runtime_seconds: float


Metric = Literal["f1", "mean_instance_gap_percent", "ratio_total_gap_percent"]


def cost_gaps(costs: Sequence[float], references: Sequence[float]) -> dict[str, float]:
    """Return both gap definitions; never silently drop invalid reference costs."""
    cost = np.asarray(costs, dtype=np.float64)
    reference = np.asarray(references, dtype=np.float64)
    if cost.ndim != 1 or cost.size == 0 or cost.shape != reference.shape:
        raise ValueError("costs and references must be nonempty equal-length vectors")
    if not np.all(np.isfinite(cost)) or not np.all(np.isfinite(reference)):
        raise ValueError("costs and references must be finite")
    if np.any(reference <= 0):
        raise ValueError("reference costs must be positive")
    return {
        "mean_instance_gap_percent": float(np.mean(100.0 * (cost - reference) / reference)),
        "ratio_total_gap_percent": float(100.0 * (cost.sum() - reference.sum()) / reference.sum()),
    }


def metric_value(records: Sequence[InstanceResult], metric: Metric) -> float:
    """Aggregate graph records, requiring feasible routes for route comparisons."""
    if not records:
        raise ValueError("records must be nonempty")
    if metric == "f1":
        tp = sum(record.true_positive for record in records)
        denominator = 2 * tp + sum(
            record.false_positive + record.false_negative for record in records
        )
        return 2 * tp / denominator if denominator else 0.0
    if not all(record.feasible for record in records):
        raise ValueError("route comparison contains infeasible results; report feasibility first")
    return cost_gaps(
        [record.decoded_cost for record in records], [record.reference_cost for record in records]
    )[metric]


def paired_bootstrap(
    candidate: Sequence[InstanceResult],
    baseline: Sequence[InstanceResult] | None = None,
    *,
    metric: Metric = "f1",
    seed: int = 0,
    num_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    """Stratified graph bootstrap; paired deltas are candidate minus baseline.

    Resample graphs inside each size, maintaining the panel's size mix. Recompute pooled
    F1 from counts for each resample; neither edges nor repeated seeds are independent graphs.
    """
    if not candidate or num_resamples < 1 or not 0 < confidence_level < 1:
        raise ValueError("invalid panel or bootstrap settings")
    ids = [record.instance_id for record in candidate]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate graph IDs; evaluate sampling seeds separately")
    paired = None
    if baseline is not None:
        lookup = {record.instance_id: record for record in baseline}
        if len(lookup) != len(baseline) or set(lookup) != set(ids):
            raise ValueError("paired panels must contain the same unique graph IDs")
        paired = [lookup[key] for key in ids]
        if any(
            left.n_customers != right.n_customers or left.reference_cost != right.reference_cost
            for left, right in zip(candidate, paired, strict=True)
        ):
            raise ValueError("paired panels must use the same sizes and references")
    estimate = metric_value(candidate, metric)
    if paired is not None:
        estimate -= metric_value(paired, metric)
    groups = [
        [index for index, record in enumerate(candidate) if record.n_customers == size]
        for size in sorted({record.n_customers for record in candidate})
    ]
    rng = np.random.default_rng(seed)
    values = np.empty(num_resamples)
    for draw in range(num_resamples):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values[draw] = metric_value([candidate[index] for index in indices], metric)
        if paired is not None:
            values[draw] -= metric_value([paired[index] for index in indices], metric)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha])
    return BootstrapInterval(
        estimate, float(lower), float(upper), confidence_level, num_resamples, len(candidate), seed
    )


def assert_disjoint_sources(training: set[str], evaluation: set[str]) -> None:
    """Fail closed on any shared source identity, including alternative references."""
    overlap = training & evaluation
    if overlap:
        raise ValueError(f"training/evaluation source overlap: {sorted(overlap)[:5]}")
