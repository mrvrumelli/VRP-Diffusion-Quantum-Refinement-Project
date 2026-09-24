"""Deterministic bootstrap confidence intervals for evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile-bootstrap interval around an arithmetic mean."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    num_resamples: int
    num_observations: int
    seed: int


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    num_resamples: int = 10_000,
    seed: int = 0,
    resample_batch_size: int = 2_048,
) -> BootstrapInterval:
    """Estimate a deterministic percentile-bootstrap interval for a sample mean.

    Resamples are generated in bounded chunks so large evaluation panels do not
    require a ``num_resamples x num_observations`` allocation.
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(sample)):
        raise ValueError("values must contain only finite numbers")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if num_resamples < 1:
        raise ValueError("num_resamples must be >= 1")
    if resample_batch_size < 1:
        raise ValueError("resample_batch_size must be >= 1")

    estimate = float(np.mean(sample))
    if sample.size == 1:
        lower = upper = estimate
    else:
        rng = np.random.default_rng(seed)
        means = np.empty(num_resamples, dtype=np.float64)
        for start in range(0, num_resamples, resample_batch_size):
            stop = min(start + resample_batch_size, num_resamples)
            indices = rng.integers(0, sample.size, size=(stop - start, sample.size))
            means[start:stop] = np.mean(sample[indices], axis=1)
        alpha = 1.0 - confidence_level
        lower, upper = (
            float(value) for value in np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
        )

    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        num_resamples=num_resamples,
        num_observations=int(sample.size),
        seed=seed,
    )
