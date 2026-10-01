"""Diffusion-biased QUBO terms (task Q1.5).

The diffusion prior predicts, for every customer pair, the probability ``p_ij`` that the two
customers share a route (``m_prob``); any pair-probability matrix in ``[0, 1]`` of shape
``[n_customers, n_customers]`` can be used. These terms nudge the refinement QUBOs toward
structures the prior believes in. They are optional, off by default, and weighted by a single
``alpha``.

Pair costs (all in ``[0, 1]`` before scaling):

* ``mode="probability"``: ``cost(p) = 1 - p`` and ``signed(p) = 1 - 2p``.
* ``mode="confidence"``: both are multiplied by the confidence ``|2p - 1|``, so uncertain
  predictions (``p`` near 0.5) contribute little.

Reorder QUBO: each transition ``a -> b`` (including from a fixed start customer and to a fixed end
customer) gains ``alpha * s * cost(p_ab)``, where ``s`` is the mean distance in the subproblem.

Exchange QUBO: assigning movable customer ``i`` to route ``R`` gains
``alpha * s * cost(affinity_iR)``, with ``affinity_iR`` the mean ``p`` to ``R``'s fixed customers
(0.5 when there are none). Each movable pair gains
``alpha * s * signed(p_ij) * [y_i == y_j]``, which rewards sharing a route when ``p_ij > 0.5``.
Here ``s`` is the mean insertion cost.

The biased builders size the constraint penalties to cover the bias, so feasibility guarantees of
the unbiased QUBOs are preserved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.quantum.neighborhoods import ExchangeSubproblem, ReorderSubproblem
from vrp_diffusion_quantum.quantum.qubo import QUBO, QUBOBuilder
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    DEFAULT_DISPERSION_WEIGHT,
    ExchangeQUBO,
    build_exchange_qubo,
)
from vrp_diffusion_quantum.quantum.qubo_exchange import (
    DEFAULT_PENALTY_FACTOR as EXCHANGE_PENALTY_FACTOR,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import (
    DEFAULT_PENALTY_FACTOR as REORDER_PENALTY_FACTOR,
)
from vrp_diffusion_quantum.quantum.qubo_reorder import ReorderQUBO, build_reorder_qubo

__all__ = [
    "BiasMode",
    "DiffusionBiasConfig",
    "build_biased_exchange_qubo",
    "build_biased_reorder_qubo",
    "exchange_bias_terms",
    "reorder_bias_terms",
]

BiasMode = Literal["probability", "confidence"]


@dataclass(frozen=True)
class DiffusionBiasConfig:
    """Toggle and weight for diffusion-biased terms."""

    enabled: bool = False
    alpha: float = 0.0
    mode: BiasMode = "probability"

    def __post_init__(self) -> None:
        if self.alpha < 0.0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite and non-negative")
        if self.mode not in ("probability", "confidence"):
            raise ValueError(f"mode must be 'probability' or 'confidence', got {self.mode!r}")

    @property
    def active(self) -> bool:
        return self.enabled and self.alpha > 0.0


def _validate_pair_prob(
    pair_prob: npt.ArrayLike,
    *,
    n_customers: int | None = None,
    customers: Sequence[int] = (),
) -> npt.NDArray[np.float64]:
    matrix = np.asarray(pair_prob, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("pair_prob must be a square matrix")
    if n_customers is not None and matrix.shape != (n_customers, n_customers):
        raise ValueError(f"pair_prob must have shape ({n_customers}, {n_customers})")
    if any(customer < 0 or customer >= matrix.shape[0] for customer in customers):
        raise ValueError("pair_prob shape does not cover the subproblem's customer indices")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("pair_prob must contain probabilities in [0, 1]")
    return matrix


def _cost(probability: float, mode: BiasMode) -> float:
    base = 1.0 - probability
    return base if mode == "probability" else base * abs(2.0 * probability - 1.0)


def _signed(probability: float, mode: BiasMode) -> float:
    base = 1.0 - 2.0 * probability
    return base if mode == "probability" else base * abs(2.0 * probability - 1.0)


def _mean_positive(values: npt.NDArray[np.float64]) -> float:
    positive = values[values > 0.0]
    return float(positive.mean()) if positive.size else 1.0


# --------------------------------------------------------------------------------------------
# Reorder


def reorder_bias_terms(
    reorder: ReorderQUBO, pair_prob: npt.ArrayLike, config: DiffusionBiasConfig
) -> QUBO:
    """Bias-only QUBO over the reorder variables (all zeros when the bias is inactive)."""
    builder = QUBOBuilder(reorder.qubo.labels)
    if not config.active:
        return builder.build()
    subproblem = reorder.subproblem
    involved = [*subproblem.customers]
    involved.extend(
        node for node in (subproblem.start_node, subproblem.end_node) if node is not None
    )
    matrix = _validate_pair_prob(pair_prob, n_customers=subproblem.n_customers, customers=involved)
    k = subproblem.size
    weight = config.alpha * _mean_positive(subproblem.distances)
    customers = subproblem.customers
    for i, first in enumerate(customers):
        if subproblem.start_node is not None:
            p = float(matrix[subproblem.start_node, first])
            builder.add_linear(reorder.variable(i, 0), weight * _cost(p, config.mode))
        if subproblem.end_node is not None:
            p = float(matrix[first, subproblem.end_node])
            builder.add_linear(reorder.variable(i, k - 1), weight * _cost(p, config.mode))
        for j, second in enumerate(customers):
            if i == j:
                continue
            cost = weight * _cost(float(matrix[first, second]), config.mode)
            for position in range(k - 1):
                builder.add_quadratic(
                    reorder.variable(i, position), reorder.variable(j, position + 1), cost
                )
    builder.record_penalty("diffusion_bias_alpha", config.alpha)
    return builder.build()


def build_biased_reorder_qubo(
    subproblem: ReorderSubproblem,
    pair_prob: npt.ArrayLike,
    config: DiffusionBiasConfig,
    *,
    penalty_factor: float = REORDER_PENALTY_FACTOR,
) -> ReorderQUBO:
    """Reorder QUBO plus bias terms; identical to the unbiased QUBO when the bias is off."""
    if not config.active:
        return build_reorder_qubo(subproblem, penalty_factor=penalty_factor)
    extra = config.alpha * _mean_positive(subproblem.distances)  # max bias of one transition
    base = build_reorder_qubo(subproblem, penalty_factor=penalty_factor, extra_edge_weight=extra)
    return ReorderQUBO(
        qubo=base.qubo + reorder_bias_terms(base, pair_prob, config), subproblem=subproblem
    )


# --------------------------------------------------------------------------------------------
# Exchange


def _affinities(
    matrix: npt.NDArray[np.float64], subproblem: ExchangeSubproblem
) -> npt.NDArray[np.float64]:
    """``[m, 2]`` mean pair probability of each movable customer to each route's fixed set."""
    rows = []
    for customer in subproblem.movable:
        rows.append(
            [
                float(matrix[customer, list(fixed)].mean()) if fixed else 0.5
                for fixed in subproblem.fixed
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _exchange_bias_parts(
    exchange: ExchangeQUBO, pair_prob: npt.ArrayLike, config: DiffusionBiasConfig
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Linear costs ``[m, 2]``, signed pair weights ``[m, m]`` and the weight scale."""
    subproblem = exchange.subproblem
    matrix = _validate_pair_prob(
        pair_prob,
        n_customers=subproblem.n_customers,
        customers=[*subproblem.movable, *subproblem.fixed[0], *subproblem.fixed[1]],
    )
    weight = config.alpha * _mean_positive(exchange.insertion_costs)
    affinity = _affinities(matrix, subproblem)
    linear = np.vectorize(lambda p: _cost(float(p), config.mode))(affinity) * weight
    m = subproblem.size
    pairs = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            p = float(matrix[subproblem.movable[i], subproblem.movable[j]])
            pairs[i, j] = weight * _signed(p, config.mode)
    return linear.astype(np.float64), pairs, weight


def exchange_bias_terms(
    exchange: ExchangeQUBO, pair_prob: npt.ArrayLike, config: DiffusionBiasConfig
) -> QUBO:
    """Bias-only QUBO over the exchange variables (slack bits untouched)."""
    builder = QUBOBuilder(exchange.qubo.labels)
    if not config.active:
        return builder.build()
    linear, pairs, _ = _exchange_bias_parts(exchange, pair_prob, config)
    m = exchange.size
    for i in range(m):
        builder.add_linear(i, float(linear[i, 0] - linear[i, 1]))
        builder.add_constant(float(linear[i, 1]))
        for j in range(i + 1, m):
            w = float(pairs[i, j])
            if w == 0.0:
                continue
            builder.add_constant(w)  # [y_i == y_j] = 1 - y_i - y_j + 2 y_i y_j
            builder.add_linear(i, -w)
            builder.add_linear(j, -w)
            builder.add_quadratic(i, j, 2.0 * w)
    builder.record_penalty("diffusion_bias_alpha", config.alpha)
    return builder.build()


def build_biased_exchange_qubo(
    instance: CVRPInstance,
    subproblem: ExchangeSubproblem,
    pair_prob: npt.ArrayLike,
    config: DiffusionBiasConfig,
    *,
    dispersion_weight: float = DEFAULT_DISPERSION_WEIGHT,
    penalty_factor: float = EXCHANGE_PENALTY_FACTOR,
) -> ExchangeQUBO:
    """Exchange QUBO plus bias terms; identical to the unbiased QUBO when the bias is off."""
    if config.active:
        _validate_pair_prob(pair_prob, n_customers=instance.n_customers)
    base = build_exchange_qubo(
        instance, subproblem, dispersion_weight=dispersion_weight, penalty_factor=penalty_factor
    )
    if not config.active:
        return base
    linear, pairs, _ = _exchange_bias_parts(base, pair_prob, config)
    extra_span = float(np.abs(linear[:, 0] - linear[:, 1]).sum() + np.abs(pairs).sum())
    widened = build_exchange_qubo(
        instance,
        subproblem,
        dispersion_weight=dispersion_weight,
        penalty_factor=penalty_factor,
        extra_span=extra_span,
    )
    return ExchangeQUBO(
        qubo=widened.qubo + exchange_bias_terms(widened, pair_prob, config),
        subproblem=widened.subproblem,
        insertion_costs=widened.insertion_costs,
        pair_weights=widened.pair_weights,
        slack_slices=widened.slack_slices,
    )
