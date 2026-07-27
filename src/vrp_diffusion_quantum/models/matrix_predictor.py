"""Simple supervised constraint-matrix predictor, without diffusion (task P2.1).

Exists to de-risk the diffusion model (`AGENTS.md` Phase 2): a small pairwise-feature MLP that
predicts `m_prob`, the same route-membership probability matrix the diffusion model will later
predict, trained directly with a binary-cross-entropy loss against `m_true` from
`vrp_diffusion_quantum.utils.constraint_matrix.build_constraint_matrix` (task P1.3).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn.functional as ff
from torch import nn

from vrp_diffusion_quantum.data.types import CVRPExample

logger = logging.getLogger(__name__)

_PAIR_FEATURE_DIM = 8
_LOSS_EPS = 1e-7


def _backward(loss: torch.Tensor) -> None:
    cast(Any, loss).backward()


def build_pair_features(
    customer_coords: torch.Tensor, customer_demands: torch.Tensor, capacity: float
) -> torch.Tensor:
    """Build per-pair (i, j) features for every customer pair.

    Features are `[coords_i, coords_j, distance_ij, demand_i, demand_j, (demand_i + demand_j) /
    capacity]`. Returns shape `[n_customers, n_customers, 8]`.
    """
    n_customers = customer_coords.shape[0]
    coords_i = customer_coords.unsqueeze(1).expand(n_customers, n_customers, 2)
    coords_j = customer_coords.unsqueeze(0).expand(n_customers, n_customers, 2)
    distance = torch.linalg.norm(coords_i - coords_j, dim=-1, keepdim=True)
    demand_i = customer_demands.unsqueeze(1).expand(n_customers, n_customers).unsqueeze(-1)
    demand_j = customer_demands.unsqueeze(0).expand(n_customers, n_customers).unsqueeze(-1)
    demand_sum_ratio = (demand_i + demand_j) / capacity
    return torch.cat([coords_i, coords_j, distance, demand_i, demand_j, demand_sum_ratio], dim=-1)


class MatrixPredictor(nn.Module):
    """Pairwise-feature MLP baseline for predicting the route-membership matrix M."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(_PAIR_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, customer_coords: torch.Tensor, customer_demands: torch.Tensor, capacity: float
    ) -> torch.Tensor:
        """Predict `m_prob`: shape `[n_customers, n_customers]`, symmetric, zero diagonal, in
        `[0, 1]`.
        """
        n_customers = customer_coords.shape[0]
        pair_features = build_pair_features(customer_coords, customer_demands, capacity)
        logits = self.pair_mlp(pair_features).squeeze(-1)
        symmetric_logits = (logits + logits.T) / 2
        m_prob = torch.sigmoid(symmetric_logits)
        off_diagonal = 1.0 - torch.eye(n_customers, device=m_prob.device, dtype=m_prob.dtype)
        return m_prob * off_diagonal


def matrix_bce_loss(m_prob: torch.Tensor, m_true: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy between `m_prob` and `m_true`, over off-diagonal entries only.

    The diagonal is excluded: `M[i, i] = 0` by construction (`AGENTS.md` section 7), so it
    carries no training signal.
    """
    n_customers = m_prob.shape[0]
    off_diagonal_mask = ~torch.eye(n_customers, dtype=torch.bool, device=m_prob.device)
    clamped_m_prob = torch.clamp(m_prob, _LOSS_EPS, 1.0 - _LOSS_EPS)
    return ff.binary_cross_entropy(
        clamped_m_prob[off_diagonal_mask], m_true[off_diagonal_mask].float()
    )


def train_matrix_predictor(
    model: MatrixPredictor,
    examples: list[CVRPExample],
    num_epochs: int,
    learning_rate: float,
) -> list[float]:
    """Train `model` on `examples`, one full-batch gradient step per example per epoch.

    Examples are not padded into a single tensor batch since `n_customers` varies across them.
    Returns the mean training loss for each epoch.

    Caller is responsible for seeding `torch` (e.g. `torch.manual_seed(seed)`) before
    constructing `model`, so weight initialization stays reproducible, per
    docs/coding_standards.md section 5.
    """
    if not examples:
        raise ValueError("cannot train on an empty list of examples")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        for example in examples:
            optimizer.zero_grad()
            customer_coords = torch.from_numpy(example.instance.customer_coords()).float()
            customer_demands = torch.from_numpy(example.instance.customer_demands()).float()
            m_true = torch.from_numpy(example.constraint_matrix).float()

            m_prob = model(customer_coords, customer_demands, example.instance.capacity)
            loss = matrix_bce_loss(m_prob, m_true)
            _backward(loss)
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(examples)
        epoch_losses.append(mean_loss)
        logger.info("epoch=%d matrix_bce_loss=%.6f", epoch, mean_loss)

    return epoch_losses
