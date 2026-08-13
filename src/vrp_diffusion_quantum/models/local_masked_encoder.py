"""Local encoder biased by the predicted customer route-membership matrix ``M_hat``.

``M_hat`` is customer-customer only, while encoder embeddings include the depot. This module maps
the customer adjacency back to full node indices and keeps it a structural prior: feasibility
remains the responsibility of the autoregressive decoder's visited and capacity masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

__all__ = [
    "AdjacencyMode",
    "LocalAttentionPrior",
    "LocalMaskedEncoder",
    "LocalMaskedEncoderOutput",
    "build_local_attention_prior",
]

AdjacencyMode = Literal["soft", "hard"]


@dataclass(frozen=True)
class LocalAttentionPrior:
    """Full-node attention structure derived from customer-only ``M_hat``.

    ``weights`` contains the symmetrized predicted customer adjacency plus explicit self/depot
    weights. ``allowed_pairs`` distinguishes structurally meaningful pairs from padding and from
    depot-to-customer pairs, which are always blocked.
    """

    weights: Tensor  # [batch, n_nodes, n_nodes], values in [0, 1]
    allowed_pairs: Tensor  # [batch, n_nodes, n_nodes], bool


@dataclass(frozen=True)
class LocalMaskedEncoderOutput:
    """Local node embeddings and their masked mean graph context."""

    node_embeddings: Tensor  # [batch, n_nodes, embedding_dim]
    graph_embedding: Tensor  # [batch, embedding_dim]


def _validate_prior_inputs(
    m_hat: Tensor,
    customer_node_indices: Tensor,
    customer_mask: Tensor,
    depot_index: Tensor,
    node_mask: Tensor,
) -> tuple[int, int, int]:
    if node_mask.ndim != 2:
        raise ValueError(
            f"node_mask must have shape [batch, n_nodes], got {tuple(node_mask.shape)}"
        )
    batch_size, n_nodes = node_mask.shape
    if customer_mask.ndim != 2 or customer_mask.shape[0] != batch_size:
        raise ValueError(
            f"customer_mask must have shape [batch, n_customers], got {tuple(customer_mask.shape)}"
        )
    n_customers = customer_mask.shape[1]
    if customer_node_indices.shape != (batch_size, n_customers):
        raise ValueError(
            "customer_node_indices must have shape "
            f"{(batch_size, n_customers)}, got {tuple(customer_node_indices.shape)}"
        )
    if m_hat.shape != (batch_size, n_customers, n_customers):
        raise ValueError(
            f"m_hat must have shape {(batch_size, n_customers, n_customers)}, "
            f"got {tuple(m_hat.shape)}"
        )
    if depot_index.shape != (batch_size,):
        raise ValueError(
            f"depot_index must have shape {(batch_size,)}, got {tuple(depot_index.shape)}"
        )
    if n_nodes < 1:
        raise ValueError("each CVRP graph must contain at least the depot")
    if not torch.is_floating_point(m_hat):
        raise ValueError("m_hat must use a floating-point dtype")
    if not torch.isfinite(m_hat).all() or torch.any(m_hat < 0) or torch.any(m_hat > 1):
        raise ValueError("m_hat must contain finite probabilities in [0, 1]")
    for name, indices in (
        ("customer_node_indices", customer_node_indices),
        ("depot_index", depot_index),
    ):
        if indices.is_floating_point() or indices.dtype == torch.bool:
            raise ValueError(f"{name} must use an integer dtype")

    devices = {
        m_hat.device,
        customer_node_indices.device,
        customer_mask.device,
        depot_index.device,
        node_mask.device,
    }
    if len(devices) != 1:
        raise ValueError("m_hat and all index/mask tensors must be on the same device")

    real_customers = customer_mask.to(dtype=torch.bool)
    real_nodes = node_mask.to(dtype=torch.bool)
    if torch.any(depot_index < 0) or torch.any(depot_index >= n_nodes):
        raise ValueError(f"depot_index values must be in [0, {n_nodes})")
    batch_indices = torch.arange(batch_size, device=node_mask.device)
    depot_long = depot_index.to(dtype=torch.long)
    if not torch.all(real_nodes[batch_indices, depot_long]):
        raise ValueError("the depot must be marked as a real node in node_mask")

    valid_indices = customer_node_indices[real_customers]
    if torch.any(valid_indices < 0) or torch.any(valid_indices >= n_nodes):
        raise ValueError(f"real customer_node_indices values must be in [0, {n_nodes})")
    padded_indices = customer_node_indices[~real_customers]
    if torch.any(padded_indices != -1):
        raise ValueError("padded customer_node_indices entries must be -1")

    return batch_size, n_nodes, n_customers


def build_local_attention_prior(
    m_hat: Tensor,
    customer_node_indices: Tensor,
    customer_mask: Tensor,
    depot_index: Tensor,
    node_mask: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> LocalAttentionPrior:
    """Map customer-only ``M_hat`` to an attention prior over the full CVRP graph.

    Customer-customer values are symmetrized. Every real node receives a self-loop. Customer
    queries may attend to the depot, but the depot attends only to itself; this supplies depot
    context without allowing separate predicted route groups to exchange information through a
    depot hub. Padded pairs are always disallowed.
    """
    _, n_nodes, n_customers = _validate_prior_inputs(
        m_hat,
        customer_node_indices,
        customer_mask,
        depot_index,
        node_mask,
    )
    output_dtype = dtype if dtype is not None else m_hat.dtype
    real_customers = customer_mask.to(dtype=torch.bool)
    real_nodes = node_mask.to(dtype=torch.bool)

    safe_customer_indices = customer_node_indices.clamp(min=0).to(dtype=torch.long)
    customer_to_node = functional.one_hot(safe_customer_indices, num_classes=n_nodes).to(
        output_dtype
    )
    customer_to_node = customer_to_node * real_customers.unsqueeze(-1)
    mapped_customer_counts = customer_to_node.sum(dim=1)
    if torch.any(mapped_customer_counts > 1):
        raise ValueError("real customer_node_indices must be unique within each instance")

    depot_nodes = functional.one_hot(depot_index.to(dtype=torch.long), num_classes=n_nodes).to(
        output_dtype
    )
    if torch.any((mapped_customer_counts * depot_nodes) > 0):
        raise ValueError("customer_node_indices must not contain the depot")
    mapped_real_nodes = mapped_customer_counts.to(dtype=torch.bool) | depot_nodes.to(
        dtype=torch.bool
    )
    if not torch.equal(mapped_real_nodes, real_nodes):
        raise ValueError("node_mask must contain exactly one depot plus all mapped customers")

    pair_customer_mask = real_customers[:, :, None] & real_customers[:, None, :]
    symmetric_m_hat = 0.5 * (m_hat + m_hat.transpose(-1, -2))
    off_diagonal = ~torch.eye(n_customers, dtype=torch.bool, device=m_hat.device)
    symmetric_m_hat = symmetric_m_hat * pair_customer_mask * off_diagonal
    symmetric_m_hat = symmetric_m_hat.to(dtype=output_dtype)

    customer_weights = torch.einsum(
        "bci,bcd,bdj->bij",
        customer_to_node,
        symmetric_m_hat,
        customer_to_node,
    )
    full_customer_nodes = mapped_customer_counts.to(dtype=torch.bool)
    customer_pairs = full_customer_nodes[:, :, None] & full_customer_nodes[:, None, :]
    customer_to_depot = (
        full_customer_nodes[:, :, None] & depot_nodes.to(dtype=torch.bool)[:, None, :]
    )
    self_pairs = torch.eye(n_nodes, dtype=torch.bool, device=m_hat.device).unsqueeze(0)
    self_pairs = self_pairs & real_nodes[:, :, None]
    allowed_pairs = customer_pairs | customer_to_depot | self_pairs

    weights = customer_weights
    weights = weights + customer_to_depot.to(dtype=output_dtype)
    weights = weights + self_pairs.to(dtype=output_dtype)
    weights = weights * allowed_pairs.to(dtype=output_dtype)
    return LocalAttentionPrior(weights=weights, allowed_pairs=allowed_pairs)


class _PriorMaskedMultiHeadAttention(nn.Module):
    """Multi-head attention with soft weighted or hard thresholded adjacency."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: Tensor,
        prior: LocalAttentionPrior,
        *,
        adjacency_mode: AdjacencyMode,
        hard_threshold: float,
        soft_bias_strength: float,
        minimum_soft_weight: float,
    ) -> Tensor:
        batch_size, n_nodes, embedding_dim = node_embeddings.shape

        def project(layer: nn.Linear) -> Tensor:
            projected: Tensor = layer(node_embeddings)
            projected = projected.view(batch_size, n_nodes, self.num_heads, self.head_dim)
            return projected.transpose(1, 2)

        query = project(self.query_projection)
        key = project(self.key_projection)
        value = project(self.value_projection)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if adjacency_mode == "soft":
            soft_weights = prior.weights.clamp_min(minimum_soft_weight)
            logits = logits + soft_bias_strength * torch.log(soft_weights[:, None, :, :])
            attention_mask = prior.allowed_pairs
        else:
            attention_mask = prior.allowed_pairs & (prior.weights >= hard_threshold)
        logits = logits.masked_fill(~attention_mask[:, None, :, :], torch.finfo(logits.dtype).min)

        attention = self.attention_dropout(torch.softmax(logits, dim=-1))
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, n_nodes, embedding_dim)
        output: Tensor = self.output_projection(attended)
        return output


class _LocalMaskedAttentionLayer(nn.Module):
    """Pre-norm masked attention and feed-forward block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = _PriorMaskedMultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: Tensor,
        node_mask: Tensor,
        prior: LocalAttentionPrior,
        *,
        adjacency_mode: AdjacencyMode,
        hard_threshold: float,
        soft_bias_strength: float,
        minimum_soft_weight: float,
    ) -> Tensor:
        attended = self.attention(
            self.attention_norm(node_embeddings),
            prior,
            adjacency_mode=adjacency_mode,
            hard_threshold=hard_threshold,
            soft_bias_strength=soft_bias_strength,
            minimum_soft_weight=minimum_soft_weight,
        )
        node_embeddings = node_embeddings + self.dropout(attended)
        node_embeddings = node_embeddings + self.dropout(
            self.feed_forward(self.feed_forward_norm(node_embeddings))
        )
        return node_embeddings * node_mask.to(dtype=node_embeddings.dtype).unsqueeze(-1)


class LocalMaskedEncoder(nn.Module):
    """Refine global node embeddings using predicted route-membership adjacency.

    In ``soft`` mode (the default), ``M_hat`` contributes an additive log-probability attention
    bias and retains gradients. In ``hard`` mode, customer edges below ``hard_threshold`` are
    removed; this is useful for hard-mask ablations but can amplify matrix prediction errors.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        feed_forward_dim: int = 512,
        dropout: float = 0.0,
        adjacency_mode: AdjacencyMode = "soft",
        hard_threshold: float = 0.5,
        soft_bias_strength: float = 1.0,
        minimum_soft_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        if feed_forward_dim < 1:
            raise ValueError(f"feed_forward_dim must be >= 1, got {feed_forward_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if adjacency_mode not in ("soft", "hard"):
            raise ValueError(f"adjacency_mode must be 'soft' or 'hard', got {adjacency_mode!r}")
        if not 0.0 < hard_threshold <= 1.0:
            raise ValueError(f"hard_threshold must be in (0, 1], got {hard_threshold}")
        if soft_bias_strength < 0.0:
            raise ValueError(f"soft_bias_strength must be >= 0, got {soft_bias_strength}")
        if not 0.0 < minimum_soft_weight <= 1.0:
            raise ValueError(f"minimum_soft_weight must be in (0, 1], got {minimum_soft_weight}")

        self.embedding_dim = embedding_dim
        self.adjacency_mode: AdjacencyMode = adjacency_mode
        self.hard_threshold = hard_threshold
        self.soft_bias_strength = soft_bias_strength
        self.minimum_soft_weight = minimum_soft_weight
        self.layers = nn.ModuleList(
            _LocalMaskedAttentionLayer(embedding_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        global_node_embeddings: Tensor,
        m_hat: Tensor,
        customer_node_indices: Tensor,
        customer_mask: Tensor,
        depot_index: Tensor,
        node_mask: Tensor,
    ) -> LocalMaskedEncoderOutput:
        """Return local embeddings in the same full-node order as the global encoder output."""
        if global_node_embeddings.ndim != 3:
            raise ValueError(
                "global_node_embeddings must have shape [batch, n_nodes, embedding_dim], got "
                f"{tuple(global_node_embeddings.shape)}"
            )
        batch_size, n_nodes, embedding_dim = global_node_embeddings.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(f"expected embedding_dim {self.embedding_dim}, got {embedding_dim}")
        if node_mask.shape != (batch_size, n_nodes):
            raise ValueError(
                f"node_mask must have shape {(batch_size, n_nodes)}, got {tuple(node_mask.shape)}"
            )
        if global_node_embeddings.device != m_hat.device:
            raise ValueError("global_node_embeddings and m_hat must be on the same device")
        if not torch.isfinite(global_node_embeddings).all():
            raise ValueError("global_node_embeddings must contain only finite values")

        prior = build_local_attention_prior(
            m_hat,
            customer_node_indices,
            customer_mask,
            depot_index,
            node_mask,
            dtype=global_node_embeddings.dtype,
        )
        mask = node_mask.to(dtype=torch.bool)
        node_embeddings = global_node_embeddings * mask.unsqueeze(-1)
        for layer in self.layers:
            node_embeddings = layer(
                node_embeddings,
                mask,
                prior,
                adjacency_mode=self.adjacency_mode,
                hard_threshold=self.hard_threshold,
                soft_bias_strength=self.soft_bias_strength,
                minimum_soft_weight=self.minimum_soft_weight,
            )
        node_embeddings = self.output_norm(node_embeddings) * mask.unsqueeze(-1)

        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(node_embeddings.dtype)
        graph_embedding = node_embeddings.sum(dim=1) / valid_counts
        return LocalMaskedEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=graph_embedding,
        )
