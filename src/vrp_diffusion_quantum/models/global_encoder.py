"""Global attention encoder over the complete depot-and-customer CVRP graph.

The encoder is intentionally independent of the customer-customer constraint matrix. It gives
the later local encoder and autoregressive decoder an unmasked view of the entire instance; the
predicted constraint matrix is a soft prior applied by the local encoder, not a feasibility rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = [
    "GLOBAL_NODE_FEATURE_DIM",
    "GlobalEncoder",
    "GlobalEncoderOutput",
    "build_global_node_features",
]

# [x, y, demand / capacity, is_depot].
GLOBAL_NODE_FEATURE_DIM = 4


@dataclass(frozen=True)
class GlobalEncoderOutput:
    """Embeddings produced from one padded CVRP batch.

    ``node_embeddings`` has shape ``[batch, n_nodes, embedding_dim]`` and is zero at padded
    positions. ``graph_embedding`` has shape ``[batch, embedding_dim]`` and is the masked mean of
    all real node embeddings, including the depot.
    """

    node_embeddings: Tensor
    graph_embedding: Tensor


def _validate_inputs(
    coords: Tensor,
    demands: Tensor,
    capacity: Tensor,
    depot_index: Tensor,
    node_mask: Tensor,
) -> None:
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape [batch, n_nodes, 2], got {tuple(coords.shape)}")
    batch_size, n_nodes, _ = coords.shape
    if demands.shape != (batch_size, n_nodes):
        raise ValueError(
            f"demands must have shape {(batch_size, n_nodes)}, got {tuple(demands.shape)}"
        )
    if capacity.shape != (batch_size,):
        raise ValueError(f"capacity must have shape {(batch_size,)}, got {tuple(capacity.shape)}")
    if depot_index.shape != (batch_size,):
        raise ValueError(
            f"depot_index must have shape {(batch_size,)}, got {tuple(depot_index.shape)}"
        )
    if node_mask.shape != (batch_size, n_nodes):
        raise ValueError(
            f"node_mask must have shape {(batch_size, n_nodes)}, got {tuple(node_mask.shape)}"
        )
    if n_nodes == 0:
        raise ValueError("each CVRP graph must contain at least the depot")
    if not torch.isfinite(coords).all():
        raise ValueError("coords must contain only finite values")
    if not torch.isfinite(demands).all():
        raise ValueError("demands must contain only finite values")
    if not torch.isfinite(capacity).all() or torch.any(capacity <= 0):
        raise ValueError("capacity must be positive and finite")
    if depot_index.is_floating_point() or depot_index.dtype == torch.bool:
        raise ValueError("depot_index must use an integer dtype")
    if torch.any(depot_index < 0) or torch.any(depot_index >= n_nodes):
        raise ValueError(f"depot_index values must be in [0, {n_nodes})")

    mask = node_mask.to(dtype=torch.bool)
    batch_indices = torch.arange(batch_size, device=depot_index.device)
    if not torch.all(mask[batch_indices, depot_index.to(dtype=torch.long)]):
        raise ValueError("the depot must be marked as a real node in node_mask")


def build_global_node_features(
    coords: Tensor,
    demands: Tensor,
    capacity: Tensor,
    depot_index: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Build full-graph node features ``[x, y, demand/capacity, is_depot]``.

    All arguments use the padded :class:`~vrp_diffusion_quantum.data.dataset.CVRPBatch` layout.
    The returned tensor has shape ``[batch, n_nodes, 4]`` and is zero at padded positions.
    """
    _validate_inputs(coords, demands, capacity, depot_index, node_mask)
    batch_size, n_nodes, _ = coords.shape
    depot = torch.zeros((batch_size, n_nodes), dtype=coords.dtype, device=coords.device)
    depot.scatter_(1, depot_index.to(device=coords.device, dtype=torch.long)[:, None], 1.0)
    demand_ratio = demands.to(dtype=coords.dtype) / capacity.to(dtype=coords.dtype)[:, None]
    features = torch.cat([coords, demand_ratio.unsqueeze(-1), depot.unsqueeze(-1)], dim=-1)
    return features * node_mask.to(device=coords.device, dtype=coords.dtype).unsqueeze(-1)


class _GlobalAttentionLayer(nn.Module):
    """Pre-norm self-attention and feed-forward block for a complete graph."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.self_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_embeddings: Tensor, node_mask: Tensor) -> Tensor:
        mask = node_mask.to(dtype=torch.bool)
        normalized = self.attention_norm(node_embeddings)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=False,
        )
        node_embeddings = node_embeddings + self.dropout(attended)
        node_embeddings = node_embeddings + self.dropout(
            self.feed_forward(self.feed_forward_norm(node_embeddings))
        )
        return node_embeddings * mask.unsqueeze(-1)


class GlobalEncoder(nn.Module):
    """Encode every depot/customer node with complete-graph self-attention.

    The caller should seed PyTorch before model construction when reproducible initialization is
    required. Runtime is quadratic in ``n_nodes``, which is appropriate for the project's target
    CVRP20/50/100 sizes.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        feed_forward_dim: int = 512,
        dropout: float = 0.0,
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

        self.embedding_dim = embedding_dim
        self.input_projection = nn.Linear(GLOBAL_NODE_FEATURE_DIM, embedding_dim)
        self.layers = nn.ModuleList(
            _GlobalAttentionLayer(embedding_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        coords: Tensor,
        demands: Tensor,
        capacity: Tensor,
        depot_index: Tensor,
        node_mask: Tensor,
    ) -> GlobalEncoderOutput:
        """Return full-node and masked pooled graph embeddings for a padded CVRP batch."""
        features = build_global_node_features(
            coords,
            demands,
            capacity,
            depot_index,
            node_mask,
        )
        mask = node_mask.to(device=coords.device, dtype=torch.bool)
        node_embeddings: Tensor = self.input_projection(features) * mask.unsqueeze(-1)
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, mask)
        node_embeddings = self.output_norm(node_embeddings) * mask.unsqueeze(-1)

        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(node_embeddings.dtype)
        graph_embedding = node_embeddings.sum(dim=1) / valid_counts
        return GlobalEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=graph_embedding,
        )
