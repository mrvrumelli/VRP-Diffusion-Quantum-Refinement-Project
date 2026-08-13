"""Learned gated fusion of aligned global and ``M_hat``-local CVRP embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["FusionEncoder", "FusionEncoderOutput"]


@dataclass(frozen=True)
class FusionEncoderOutput:
    """Fused node/context embeddings and the learned local-information gate.

    ``fusion_gate`` has the same shape as ``node_embeddings``. Values near zero favor the global
    representation; values near one favor the local representation. Padded positions are zero.
    """

    node_embeddings: Tensor  # [batch, n_nodes, embedding_dim]
    graph_embedding: Tensor  # [batch, embedding_dim]
    fusion_gate: Tensor  # [batch, n_nodes, embedding_dim]


class FusionEncoder(nn.Module):
    """Fuse global and local node embeddings with a learned guarded residual.

    For aligned embeddings ``h_global`` and ``h_local`` the module computes::

        delta = h_local - h_global
        gate = sigmoid(MLP([h_global, h_local, abs(delta)]))
        mixed = h_global + gate * delta

    A position-wise residual feed-forward block then refines ``mixed``. The explicit global path
    means a poor predicted route-membership matrix cannot erase full-graph information unless
    training learns to open the local gate.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        gate_hidden_dim: int | None = None,
        feed_forward_dim: int = 512,
        dropout: float = 0.0,
        initial_local_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if gate_hidden_dim is not None and gate_hidden_dim < 1:
            raise ValueError(f"gate_hidden_dim must be >= 1, got {gate_hidden_dim}")
        if feed_forward_dim < 1:
            raise ValueError(f"feed_forward_dim must be >= 1, got {feed_forward_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not 0.0 < initial_local_weight < 1.0:
            raise ValueError(
                f"initial_local_weight must be strictly between 0 and 1, got {initial_local_weight}"
            )

        self.embedding_dim = embedding_dim
        hidden_dim = gate_hidden_dim if gate_hidden_dim is not None else embedding_dim
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        gate_output = self.gate_network[-1]
        assert isinstance(gate_output, nn.Linear)
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(
            gate_output.bias,
            math.log(initial_local_weight / (1.0 - initial_local_weight)),
        )

        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        global_node_embeddings: Tensor,
        local_node_embeddings: Tensor,
        node_mask: Tensor,
    ) -> FusionEncoderOutput:
        """Fuse two aligned full-node tensors and return node plus graph representations."""
        if global_node_embeddings.ndim != 3:
            raise ValueError(
                "global_node_embeddings must have shape [batch, n_nodes, embedding_dim], got "
                f"{tuple(global_node_embeddings.shape)}"
            )
        if local_node_embeddings.shape != global_node_embeddings.shape:
            raise ValueError(
                "local_node_embeddings must match global_node_embeddings shape "
                f"{tuple(global_node_embeddings.shape)}, got {tuple(local_node_embeddings.shape)}"
            )
        batch_size, n_nodes, embedding_dim = global_node_embeddings.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(f"expected embedding_dim {self.embedding_dim}, got {embedding_dim}")
        if node_mask.shape != (batch_size, n_nodes):
            raise ValueError(
                f"node_mask must have shape {(batch_size, n_nodes)}, got {tuple(node_mask.shape)}"
            )
        if global_node_embeddings.device != local_node_embeddings.device:
            raise ValueError("global and local embeddings must be on the same device")
        if global_node_embeddings.device != node_mask.device:
            raise ValueError("embeddings and node_mask must be on the same device")
        if global_node_embeddings.dtype != local_node_embeddings.dtype:
            raise ValueError("global and local embeddings must use the same dtype")
        if not torch.isfinite(global_node_embeddings).all():
            raise ValueError("global_node_embeddings must contain only finite values")
        if not torch.isfinite(local_node_embeddings).all():
            raise ValueError("local_node_embeddings must contain only finite values")

        mask = node_mask.to(dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every graph must contain at least one real node")
        mask_values = mask.to(dtype=global_node_embeddings.dtype).unsqueeze(-1)
        global_embeddings = global_node_embeddings * mask_values
        local_embeddings = local_node_embeddings * mask_values

        difference = local_embeddings - global_embeddings
        gate_inputs = torch.cat(
            [global_embeddings, local_embeddings, torch.abs(difference)],
            dim=-1,
        )
        gate_logits: Tensor = self.gate_network(gate_inputs)
        fusion_gate = torch.sigmoid(gate_logits) * mask_values
        mixed_embeddings = global_embeddings + fusion_gate * difference
        feed_forward_update: Tensor = self.feed_forward(self.feed_forward_norm(mixed_embeddings))
        node_embeddings = self.output_norm(mixed_embeddings + self.dropout(feed_forward_update))
        node_embeddings = node_embeddings * mask_values

        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(node_embeddings.dtype)
        graph_embedding = node_embeddings.sum(dim=1) / valid_counts
        return FusionEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=graph_embedding,
            fusion_gate=fusion_gate,
        )
