"""Anisotropic graph denoiser for constraint matrix ``M`` (CMD section IV-B2, eqs. 10-15).

``h^0 = GAT(G)`` (optional pretrained) + noisy ``m_t`` → logits for clean ``M``.
Uses LayerNorm; pass ``customer_mask`` to ignore padded customers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from vrp_diffusion_quantum.models.gat_encoder import (
    NODE_FEATURE_DIM,
    NodeGATEncoder,
    build_customer_node_features,
    load_gat_encoder_checkpoint,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ConstraintDenoiser",
    "sinusoidal_timestep_embedding",
]

NodeEncoderType = Literal["linear", "gat"]

# Edge feature layout: [m_t, pairwise_distance].
_EDGE_FEATURE_DIM = 2


def sinusoidal_timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Sinusoidal timestep embedding (CMD eqs. 14-15)."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    half = dim // 2
    device = timesteps.device
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    exponents = torch.arange(half, device=device, dtype=torch.float32) / half
    frequencies = torch.exp(-math.log(10000.0) * exponents)
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, embedding.new_zeros(embedding.shape[0], 1)], dim=-1)
    return embedding


class _AnisotropicLayer(nn.Module):
    """One anisotropic gated MP layer (CMD eqs. 10-13)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.source_node = nn.Linear(hidden_dim, hidden_dim)
        self.target_node = nn.Linear(hidden_dim, hidden_dim)
        self.edge = nn.Linear(hidden_dim, hidden_dim)
        self.gate_value = nn.Linear(hidden_dim, hidden_dim)
        self.node_self = nn.Linear(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embed: Tensor,  # [batch, n, hidden]
        edge_embed: Tensor,  # [batch, n, n, hidden]
        time_embed: Tensor,  # [batch, hidden]
        pair_mask: Tensor,  # [batch, n, n] float in {0, 1}
        node_mask: Tensor,  # [batch, n] float in {0, 1}
    ) -> tuple[Tensor, Tensor]:
        # eq. 11: refine the edge feature and inject the timestep.
        edge_bar = self.edge_mlp(self.edge_norm(edge_embed))
        edge_bar = edge_bar + self.time_proj(time_embed)[:, None, None, :]

        # eq. 10: new edge representation from source/target nodes and the refined edge feature.
        source = self.source_node(node_embed)[:, :, None, :]
        target = self.target_node(node_embed)[:, None, :, :]
        edge_new = source + target + self.edge(edge_bar)

        # eq. 12: anisotropic gate over the target node's value, masked to real customer pairs.
        gate = torch.sigmoid(edge_new) * self.gate_value(node_embed)[:, None, :, :]
        gate = gate * pair_mask[..., None]

        # eq. 13: residual node update aggregating gated messages from neighbors j.
        aggregated = gate.sum(dim=2)
        node_new = node_embed + torch.relu(self.node_norm(self.node_self(node_embed) + aggregated))

        node_new = node_new * node_mask[..., None]
        edge_new = edge_new * pair_mask[..., None]
        return node_new, edge_new


class ConstraintDenoiser(nn.Module):
    """Anisotropic graph denoiser predicting ``m_prob`` logits from a noisy ``m_t`` (task P3.2).

    Caller is responsible for seeding ``torch`` (e.g. ``torch.manual_seed(seed)``) before
    constructing the model so weight initialization is reproducible, per
    ``docs/coding_standards.md`` section 5.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        time_embed_dim: int = 64,
        *,
        node_encoder_type: NodeEncoderType = "linear",
        gat_num_layers: int = 5,
        gat_num_heads: int = 4,
        gat_dropout: float = 0.0,
        freeze_node_encoder: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if node_encoder_type not in ("linear", "gat"):
            raise ValueError(
                f"node_encoder_type must be 'linear' or 'gat', got {node_encoder_type!r}"
            )
        self.hidden_dim = hidden_dim
        self.node_encoder_type: NodeEncoderType = node_encoder_type
        self.gat_num_layers = gat_num_layers
        self.gat_num_heads = gat_num_heads
        self.freeze_node_encoder = freeze_node_encoder

        if node_encoder_type == "linear":
            self.node_encoder: nn.Module = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        else:
            self.node_encoder = NodeGATEncoder(
                in_dim=NODE_FEATURE_DIM,
                hidden_dim=hidden_dim,
                num_layers=gat_num_layers,
                num_heads=gat_num_heads,
                dropout=gat_dropout,
            )
        self.edge_encoder = nn.Linear(_EDGE_FEATURE_DIM, hidden_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embed_dim = time_embed_dim
        self.layers = nn.ModuleList(_AnisotropicLayer(hidden_dim) for _ in range(num_layers))
        self.output_head = nn.Linear(hidden_dim, 1)
        if freeze_node_encoder:
            self.set_node_encoder_trainable(False)

    def set_node_encoder_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze ``h^0`` encoder (pretrained GAT stays fixed while denoising)."""
        for param in self.node_encoder.parameters():
            param.requires_grad = trainable
        self.freeze_node_encoder = not trainable

    def load_gat_pretrained(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        """Load a :class:`NodeGATEncoder` checkpoint into this denoiser (requires ``gat`` type)."""
        if self.node_encoder_type != "gat":
            raise ValueError("load_gat_pretrained requires node_encoder_type='gat'")
        assert isinstance(self.node_encoder, NodeGATEncoder)
        return load_gat_encoder_checkpoint(path, self.node_encoder, strict=strict)

    def forward(
        self,
        customer_coords: Tensor,  # [batch, n, 2]
        customer_demands: Tensor,  # [batch, n]
        capacity: Tensor,  # [batch]
        m_t: Tensor,  # [batch, n, n]
        t: Tensor,  # [batch]
        *,
        customer_mask: Tensor | None = None,  # [batch, n] bool
    ) -> Tensor:
        """Predict per-pair logits for ``m_prob``, shape ``[batch, n, n]``, symmetric.

        The returned logits are symmetric (``L_ij == L_ji``); apply :meth:`predict_proba` for the
        zero-diagonal probability matrix, or a with-logits loss over off-diagonal pairs (P3.3).
        """
        batch_size, n_customers, _ = customer_coords.shape
        if customer_mask is None:
            node_mask = customer_coords.new_ones(batch_size, n_customers)
        else:
            node_mask = customer_mask.to(dtype=customer_coords.dtype)
        pair_mask = node_mask[:, :, None] * node_mask[:, None, :]

        node_embed = self._encode_nodes(customer_coords, customer_demands, capacity, node_mask)
        edge_embed = self._encode_edges(customer_coords, m_t, pair_mask)
        time_embed = self.time_encoder(sinusoidal_timestep_embedding(t, self.time_embed_dim))

        for layer in self.layers:
            node_embed, edge_embed = layer(node_embed, edge_embed, time_embed, pair_mask, node_mask)

        edge_logits: Tensor = self.output_head(edge_embed)
        logits = edge_logits.squeeze(-1)
        symmetric_logits = 0.5 * (logits + logits.transpose(-1, -2))
        return symmetric_logits * pair_mask

    def predict_proba(
        self,
        customer_coords: Tensor,
        customer_demands: Tensor,
        capacity: Tensor,
        m_t: Tensor,
        t: Tensor,
        *,
        customer_mask: Tensor | None = None,
    ) -> Tensor:
        """Return ``m_prob`` in ``[0, 1]``, symmetric with a zero diagonal (``M_ii = 0``).

        Padded customer pairs (per ``customer_mask``) are set to zero as well.
        """
        logits = self.forward(
            customer_coords, customer_demands, capacity, m_t, t, customer_mask=customer_mask
        )
        n_customers = logits.shape[-1]
        off_diagonal = 1.0 - torch.eye(n_customers, device=logits.device, dtype=logits.dtype)
        m_prob = torch.sigmoid(logits) * off_diagonal
        if customer_mask is not None:
            mask = customer_mask.to(dtype=m_prob.dtype)
            m_prob = m_prob * mask[:, :, None] * mask[:, None, :]
        return m_prob

    def _encode_nodes(
        self, customer_coords: Tensor, customer_demands: Tensor, capacity: Tensor, node_mask: Tensor
    ) -> Tensor:
        node_features = build_customer_node_features(customer_coords, customer_demands, capacity)
        encoded: Tensor
        if self.node_encoder_type == "gat":
            assert isinstance(self.node_encoder, NodeGATEncoder)
            encoded = self.node_encoder(node_features, node_mask, customer_coords=customer_coords)
        else:
            encoded = self.node_encoder(node_features)
        return encoded * node_mask[..., None]

    def _encode_edges(self, customer_coords: Tensor, m_t: Tensor, pair_mask: Tensor) -> Tensor:
        coords_i = customer_coords[:, :, None, :]
        coords_j = customer_coords[:, None, :, :]
        distance = torch.linalg.norm(coords_i - coords_j, dim=-1)
        edge_features = torch.stack([m_t, distance], dim=-1)
        encoded: Tensor = self.edge_encoder(edge_features)
        return encoded * pair_mask[..., None]
