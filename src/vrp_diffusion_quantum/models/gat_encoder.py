"""GAT node encoder for CMD (``h^0 = GAT(G)``) + pair head for M-pretrain."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

__all__ = [
    "NODE_FEATURE_DIM",
    "GATConstraintPretrainer",
    "NodeGATEncoder",
    "build_customer_node_features",
    "compat_layernorm_state_dict",
    "load_gat_encoder_checkpoint",
    "save_gat_encoder_checkpoint",
]

_NORM_MODULE_NAMES = frozenset({"out_norm", "edge_norm", "node_norm"})


def compat_layernorm_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Map legacy ChannelNorm keys ``*.out_norm.norm.weight`` → ``*.out_norm.weight``."""
    remapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if (
            len(parts) >= 3
            and parts[-2] == "norm"
            and parts[-3] in _NORM_MODULE_NAMES
            and parts[-1] in ("weight", "bias")
        ):
            remapped[".".join([*parts[:-2], parts[-1]])] = value
        else:
            remapped[key] = value
    return remapped


# Shared with ConstraintDenoiser: [x, y, demand, demand / capacity].
NODE_FEATURE_DIM = 4


def build_customer_node_features(
    customer_coords: Tensor,
    customer_demands: Tensor,
    capacity: Tensor,
) -> Tensor:
    """Stack per-customer features ``[x, y, demand, demand/capacity]``, shape ``[B, N, 4]``."""
    demand = customer_demands.unsqueeze(-1)
    demand_ratio = demand / capacity[:, None, None].clamp_min(1e-8)
    return torch.cat([customer_coords, demand, demand_ratio], dim=-1)


class _GATLayer(nn.Module):
    """One multi-head graph-attention layer on a dense (fully connected) graph."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_distance_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_distance_bias = use_distance_bias

        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))
        self.distance_bias = nn.Linear(1, num_heads, bias=False) if use_distance_bias else None
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_embed: Tensor,  # [B, N, in]
        node_mask: Tensor,  # [B, N] float {0,1}
        pairwise_distance: Tensor | None = None,  # [B, N, N]
    ) -> Tensor:
        batch, n_nodes, _ = node_embed.shape
        h = self.proj(node_embed).view(batch, n_nodes, self.num_heads, self.head_dim)
        # Attention logits: e_ij = LeakyReLU(a_src·h_i + a_dst·h_j [+ dist bias])
        score_src = (h * self.attn_src).sum(dim=-1)  # [B, N, H]
        score_dst = (h * self.attn_dst).sum(dim=-1)
        logits = score_src.unsqueeze(2) + score_dst.unsqueeze(1)  # [B, N, N, H]
        if self.distance_bias is not None and pairwise_distance is not None:
            logits = logits + self.distance_bias(pairwise_distance.unsqueeze(-1))
        logits = self.leaky_relu(logits)

        # Mask padded keys/queries (self-loops remain; padding pairs are zeroed).
        key_ok = node_mask[:, None, :, None]  # [B, 1, N, 1]
        query_ok = node_mask[:, :, None, None]
        logits = logits.masked_fill(key_ok * query_ok == 0, -1e9)

        attn = torch.softmax(logits, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn) * key_ok

        # Aggregate: [B,N,N,H] x [B,N,H,D] -> [B,N,H,D]
        messages = torch.einsum("bijn,bjnd->bind", attn, h)
        out: Tensor = messages.reshape(batch, n_nodes, self.num_heads * self.head_dim)
        out = self.out_norm(out) * node_mask.unsqueeze(-1)
        return out


class NodeGATEncoder(nn.Module):
    """Five-layer (default) GAT producing node embeddings ``h^0`` for the CMD denoiser."""

    def __init__(
        self,
        *,
        in_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_distance_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                _GATLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_distance_bias=use_distance_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.residuals = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        node_features: Tensor,  # [B, N, F]
        node_mask: Tensor,  # [B, N] bool/float
        *,
        customer_coords: Tensor | None = None,  # [B, N, 2] for distance bias
    ) -> Tensor:
        mask = node_mask.to(dtype=node_features.dtype)
        h: Tensor = self.input_proj(node_features) * mask.unsqueeze(-1)
        distance: Tensor | None = None
        if customer_coords is not None:
            distance = torch.linalg.norm(
                customer_coords[:, :, None, :] - customer_coords[:, None, :, :], dim=-1
            )
        for layer, skip in zip(self.layers, self.residuals, strict=True):
            h_new = layer(h, mask, distance)
            h = (h_new + skip(h)) * mask.unsqueeze(-1)
        return h


class GATConstraintPretrainer(nn.Module):
    """GAT encoder + symmetric pair head for supervised pretraining on ``m_true``."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        gat_num_layers: int = 5,
        gat_num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = NodeGATEncoder(
            hidden_dim=hidden_dim,
            num_layers=gat_num_layers,
            num_heads=gat_num_heads,
            dropout=dropout,
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        customer_coords: Tensor,
        customer_demands: Tensor,
        capacity: Tensor,
        customer_mask: Tensor | None = None,
    ) -> Tensor:
        batch, n, _ = customer_coords.shape
        if customer_mask is None:
            node_mask = customer_coords.new_ones(batch, n)
        else:
            node_mask = customer_mask.to(dtype=customer_coords.dtype)
        feats = build_customer_node_features(customer_coords, customer_demands, capacity)
        encoded: Tensor = self.encoder(feats, node_mask, customer_coords=customer_coords)
        return encoded

    def forward(
        self,
        customer_coords: Tensor,
        customer_demands: Tensor,
        capacity: Tensor,
        *,
        customer_mask: Tensor | None = None,
    ) -> Tensor:
        """Return symmetric pair logits ``[B, N, N]`` (diagonal left unconstrained)."""
        h = self.encode(customer_coords, customer_demands, capacity, customer_mask)
        batch, n, _ = h.shape
        hi = h[:, :, None, :].expand(batch, n, n, h.shape[-1])
        hj = h[:, None, :, :].expand(batch, n, n, h.shape[-1])
        dist = torch.linalg.norm(
            customer_coords[:, :, None, :] - customer_coords[:, None, :, :], dim=-1, keepdim=True
        )
        pair = torch.cat([hi, hj, torch.abs(hi - hj), dist], dim=-1)
        logits: Tensor = self.pair_mlp(pair).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        if customer_mask is not None:
            mask = customer_mask.to(dtype=logits.dtype)
            logits = logits * mask[:, :, None] * mask[:, None, :]
        return logits


def save_gat_encoder_checkpoint(
    path: str | Path,
    encoder: NodeGATEncoder,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write encoder weights (+ optional metadata) for loading into :class:`ConstraintDenoiser`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "encoder": encoder.state_dict(),
        "hidden_dim": encoder.hidden_dim,
        "num_layers": encoder.num_layers,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, target)
    logger.info("wrote GAT encoder checkpoint %s", target)
    return target


def load_gat_encoder_checkpoint(
    path: str | Path,
    encoder: NodeGATEncoder,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint produced by :func:`save_gat_encoder_checkpoint` into ``encoder``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = payload["encoder"] if isinstance(payload, dict) and "encoder" in payload else payload
    encoder.load_state_dict(compat_layernorm_state_dict(state), strict=strict)
    return payload if isinstance(payload, dict) else {}
