"""Faithful GAT encoder and sum-MLP fusion path for the CMD paper baseline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from vrp_diffusion_quantum.models.fusion_encoder import FusionEncoderOutput
from vrp_diffusion_quantum.models.gat_encoder import (
    NODE_FEATURE_DIM,
    NodeGATEncoder,
    build_customer_node_features,
    compat_layernorm_state_dict,
)
from vrp_diffusion_quantum.models.global_encoder import GlobalEncoderOutput, _validate_inputs
from vrp_diffusion_quantum.models.local_masked_encoder import (
    LocalAttentionPrior,
    LocalMaskedEncoderOutput,
)

__all__ = [
    "PaperGlobalGATEncoder",
    "PaperMaskedGATEncoder",
    "PaperSumMLPFusion",
    "load_diffusion_gat_checkpoint",
]


def _extract_gat_state(payload: object) -> Mapping[str, Tensor]:
    """Extract the pretrained node GAT from either GAT-only or denoiser checkpoints."""
    if not isinstance(payload, Mapping):
        raise ValueError("GAT checkpoint must contain a mapping")
    encoder = payload.get("encoder")
    if isinstance(encoder, Mapping):
        return encoder

    model = payload.get("model")
    if isinstance(model, Mapping):
        for prefix in ("node_encoder.", "module.node_encoder."):
            extracted = {
                str(key)[len(prefix) :]: value
                for key, value in model.items()
                if str(key).startswith(prefix) and isinstance(value, Tensor)
            }
            if extracted:
                return extracted

    if payload and all(isinstance(value, Tensor) for value in payload.values()):
        return payload
    raise ValueError(
        "checkpoint does not contain an 'encoder' state or a denoiser 'model.node_encoder' state"
    )


def load_diffusion_gat_checkpoint(
    path: str | Path,
    encoder: NodeGATEncoder,
) -> dict[str, Any]:
    """Strictly load the exact diffusion GAT into ``encoder``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = compat_layernorm_state_dict(_extract_gat_state(payload))
    encoder.load_state_dict(state, strict=True)
    return dict(payload) if isinstance(payload, Mapping) else {}


def gat_checkpoint_input_mode(payload: Mapping[str, Any]) -> tuple[str, bool]:
    """Return ``(coordinate_frame, depot_node)`` for a GAT-only or denoiser checkpoint."""
    model_cfg = (payload.get("extra") or {}).get("model") or {}
    frame = str(model_cfg.get("coordinate_frame", "absolute"))
    depot_node = bool(model_cfg.get("depot_node", model_cfg.get("gat_depot_node", False)))
    return frame, depot_node


def _full_node_features(coords: Tensor, demands: Tensor, capacity: Tensor) -> Tensor:
    """Use the diffusion GAT's exact feature order for full depot/customer tensors."""
    return build_customer_node_features(coords, demands.to(coords.dtype), capacity.to(coords.dtype))


def _masked_mean(node_embeddings: Tensor, node_mask: Tensor) -> Tensor:
    mask = node_mask.to(dtype=node_embeddings.dtype)
    return node_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


class PaperGlobalGATEncoder(nn.Module):
    """Frozen copy of the diffusion model's pretrained node GAT.

    Two weight families are accepted: legacy customer-only GATs in absolute coordinates, and
    depot-node GATs in the depot-relative frame (the paper's GAT over the full instance graph).
    For the latter the real depot node is marked and all coordinates are shifted so the depot is
    at the origin, which reproduces exactly the input the diffusion denoiser gave the same GAT.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        checkpoint: str | Path | None,
        depot_node: bool | None = None,
    ) -> None:
        super().__init__()
        payload: dict[str, Any] = {}
        resolved_depot_node = bool(depot_node)
        if checkpoint is not None:
            raw = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
            payload = dict(raw) if isinstance(raw, Mapping) else {}
            frame, checkpoint_depot_node = gat_checkpoint_input_mode(payload)
            if (frame, checkpoint_depot_node) not in {
                ("absolute", False),
                ("depot_relative", True),
            }:
                raise ValueError(
                    "paper_cmd global GAT requires absolute customer-only weights or "
                    f"depot-relative depot-node weights, got frame={frame!r} "
                    f"depot_node={checkpoint_depot_node}"
                )
            if depot_node is not None and bool(depot_node) != checkpoint_depot_node:
                raise ValueError("global_gat_depot_node does not match the GAT checkpoint")
            resolved_depot_node = checkpoint_depot_node
        self.gat = NodeGATEncoder(
            in_dim=NODE_FEATURE_DIM,
            hidden_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            depot_node=resolved_depot_node,
        )
        self.checkpoint_source: str | None = None
        self.checkpoint_payload: dict[str, Any] = payload
        if checkpoint is not None:
            state = compat_layernorm_state_dict(_extract_gat_state(payload))
            self.gat.load_state_dict(state, strict=True)
            self.checkpoint_source = str(Path(checkpoint))
        self.freeze()

    @property
    def depot_node(self) -> bool:
        return self.gat.depot_node

    def freeze(self) -> None:
        """Keep the pretrained encoder immutable and deterministic during policy training."""
        self.gat.requires_grad_(False)
        self.gat.eval()

    def train(self, mode: bool = True) -> PaperGlobalGATEncoder:
        super().train(mode)
        self.gat.eval()
        return self

    def verify_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.gat.parameters()):
            raise RuntimeError("the paper_cmd global diffusion GAT must remain frozen")
        if any(parameter.grad is not None for parameter in self.gat.parameters()):
            raise RuntimeError("the frozen paper_cmd global diffusion GAT received gradients")

    def forward(
        self,
        coords: Tensor,
        demands: Tensor,
        capacity: Tensor,
        depot_index: Tensor,
        node_mask: Tensor,
    ) -> GlobalEncoderOutput:
        _validate_inputs(coords, demands, capacity, depot_index, node_mask)
        mask = node_mask.to(device=coords.device, dtype=torch.bool)
        if self.gat.depot_node:
            rows = torch.arange(coords.shape[0], device=coords.device)
            depot_index = depot_index.to(device=coords.device, dtype=torch.long)
            relative = (coords - coords[rows, depot_index].unsqueeze(1)) * mask.unsqueeze(-1)
            indicator = torch.zeros(mask.shape, dtype=coords.dtype, device=coords.device)
            indicator[rows, depot_index] = 1.0
            node_embeddings = self.gat(
                _full_node_features(relative, demands, capacity),
                mask,
                customer_coords=relative,
                depot_indicator=indicator,
            )
        else:
            features = _full_node_features(coords, demands, capacity)
            node_embeddings = self.gat(features, mask, customer_coords=coords)
        node_embeddings = node_embeddings * mask.unsqueeze(-1)
        return GlobalEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=_masked_mean(node_embeddings, mask),
        )


class PaperMaskedGATEncoder(nn.Module):
    """Trainable local GAT whose message passing is masked by predicted ``M``."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        hard_threshold: float,
    ) -> None:
        super().__init__()
        self.hard_threshold = hard_threshold
        self.gat = NodeGATEncoder(
            in_dim=NODE_FEATURE_DIM,
            hidden_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        coords: Tensor,
        demands: Tensor,
        capacity: Tensor,
        node_mask: Tensor,
        prior: LocalAttentionPrior,
    ) -> LocalMaskedEncoderOutput:
        mask = node_mask.to(device=coords.device, dtype=torch.bool)
        adjacency = prior.allowed_pairs & (prior.weights >= self.hard_threshold)
        node_embeddings = self.gat(
            _full_node_features(coords, demands, capacity),
            mask,
            customer_coords=coords,
            adjacency_mask=adjacency,
        )
        node_embeddings = node_embeddings * mask.unsqueeze(-1)
        return LocalMaskedEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=_masked_mean(node_embeddings, mask),
        )


class PaperSumMLPFusion(nn.Module):
    """Paper fusion: aligned global and local embeddings are summed, then mapped by an MLP."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
        )

    def forward(
        self,
        global_node_embeddings: Tensor,
        local_node_embeddings: Tensor,
        node_mask: Tensor,
    ) -> FusionEncoderOutput:
        if global_node_embeddings.shape != local_node_embeddings.shape:
            raise ValueError("global and local embeddings must have identical shapes")
        if global_node_embeddings.ndim != 3:
            raise ValueError("global and local embeddings must have shape [batch, nodes, dim]")
        if global_node_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected embedding_dim {self.embedding_dim}, "
                f"got {global_node_embeddings.shape[-1]}"
            )
        if node_mask.shape != global_node_embeddings.shape[:2]:
            raise ValueError("node_mask shape must match the first two embedding dimensions")
        mask = node_mask.to(dtype=global_node_embeddings.dtype).unsqueeze(-1)
        node_embeddings = self.mlp(global_node_embeddings + local_node_embeddings) * mask
        return FusionEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embedding=_masked_mean(node_embeddings, node_mask),
            fusion_gate=None,
        )
