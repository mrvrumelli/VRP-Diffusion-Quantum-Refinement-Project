"""Tests for the ×9 GAT node encoder and constraint pretrainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vrp_diffusion_quantum.data.dataset import collate_batch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.gat_encoder import (
    GATConstraintPretrainer,
    NodeGATEncoder,
    build_customer_node_features,
    load_gat_encoder_checkpoint,
    save_gat_encoder_checkpoint,
)
from vrp_diffusion_quantum.train.train_diffusion import customer_tensors_from_batch


def _example(n: int, *, seed: int = 0) -> CVRPExample:
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.random((n, 2))])
    demands = np.concatenate([[0.0], np.ones(n)])
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=float(n),
        depot_index=0,
        instance_id=f"cvrp{n}_{seed}",
        n_customers=n,
        seed=seed,
        generator_settings={},
    )
    mid = max(1, n // 2)
    routes = [list(range(mid)), list(range(mid, n))]
    routes = [r for r in routes if r]
    solution = LabeledSolution(
        routes=routes,
        cost=1.0,
        num_vehicles=len(routes),
        feasible=True,
        solver_name="unit",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_build_customer_node_features_shape() -> None:
    coords = torch.rand(2, 5, 2)
    demands = torch.rand(2, 5)
    capacity = torch.tensor([10.0, 12.0])
    feats = build_customer_node_features(coords, demands, capacity)
    assert feats.shape == (2, 5, 4)


def test_node_gat_encoder_respects_padding() -> None:
    torch.manual_seed(0)
    enc = NodeGATEncoder(hidden_dim=16, num_layers=5, num_heads=4)
    feats = torch.randn(1, 4, 4)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    coords = torch.rand(1, 4, 2)
    out = enc(feats, mask, customer_coords=coords)
    assert out.shape == (1, 4, 16)
    assert torch.allclose(out[0, 3], torch.zeros(16))


def test_gat_pretrainer_symmetric_logits() -> None:
    torch.manual_seed(0)
    model = GATConstraintPretrainer(hidden_dim=16, gat_num_layers=2, gat_num_heads=4)
    batch = collate_batch([_example(6, seed=0), _example(4, seed=1)])
    coords, demands, capacity = customer_tensors_from_batch(batch)
    logits = model(coords, demands, capacity, customer_mask=batch.customer_mask)
    assert logits.shape[0] == 2
    assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-5)


def test_gat_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    enc = NodeGATEncoder(hidden_dim=16, num_layers=5, num_heads=4)
    path = tmp_path / "gat.pt"
    save_gat_encoder_checkpoint(path, enc, extra={"seed": 0})
    enc2 = NodeGATEncoder(hidden_dim=16, num_layers=5, num_heads=4)
    load_gat_encoder_checkpoint(path, enc2)
    for a, b in zip(enc.parameters(), enc2.parameters(), strict=True):
        assert torch.equal(a, b)


def test_denoiser_with_gat_and_pretrained_load(tmp_path: Path) -> None:
    torch.manual_seed(0)
    enc = NodeGATEncoder(hidden_dim=16, num_layers=5, num_heads=4)
    path = tmp_path / "gat.pt"
    save_gat_encoder_checkpoint(path, enc)
    model = ConstraintDenoiser(
        hidden_dim=16,
        num_layers=1,
        time_embed_dim=16,
        node_encoder_type="gat",
        gat_num_layers=5,
        gat_num_heads=4,
        freeze_node_encoder=True,
    )
    model.load_gat_pretrained(path)
    assert not any(p.requires_grad for p in model.node_encoder.parameters())
    batch = collate_batch([_example(5, seed=0)])
    coords, demands, capacity = customer_tensors_from_batch(batch)
    t = torch.zeros(1, dtype=torch.long)
    m_t = batch.constraint_matrix
    logits = model(coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask)
    assert logits.shape == (1, 5, 5)
    assert torch.isfinite(logits).all()
