"""Tests for the anisotropic graph constraint denoiser (task P3.2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as ff

from vrp_diffusion_quantum.data.dataset import CVRPBatch, collate_batch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.constraint_denoiser import (
    ConstraintDenoiser,
    sinusoidal_timestep_embedding,
)
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule


def _example(n_customers: int, *, num_routes: int = 3, seed: int = 0) -> CVRPExample:
    """A labeled CVRP example whose customers are split across a few routes."""
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.random((n_customers, 2))])
    demands = np.concatenate([[0.0], np.ones(n_customers)])
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=float(n_customers),
        depot_index=0,
        instance_id=f"cvrp{n_customers}_{seed}",
        n_customers=n_customers,
        seed=seed,
        generator_settings={},
    )
    customer_ids = list(range(n_customers))
    routes = [customer_ids[i::num_routes] for i in range(num_routes)]
    routes = [route for route in routes if route]
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


def _customer_tensors(batch: CVRPBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather per-customer coords/demands (depot excluded) from a collated batch."""
    idx = batch.customer_node_indices.clamp(min=0)
    coords = torch.gather(batch.coords, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    demands = torch.gather(batch.demands, 1, idx)
    mask = batch.customer_mask.unsqueeze(-1)
    return coords * mask, demands * batch.customer_mask, batch.capacity


def _noise(batch: CVRPBatch, *, t: int, seed: int) -> torch.Tensor:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    return schedule.q_sample(
        batch.constraint_matrix,
        t,
        customer_mask=batch.customer_mask,
        generator=torch.Generator().manual_seed(seed),
    )


def test_sinusoidal_timestep_embedding_shape() -> None:
    emb = sinusoidal_timestep_embedding(torch.tensor([0, 10, 999]), dim=64)
    assert emb.shape == (3, 64)
    assert torch.all(torch.isfinite(emb))
    # Odd dimensions are padded, not dropped.
    assert sinusoidal_timestep_embedding(torch.tensor([1, 2]), dim=7).shape == (2, 7)


@pytest.mark.parametrize("n", [20, 50])
def test_forward_pass_on_cvrp_batches(n: int) -> None:
    torch.manual_seed(0)
    examples = [_example(n, seed=i) for i in range(4)]
    batch = collate_batch(examples)
    coords, demands, capacity = _customer_tensors(batch)
    m_t = _noise(batch, t=500, seed=1)
    t = torch.full((len(examples),), 500, dtype=torch.long)

    model = ConstraintDenoiser()
    logits = model(coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask)

    assert logits.shape == (4, n, n)
    assert torch.all(torch.isfinite(logits))
    # The head is symmetrized, so predictions are order-invariant in (i, j).
    assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-6)


def test_predict_proba_is_valid_matrix() -> None:
    torch.manual_seed(0)
    batch = collate_batch([_example(20, seed=0), _example(20, seed=1)])
    coords, demands, capacity = _customer_tensors(batch)
    m_t = _noise(batch, t=300, seed=2)
    t = torch.tensor([300, 300])

    model = ConstraintDenoiser()
    m_prob = model.predict_proba(
        coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask
    )

    assert torch.all((m_prob >= 0.0) & (m_prob <= 1.0))
    assert torch.allclose(m_prob, m_prob.transpose(-1, -2), atol=1e-6)
    diag = torch.diagonal(m_prob, dim1=-2, dim2=-1)
    assert torch.all(diag == 0.0), "diagonal must be zero (M_ii = 0)"


def test_forward_on_mixed_size_batch_respects_padding() -> None:
    torch.manual_seed(0)
    batch = collate_batch([_example(20, seed=0), _example(50, seed=1)])
    assert batch.constraint_matrix.shape == (2, 50, 50)
    coords, demands, capacity = _customer_tensors(batch)
    m_t = _noise(batch, t=400, seed=3)
    t = torch.tensor([400, 400])

    model = ConstraintDenoiser()
    m_prob = model.predict_proba(
        coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask
    )

    # The CVRP20 example (index 0) must have its padded customers (20..50) zeroed out.
    padded = ~batch.customer_mask[0]
    assert torch.all(m_prob[0, padded, :] == 0.0)
    assert torch.all(m_prob[0, :, padded] == 0.0)


def test_padding_does_not_affect_real_customer_predictions() -> None:
    """Extra padding around a CVRP20 instance must not change its real-customer outputs."""
    torch.manual_seed(0)
    model = ConstraintDenoiser()

    example = _example(20, seed=7)
    unpadded = collate_batch([example])
    # Force padding to size 50 by collating alongside a larger throwaway example.
    padded = collate_batch([example, _example(50, seed=8)])

    def run(batch: CVRPBatch) -> torch.Tensor:
        coords, demands, capacity = _customer_tensors(batch)
        m_t = batch.constraint_matrix  # deterministic input (skip stochastic noising here)
        t = torch.zeros(coords.shape[0], dtype=torch.long)
        return model.predict_proba(
            coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask
        )

    out_unpadded = run(unpadded)[0, :20, :20]
    out_padded = run(padded)[0, :20, :20]
    assert torch.allclose(out_unpadded, out_padded, atol=1e-5)


def test_forward_is_deterministic_with_seeded_init() -> None:
    batch = collate_batch([_example(20, seed=0), _example(20, seed=1)])
    coords, demands, capacity = _customer_tensors(batch)
    m_t = _noise(batch, t=250, seed=4)
    t = torch.tensor([250, 250])

    torch.manual_seed(123)
    model_a = ConstraintDenoiser()
    torch.manual_seed(123)
    model_b = ConstraintDenoiser()

    out_a = model_a(coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask)
    out_b = model_b(coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask)
    assert torch.equal(out_a, out_b)


def test_backward_populates_gradients() -> None:
    torch.manual_seed(0)
    batch = collate_batch([_example(20, seed=0)])
    coords, demands, capacity = _customer_tensors(batch)
    m_t = _noise(batch, t=600, seed=5)
    t = torch.tensor([600])

    model = ConstraintDenoiser()
    logits = model(coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask)

    n = logits.shape[-1]
    off_diagonal = ~torch.eye(n, dtype=torch.bool)
    target = batch.constraint_matrix[0]
    loss = ff.binary_cross_entropy_with_logits(logits[0][off_diagonal], target[off_diagonal])
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected at least one populated gradient"
    assert all(torch.all(torch.isfinite(g)) for g in grads)
