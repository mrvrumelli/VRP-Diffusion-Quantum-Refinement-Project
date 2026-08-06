"""Tests for the discrete Bernoulli diffusion noise schedule (task P3.1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import collate_batch, make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.diffusion import (
    BETA_END,
    BETA_START,
    BernoulliDiffusionSchedule,
    linear_beta_schedule,
)


def _random_constraint_matrix(n: int, batch: int, *, density: float, seed: int) -> torch.Tensor:
    """Return a symmetric, zero-diagonal binary matrix batch of shape ``[batch, n, n]``."""
    generator = torch.Generator().manual_seed(seed)
    upper = (torch.rand(batch, n, n, generator=generator) < density).float()
    upper = torch.triu(upper, diagonal=1)
    return upper + upper.transpose(-1, -2)


def _is_valid_matrix(m: torch.Tensor) -> None:
    assert torch.all((m == 0) | (m == 1)), "values must be binary"
    assert torch.equal(m, m.transpose(-1, -2)), "matrix must be symmetric"
    diag = torch.diagonal(m, dim1=-2, dim2=-1)
    assert torch.all(diag == 0), "diagonal must be zero"


def _example(n_customers: int, *, seed: int) -> CVRPExample:
    """A minimal labeled CVRP example: all customers on one route (fully connected M)."""
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.random((n_customers, 2))])
    demands = np.concatenate([[0.0], np.ones(n_customers)])
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=float(n_customers + 1),
        depot_index=0,
        instance_id=f"cvrp{n_customers}_{seed}",
        n_customers=n_customers,
        seed=seed,
        generator_settings={},
    )
    solution = LabeledSolution(
        routes=[list(range(n_customers))],
        cost=1.0,
        num_vehicles=1,
        feasible=True,
        solver_name="unit",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_linear_beta_schedule_endpoints() -> None:
    betas = linear_beta_schedule(1000)
    assert betas.shape == (1000,)
    assert betas[0].item() == pytest.approx(BETA_START)
    assert betas[-1].item() == pytest.approx(BETA_END)
    assert torch.all(betas[1:] >= betas[:-1]), "linear schedule is non-decreasing"


def test_beta_schedule_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        linear_beta_schedule(0)
    with pytest.raises(ValueError):
        linear_beta_schedule(10, beta_start=0.6, beta_end=0.7)  # 1 - 2*beta would be negative
    with pytest.raises(ValueError):
        linear_beta_schedule(10, beta_start=0.1, beta_end=0.01)  # start > end


def test_cumulative_flip_probability_grows_to_half() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    flip = schedule.q_bar_flip
    assert flip.shape == (1000,)
    assert torch.all(flip[1:] >= flip[:-1] - 1e-6), "cumulative flip prob is non-decreasing"
    assert flip[0].item() < 1e-3, "almost no corruption at the first step"
    # Terminal step reaches the maximum-entropy state Q_bar_T = [[.5, .5], [.5, .5]].
    assert flip[-1].item() == pytest.approx(0.5, abs=1e-3)


def test_marginal_prob_matches_closed_form() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=100)
    m = _random_constraint_matrix(8, batch=2, density=0.3, seed=0)
    t = 40
    prob = schedule.marginal_prob(m, t)
    flip = schedule.q_bar_flip[t]
    expected = m * (1.0 - flip) + (1.0 - m) * flip
    assert torch.allclose(prob, expected)


def test_q_sample_preserves_structure_and_is_near_identity_at_t0() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    m = _random_constraint_matrix(50, batch=4, density=0.15, seed=1)
    generator = torch.Generator().manual_seed(123)

    m_t = schedule.q_sample(m, 0, generator=generator)
    _is_valid_matrix(m_t)
    # beta_1 = 1e-4, so almost nothing flips at the first step.
    assert (m_t != m).float().mean().item() < 0.02


def test_q_sample_reaches_max_entropy_at_final_step() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    # A very sparse input still becomes ~50% dense at the terminal step.
    m = _random_constraint_matrix(60, batch=8, density=0.05, seed=2)
    generator = torch.Generator().manual_seed(7)

    m_t = schedule.q_sample(m, schedule.num_timesteps - 1, generator=generator)
    _is_valid_matrix(m_t)
    off_diag = torch.triu(torch.ones_like(m_t), diagonal=1).bool()
    density = m_t[off_diag].mean().item()
    assert 0.4 < density < 0.6, f"expected near-uniform density, got {density}"


def test_q_sample_is_reproducible_with_seeded_generator() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    m = _random_constraint_matrix(30, batch=3, density=0.2, seed=3)
    t = torch.tensor([100, 500, 900])

    same_a = schedule.q_sample(m, t, generator=torch.Generator().manual_seed(42))
    same_b = schedule.q_sample(m, t, generator=torch.Generator().manual_seed(42))
    different = schedule.q_sample(m, t, generator=torch.Generator().manual_seed(43))

    assert torch.equal(same_a, same_b), "same seed must give identical noising"
    assert not torch.equal(same_a, different), "different seed should differ"


def test_per_sample_timesteps_broadcast() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    m = _random_constraint_matrix(20, batch=5, density=0.2, seed=4)
    t = schedule.sample_timesteps(5, generator=torch.Generator().manual_seed(9))
    assert t.shape == (5,)

    m_t = schedule.q_sample(m, t, generator=torch.Generator().manual_seed(9))
    assert m_t.shape == m.shape
    _is_valid_matrix(m_t)


def test_customer_mask_zeros_padded_customers() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    batch, n, valid = 2, 50, 20
    m = _random_constraint_matrix(n, batch=batch, density=0.2, seed=5)
    customer_mask = torch.zeros(batch, n, dtype=torch.bool)
    customer_mask[:, :valid] = True

    m_t = schedule.q_sample(
        m,
        schedule.num_timesteps - 1,
        customer_mask=customer_mask,
        generator=torch.Generator().manual_seed(1),
    )
    _is_valid_matrix(m_t)
    assert torch.all(m_t[:, valid:, :] == 0), "padded rows must stay zero"
    assert torch.all(m_t[:, :, valid:] == 0), "padded columns must stay zero"


@pytest.mark.parametrize("n", [20, 50])
def test_forward_pass_on_cvrp_batches(n: int) -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    batch = 16
    m = _random_constraint_matrix(n, batch=batch, density=0.2, seed=n)
    t = schedule.sample_timesteps(batch, generator=torch.Generator().manual_seed(n))

    m_t = schedule.q_sample(m, t, generator=torch.Generator().manual_seed(n))
    assert m_t.shape == (batch, n, n)
    _is_valid_matrix(m_t)


def test_forward_noising_on_collated_mixed_size_batch() -> None:
    """Noising works on a real CVRP20/CVRP50 batch and respects the collate padding mask."""
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    examples = [_example(20, seed=0), _example(50, seed=1)]
    batch = collate_batch(examples)
    assert batch.constraint_matrix.shape == (2, 50, 50)

    m_t = schedule.q_sample(
        batch.constraint_matrix,
        schedule.num_timesteps - 1,
        customer_mask=batch.customer_mask,
        generator=torch.Generator().manual_seed(11),
    )
    _is_valid_matrix(m_t)
    # The CVRP20 example (index 0) must have its 20..50 padding region left at zero.
    padded = ~batch.customer_mask[0]
    assert torch.all(m_t[0, padded, :] == 0)
    assert torch.all(m_t[0, :, padded] == 0)


def test_posterior_collapses_to_x0_at_t0() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    m = _random_constraint_matrix(16, batch=2, density=0.25, seed=6)
    m_t = schedule.q_sample(m, 0, generator=torch.Generator().manual_seed(0))
    posterior = schedule.q_posterior_prob(m_t, m, 0)
    assert torch.allclose(posterior, m, atol=1e-6)


def test_posterior_probabilities_are_valid() -> None:
    schedule = BernoulliDiffusionSchedule(num_timesteps=1000)
    m = _random_constraint_matrix(24, batch=4, density=0.2, seed=8)
    t = torch.tensor([10, 200, 600, 999])
    m_t = schedule.q_sample(m, t, generator=torch.Generator().manual_seed(2))
    posterior = schedule.q_posterior_prob(m_t, m, t)
    assert torch.all((posterior >= 0.0) & (posterior <= 1.0))
