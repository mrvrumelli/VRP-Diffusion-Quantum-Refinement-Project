"""Tests for CMD reverse sampling / matrix prediction (task P3.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.inference.predict_matrix import (
    evaluate_full_chain_sampling,
    example_to_model_inputs,
    load_denoiser_checkpoint,
    predict_matrix_batch,
    predict_matrix_one_shot,
    predict_matrix_persize,
    sample_constraint_matrix,
    select_examples_by_size,
    symmetrize_zero_diagonal,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.train.train_diffusion import save_denoiser_checkpoint


def _example(n_customers: int, *, seed: int) -> CVRPExample:
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
    mid = max(1, n_customers // 2)
    routes = [list(range(mid)), list(range(mid, n_customers))]
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


def _assert_valid_hard(m: np.ndarray, n: int) -> None:
    assert m.shape == (n, n)
    assert np.all((m == 0) | (m == 1))
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)


def _assert_valid_prob(m: np.ndarray, n: int) -> None:
    assert m.shape == (n, n)
    assert np.all(m >= 0.0) and np.all(m <= 1.0)
    assert np.allclose(m, m.T, atol=1e-5)
    assert np.allclose(np.diag(m), 0.0, atol=1e-5)


def test_symmetrize_zero_diagonal() -> None:
    raw = torch.tensor([[[1.0, 0.7, 0.2], [0.1, 1.0, 0.9], [0.3, 0.4, 1.0]]])
    out = symmetrize_zero_diagonal(raw)
    assert torch.equal(out, out.transpose(-1, -2))
    assert torch.all(torch.diagonal(out, dim1=-2, dim2=-1) == 0)


def test_sample_constraint_matrix_valid_shape() -> None:
    torch.manual_seed(0)
    n = 5
    example = _example(n, seed=0)
    coords, demands, capacity, _m_true, mask = example_to_model_inputs(example)
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=8)
    result = sample_constraint_matrix(
        model,
        schedule,
        coords=coords,
        demands=demands,
        capacity=capacity,
        customer_mask=mask,
        generator=torch.Generator().manual_seed(0),
        snapshot_every=2,
    )
    _assert_valid_hard(result.m_hat, n)
    _assert_valid_prob(result.m_prob, n)
    assert result.trajectory is not None
    assert result.trajectory_timesteps is not None
    assert len(result.trajectory) == len(result.trajectory_timesteps)
    for snap in result.trajectory:
        _assert_valid_hard(snap, n)


def test_deterministic_sampler_is_seed_independent() -> None:
    torch.manual_seed(2)
    example = _example(5, seed=2)
    coords, demands, capacity, _m_true, mask = example_to_model_inputs(example)
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=5)
    kwargs = {
        "coords": coords,
        "demands": demands,
        "capacity": capacity,
        "customer_mask": mask,
        "transition_mode": "deterministic",
        "prior_positive_probability": 0.0,
    }
    first = sample_constraint_matrix(
        model, schedule, generator=torch.Generator().manual_seed(1), **kwargs
    )
    second = sample_constraint_matrix(
        model, schedule, generator=torch.Generator().manual_seed(99), **kwargs
    )
    assert np.array_equal(first.m_hat, second.m_hat)
    assert np.allclose(first.m_prob, second.m_prob)


def test_sampler_rejects_invalid_prior_and_mode() -> None:
    example = _example(4, seed=4)
    coords, demands, capacity, _m_true, mask = example_to_model_inputs(example)
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=3)
    kwargs = {"coords": coords, "demands": demands, "capacity": capacity, "customer_mask": mask}
    with pytest.raises(ValueError, match="positive_probability"):
        sample_constraint_matrix(model, schedule, prior_positive_probability=1.1, **kwargs)
    with pytest.raises(ValueError, match="transition_mode"):
        sample_constraint_matrix(model, schedule, transition_mode="invalid", **kwargs)  # type: ignore[arg-type]


def test_predict_matrix_one_shot_valid() -> None:
    torch.manual_seed(1)
    n = 6
    example = _example(n, seed=1)
    coords, demands, capacity, _m_true, mask = example_to_model_inputs(example)
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=16)
    result = predict_matrix_one_shot(
        model,
        schedule,
        coords=coords,
        demands=demands,
        capacity=capacity,
        customer_mask=mask,
        generator=torch.Generator().manual_seed(1),
    )
    _assert_valid_hard(result.m_hat, n)
    _assert_valid_prob(result.m_prob, n)


def test_load_denoiser_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "best.pt"
    save_denoiser_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        row={"val_loss": 0.2},
        best_metric_name="val_loss",
        best_metric_value=0.2,
        extra={"model": {"hidden_dim": 16, "num_layers": 1, "time_embed_dim": 16}},
    )
    loaded, payload = load_denoiser_checkpoint(path)
    assert payload["epoch"] == 3
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters(), strict=True):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_select_examples_by_size() -> None:
    pool = [_example(4, seed=i) for i in range(3)] + [_example(6, seed=10 + i) for i in range(5)]
    selected = select_examples_by_size(pool, sizes=(4, 6), per_size=2, seed=0)
    assert len(selected) == 4
    assert sum(e.instance.n_customers == 4 for e in selected) == 2
    assert sum(e.instance.n_customers == 6 for e in selected) == 2


def test_predict_matrix_batch_both_modes() -> None:
    torch.manual_seed(0)
    examples = [_example(4, seed=0), _example(5, seed=1)]
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    schedule = BernoulliDiffusionSchedule(num_timesteps=4)
    for mode in ("one_shot", "full_chain"):
        probs, hards = predict_matrix_batch(
            model, schedule, examples, mode=mode, device="cpu", seed=0
        )
        assert len(probs) == len(examples) == len(hards)
        for example, prob, hard in zip(examples, probs, hards, strict=True):
            n = example.instance.n_customers
            _assert_valid_prob(prob, n)
            _assert_valid_hard(hard, n)


def test_predict_matrix_batch_rejects_unknown_mode() -> None:
    examples = [_example(4, seed=0)]
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    schedule = BernoulliDiffusionSchedule(num_timesteps=4)
    with pytest.raises(ValueError, match="unknown mode"):
        predict_matrix_batch(model, schedule, examples, mode="bogus", device="cpu", seed=0)  # type: ignore[arg-type]


def test_predict_matrix_persize_preserves_order_across_mixed_sizes() -> None:
    torch.manual_seed(0)
    model_a = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    model_b = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    schedule = BernoulliDiffusionSchedule(num_timesteps=4)
    checkpoints_by_size = {4: (model_a, schedule), 5: (model_b, schedule)}

    examples = [
        _example(4, seed=0),
        _example(5, seed=1),
        _example(4, seed=2),
        _example(5, seed=3),
    ]
    probs, hards = predict_matrix_persize(
        checkpoints_by_size, examples, mode="one_shot", device="cpu", seed=0
    )
    assert len(probs) == len(examples) == len(hards)
    for example, prob, hard in zip(examples, probs, hards, strict=True):
        n = example.instance.n_customers
        _assert_valid_prob(prob, n)
        _assert_valid_hard(hard, n)

    # Cross-check against calling each size's model directly, in isolation.
    direct_probs_4, _ = predict_matrix_batch(
        model_a, schedule, [examples[0], examples[2]], mode="one_shot", device="cpu", seed=0
    )
    assert np.allclose(probs[0], direct_probs_4[0])
    assert np.allclose(probs[2], direct_probs_4[1])


def test_predict_matrix_persize_raises_on_missing_size() -> None:
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    schedule = BernoulliDiffusionSchedule(num_timesteps=4)
    examples = [_example(4, seed=0), _example(6, seed=1)]
    with pytest.raises(ValueError, match=r"no checkpoint configured for sizes: \[6\]"):
        predict_matrix_persize(
            {4: (model, schedule)}, examples, mode="one_shot", device="cpu", seed=0
        )


def test_evaluate_full_chain_sampling_keys() -> None:
    torch.manual_seed(0)
    examples = [_example(4, seed=0), _example(5, seed=1)]
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=4)
    metrics = evaluate_full_chain_sampling(model, schedule, examples, seed=0)
    assert "sample_f1" in metrics
    assert "sample_f1" in metrics
    assert metrics["sample_num_examples"] == 2
    assert "sample_f1_n4" in metrics
    assert "sample_f1_n5" in metrics
    assert metrics["route_num_examples"] == 2
    assert "route_feasible_rate" in metrics
    assert "route_mean_cost_gap_percent_n4" in metrics


def test_evaluate_full_chain_sampling_supports_same_size_batches() -> None:
    torch.manual_seed(8)
    examples = [_example(4, seed=8), _example(4, seed=9), _example(5, seed=10)]
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=3)

    metrics = evaluate_full_chain_sampling(
        model,
        schedule,
        examples,
        seed=8,
        batch_size=2,
    )

    assert metrics["sample_num_examples"] == 3
    assert metrics["sample_batch_size"] == 2
    assert metrics["sample_num_examples_n4"] == 2
    assert metrics["sample_num_examples_n5"] == 1


def test_evaluate_full_chain_sampling_rejects_invalid_batch_size() -> None:
    example = _example(4, seed=11)
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=3)
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_full_chain_sampling(model, schedule, [example], batch_size=0)
