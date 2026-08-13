"""CUDA-only forward, backward, diffusion, and checkpoint smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as ff

from vrp_diffusion_quantum.data.dataset import make_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.inference.predict_matrix import load_denoiser_checkpoint
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.models.gat_encoder import GATConstraintPretrainer
from vrp_diffusion_quantum.train.train_diffusion import (
    save_denoiser_checkpoint,
    train_constraint_denoiser,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


def _cuda_batch(n_customers: int = 6) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    coords = torch.rand(2, n_customers, 2, device=device)
    demands = torch.ones(2, n_customers, device=device)
    capacity = torch.full((2,), float(n_customers), device=device)
    customer_mask = torch.ones(2, n_customers, dtype=torch.bool, device=device)
    m_true = torch.zeros(2, n_customers, n_customers, device=device)
    m_true[:, : n_customers // 2, : n_customers // 2] = 1.0
    diagonal = torch.arange(n_customers, device=device)
    m_true[:, diagonal, diagonal] = 0.0
    return coords, demands, capacity, customer_mask, m_true


def _cuda_example(seed: int) -> CVRPExample:
    rng = np.random.default_rng(seed)
    instance = CVRPInstance(
        coords=np.vstack(([0.5, 0.5], rng.random((6, 2)))),
        demands=np.array([0.0, 1, 1, 1, 1, 1, 1]),
        capacity=6.0,
        depot_index=0,
        instance_id=f"cuda_{seed}",
        n_customers=6,
        seed=seed,
        generator_settings={},
    )
    solution = LabeledSolution(
        routes=[[0, 1, 2], [3, 4, 5]],
        cost=1.0,
        num_vehicles=2,
        feasible=True,
        solver_name="unit",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_cuda_gat_forward_backward_and_optimizer_step() -> None:
    coords, demands, capacity, customer_mask, m_true = _cuda_batch()
    model = GATConstraintPretrainer(hidden_dim=16, gat_num_layers=1, gat_num_heads=4).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logits = model(coords, demands, capacity, customer_mask=customer_mask)
    off_diagonal = ~torch.eye(logits.shape[-1], dtype=torch.bool, device="cuda")
    loss = ff.binary_cross_entropy_with_logits(logits[:, off_diagonal], m_true[:, off_diagonal])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cuda_diffusion_noising_and_denoising_backward() -> None:
    coords, demands, capacity, customer_mask, m_true = _cuda_batch()
    schedule = BernoulliDiffusionSchedule(num_timesteps=8).cuda()
    timesteps = torch.tensor([2, 6], device="cuda")
    m_t = schedule.q_sample(m_true, timesteps, customer_mask=customer_mask)
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16).cuda()

    logits = model(coords, demands, capacity, m_t, timesteps, customer_mask=customer_mask)
    loss = logits.square().mean()
    loss.backward()

    assert logits.is_cuda
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)


def test_cuda_checkpoint_loads_on_cpu_and_cuda(tmp_path: Path) -> None:
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "cuda.pt"
    save_denoiser_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=0,
        row={"val_loss": 1.0},
        best_metric_name="val_loss",
        best_metric_value=1.0,
        extra={"model": {"hidden_dim": 16, "num_layers": 1, "time_embed_dim": 16}},
    )

    cpu_model, _ = load_denoiser_checkpoint(checkpoint, device="cpu")
    cuda_model, _ = load_denoiser_checkpoint(checkpoint, device="cuda")

    assert next(cpu_model.parameters()).device.type == "cpu"
    assert next(cuda_model.parameters()).device.type == "cuda"


def test_cuda_amp_training_step_with_gradient_accumulation() -> None:
    examples = [_cuda_example(0), _cuda_example(1)]
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=8)

    history = train_constraint_denoiser(
        model,
        schedule,
        examples,
        num_epochs=1,
        learning_rate=1e-3,
        batch_size=1,
        seed=0,
        device="cuda",
        mixed_precision=True,
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
    )

    assert history[0]["mixed_precision"] is True
    assert history[0]["optimizer_steps"] == 1
    assert history[0]["peak_cuda_memory_bytes"] > 0
    assert torch.isfinite(torch.tensor(history[0]["train_loss"]))
