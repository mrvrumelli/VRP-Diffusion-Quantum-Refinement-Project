"""Policy-only helpers kept out of the existing diffusion/data modules.

Collation, geometric views, and a frozen ``M_hat`` provider live here so the original
``dataset``, ``augment``, and ``predict_matrix`` pipelines stay unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from vrp_diffusion_quantum.data.augment import (
    AUGMENT_NUM,
    transform_coords_d4,
    transform_coords_rotation,
)
from vrp_diffusion_quantum.data.dataset import CVRPBatch
from vrp_diffusion_quantum.data.types import CVRPInstance
from vrp_diffusion_quantum.inference.predict_matrix import symmetrize_zero_diagonal
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule

__all__ = [
    "BatchMatrixPrediction",
    "PriorProvider",
    "augment_instance",
    "batch_to_device",
    "collate_instances",
    "constraint_matrix_prior",
    "denoiser_prior",
    "sample_constraint_matrix_batch",
]

PriorProvider = Callable[[CVRPBatch], Tensor]

# Same nine-view layout as ``augment_example`` (identity + 4 D4 + 4 rotations).
_GEO_D4_KS = (1, 2, 3, 4)
_ROTATION_ANGLES_DEGREES = (45.0, 135.0, 225.0, 315.0)


@dataclass(frozen=True)
class BatchMatrixPrediction:
    """Batched reverse-chain output, kept as tensors for the policy encoder."""

    m_hat: Tensor  # [batch, n, n] in {0, 1}, symmetric, zero diagonal
    m_prob: Tensor  # [batch, n, n] in [0, 1], symmetric, zero diagonal
    trajectory: list[Tensor] | None = None
    trajectory_timesteps: list[int] | None = None


def collate_instances(instances: list[CVRPInstance]) -> CVRPBatch:
    """Pad and stack unlabelled instances for policy inference.

    Label fields (``constraint_matrix``, ``cost``, ``num_vehicles``, ``feasible``) stay zero.
    """
    if not instances:
        raise ValueError("cannot collate an empty list of instances")

    max_n_nodes = max(instance.coords.shape[0] for instance in instances)
    max_n_customers = max(instance.n_customers for instance in instances)
    batch_size = len(instances)

    coords = torch.zeros((batch_size, max_n_nodes, 2), dtype=torch.float32)
    demands = torch.zeros((batch_size, max_n_nodes), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_n_nodes), dtype=torch.bool)
    depot_index = torch.zeros((batch_size,), dtype=torch.int64)
    customer_node_indices = torch.full((batch_size, max_n_customers), -1, dtype=torch.int64)
    capacity = torch.zeros((batch_size,), dtype=torch.float32)
    customer_mask = torch.zeros((batch_size, max_n_customers), dtype=torch.bool)
    metadata: list[dict[str, Any]] = []

    for i, instance in enumerate(instances):
        n_nodes = instance.coords.shape[0]
        n_customers = instance.n_customers
        coords[i, :n_nodes] = torch.from_numpy(instance.coords).to(torch.float32)
        demands[i, :n_nodes] = torch.from_numpy(instance.demands).to(torch.float32)
        node_mask[i, :n_nodes] = True
        depot_index[i] = instance.depot_index
        customer_nodes = instance.customer_node_indices()
        customer_node_indices[i, :n_customers] = torch.tensor(customer_nodes, dtype=torch.int64)
        capacity[i] = instance.capacity
        customer_mask[i, :n_customers] = True
        metadata.append(
            {
                "instance_id": instance.instance_id,
                "n_customers": n_customers,
                "depot_index": instance.depot_index,
                "customer_node_indices": customer_nodes,
                "seed": instance.seed,
                "generator_settings": instance.generator_settings,
            }
        )

    return CVRPBatch(
        coords=coords,
        demands=demands,
        node_mask=node_mask,
        depot_index=depot_index,
        customer_node_indices=customer_node_indices,
        capacity=capacity,
        constraint_matrix=torch.zeros(
            (batch_size, max_n_customers, max_n_customers), dtype=torch.float32
        ),
        customer_mask=customer_mask,
        cost=torch.zeros((batch_size,), dtype=torch.float32),
        num_vehicles=torch.zeros((batch_size,), dtype=torch.int64),
        feasible=torch.zeros((batch_size,), dtype=torch.bool),
        metadata=metadata,
    )


def batch_to_device(batch: CVRPBatch, device: torch.device | str) -> CVRPBatch:
    """Move tensor fields of a batch onto ``device``; metadata stays on CPU."""
    return CVRPBatch(
        coords=batch.coords.to(device),
        demands=batch.demands.to(device),
        node_mask=batch.node_mask.to(device),
        depot_index=batch.depot_index.to(device),
        customer_node_indices=batch.customer_node_indices.to(device),
        capacity=batch.capacity.to(device),
        constraint_matrix=batch.constraint_matrix.to(device),
        customer_mask=batch.customer_mask.to(device),
        cost=batch.cost.to(device),
        num_vehicles=batch.num_vehicles.to(device),
        feasible=batch.feasible.to(device),
        metadata=batch.metadata,
    )


def augment_instance(instance: CVRPInstance, variant: int) -> CVRPInstance:
    """One of nine distance-preserving geometric views of an unlabelled instance."""
    if variant < 0 or variant >= AUGMENT_NUM:
        raise ValueError(f"variant must be in 0..{AUGMENT_NUM - 1}, got {variant}")
    if variant == 0:
        return instance
    if variant <= 4:
        k = _GEO_D4_KS[variant - 1]
        return CVRPInstance(
            coords=transform_coords_d4(instance.coords, k),
            demands=np.asarray(instance.demands, dtype=np.float64).copy(),
            capacity=float(instance.capacity),
            depot_index=int(instance.depot_index),
            instance_id=f"{instance.instance_id}_d4{k}",
            n_customers=int(instance.n_customers),
            seed=instance.seed,
            generator_settings={**instance.generator_settings, "d4": int(k)},
        )
    angle = _ROTATION_ANGLES_DEGREES[variant - 5]
    return CVRPInstance(
        coords=transform_coords_rotation(instance.coords, angle),
        demands=np.asarray(instance.demands, dtype=np.float64).copy(),
        capacity=float(instance.capacity),
        depot_index=int(instance.depot_index),
        instance_id=f"{instance.instance_id}_rot{angle:g}",
        n_customers=int(instance.n_customers),
        seed=instance.seed,
        generator_settings={
            **instance.generator_settings,
            "rotation_degrees": float(angle),
        },
    )


def constraint_matrix_prior(batch: CVRPBatch) -> Tensor:
    """Oracle prior: the labelled ``M``. Used in tests and prior-quality ablations."""
    return batch.constraint_matrix


def _rand_tensor(
    *shape: int,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if generator is not None and torch.device(generator.device).type != device.type:
        return torch.rand(
            *shape, generator=generator, device=generator.device, dtype=torch.float32
        ).to(device=device, dtype=dtype)
    return torch.rand(*shape, generator=generator, device=device, dtype=dtype)


def _sample_prior(
    batch_size: int,
    n_customers: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    customer_mask: Tensor | None,
    generator: torch.Generator | None,
    positive_probability: float = 0.5,
) -> Tensor:
    if not 0.0 <= positive_probability <= 1.0:
        raise ValueError(f"positive_probability must be in [0, 1], got {positive_probability}")
    noise = _rand_tensor(
        batch_size,
        n_customers,
        n_customers,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    hard = (noise < positive_probability).to(dtype=dtype)
    return symmetrize_zero_diagonal(hard, customer_mask)


def _inference_timesteps(
    num_timesteps: int,
    *,
    num_inference_steps: int | None,
    step_stride: int,
) -> list[int]:
    """Descending model-evaluation timesteps, including both endpoints when possible."""
    if num_inference_steps is None:
        timesteps = list(range(num_timesteps - 1, -1, -step_stride))
        if timesteps[-1] != 0:
            timesteps.append(0)
        return timesteps
    if num_inference_steps >= num_timesteps:
        return list(range(num_timesteps - 1, -1, -1))
    if num_inference_steps == 1:
        return [num_timesteps - 1]
    span = num_timesteps - 1
    spaced = [
        round(span * (1.0 - index / (num_inference_steps - 1)))
        for index in range(num_inference_steps)
    ]
    # Rounding can collide on adjacent entries; keep the chain strictly descending.
    deduplicated: list[int] = []
    for value in spaced:
        if not deduplicated or value < deduplicated[-1]:
            deduplicated.append(value)
    if deduplicated[-1] != 0:
        deduplicated.append(0)
    return deduplicated


@torch.no_grad()
def sample_constraint_matrix_batch(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    *,
    coords: Tensor,
    demands: Tensor,
    capacity: Tensor,
    customer_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
    threshold: float = 0.5,
    snapshot_every: int | None = None,
    num_inference_steps: int | None = None,
    step_stride: int = 1,
    x0_clamp: float = 1e-3,
    transition_mode: Literal["stochastic", "deterministic"] = "stochastic",
    sampler: Literal["skipped_posterior", "one_step_approx"] = "skipped_posterior",
    prior_positive_probability: float = 0.5,
) -> BatchMatrixPrediction:
    """Run the reverse chain for a padded batch and return tensors for the policy.

    Args:
        num_inference_steps: total number of denoising steps to run, spread evenly over the
            schedule and always ending at ``t=0``. This is the paper's ``T'`` (50). Takes
            precedence over ``step_stride``, which is a skip size and therefore yields
            ``num_timesteps / step_stride`` steps instead.
    """
    if step_stride < 1:
        raise ValueError(f"step_stride must be >= 1, got {step_stride}")
    if num_inference_steps is not None and num_inference_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
    if transition_mode not in {"stochastic", "deterministic"}:
        raise ValueError(f"unsupported transition_mode: {transition_mode}")
    if sampler not in {"skipped_posterior", "one_step_approx"}:
        raise ValueError(f"unsupported sampler: {sampler}")
    model.eval()
    batch_size, n_customers, _ = coords.shape
    device = coords.device
    m_t = _sample_prior(
        batch_size,
        n_customers,
        device=device,
        dtype=coords.dtype,
        customer_mask=customer_mask,
        generator=generator,
        positive_probability=prior_positive_probability,
    )

    trajectory: list[Tensor] = []
    trajectory_t: list[int] = []
    if snapshot_every is not None:
        trajectory.append(m_t.detach().clone())
        trajectory_t.append(schedule.num_timesteps - 1)

    timesteps = _inference_timesteps(
        schedule.num_timesteps,
        num_inference_steps=num_inference_steps,
        step_stride=step_stride,
    )

    m0_prob = m_t
    for step_index, t_int in enumerate(timesteps):
        t_tensor = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        m0_prob = model.predict_proba(
            coords,
            demands,
            capacity,
            m_t,
            t_tensor,
            customer_mask=customer_mask,
        )
        m0_prob = symmetrize_zero_diagonal(m0_prob, customer_mask)
        if x0_clamp > 0:
            m0_prob = m0_prob.clamp(x0_clamp, 1.0 - x0_clamp)

        final_step = step_index == len(timesteps) - 1
        if final_step:
            m_t = symmetrize_zero_diagonal(
                (m0_prob >= threshold).to(dtype=m0_prob.dtype), customer_mask
            )
        else:
            target_t = timesteps[step_index + 1]
            if sampler == "skipped_posterior":
                post = schedule.q_posterior_between_prob(
                    m_t,
                    m0_prob,
                    t=t_tensor,
                    target_t=target_t,
                )
            else:
                post = schedule.q_posterior_prob(m_t, m0_prob, t_tensor)
            if transition_mode == "stochastic":
                noise = _rand_tensor(
                    *post.shape, generator=generator, device=post.device, dtype=post.dtype
                )
                next_state = noise < post
            else:
                next_state = post >= threshold
            m_t = symmetrize_zero_diagonal(next_state.to(dtype=post.dtype), customer_mask)

        if snapshot_every is not None and (
            t_int % snapshot_every == 0 or t_int == 0 or final_step
        ):
            trajectory.append(m_t.detach().clone())
            trajectory_t.append(t_int)

    return BatchMatrixPrediction(
        m_hat=m_t,
        m_prob=m0_prob,
        trajectory=trajectory if snapshot_every is not None else None,
        trajectory_timesteps=trajectory_t if snapshot_every is not None else None,
    )


def denoiser_prior(
    denoiser: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    *,
    num_inference_steps: int | None = None,
    step_stride: int = 1,
    threshold: float = 0.5,
    seed: int = 0,
    use_probabilities: bool = True,
    sampler: Literal["skipped_posterior", "one_step_approx"] = "skipped_posterior",
) -> PriorProvider:
    """Frozen diffusion model that supplies batched ``M_hat`` to the policy."""
    denoiser.eval()
    for parameter in denoiser.parameters():
        parameter.requires_grad_(False)
    call_count = {"value": 0}

    def provide(batch: CVRPBatch) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed + call_count["value"])
        call_count["value"] += 1
        rows = torch.arange(batch.coords.shape[0], device=batch.coords.device)[:, None]
        customer_nodes = batch.customer_node_indices.clamp_min(0)
        coords = batch.coords[rows, customer_nodes] * batch.customer_mask[..., None]
        demands = batch.demands[rows, customer_nodes] * batch.customer_mask
        predicted = sample_constraint_matrix_batch(
            denoiser,
            schedule,
            coords=coords,
            demands=demands,
            capacity=batch.capacity,
            customer_mask=batch.customer_mask,
            generator=generator,
            threshold=threshold,
            num_inference_steps=num_inference_steps,
            step_stride=step_stride,
            sampler=sampler,
        )
        return predicted.m_prob if use_probabilities else predicted.m_hat

    return provide
