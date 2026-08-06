"""Bernoulli diffusion schedule for binary constraint matrices (CMD §IV-B2).

Symmetric bit-flip kernel ``Q_t`` with linear ``beta`` in ``[1e-4, 2e-2]``. Cumulative
flip rate ``q_bar_flip`` goes from ~0 to 0.5. Sampling keeps ``M`` symmetric with zero
diagonal; pass ``customer_mask`` for padded batches.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

__all__ = [
    "BETA_END",
    "BETA_START",
    "NUM_TIMESTEPS",
    "BernoulliDiffusionSchedule",
    "linear_beta_schedule",
]

BETA_START: float = 1e-4
BETA_END: float = 2e-2
NUM_TIMESTEPS: int = 700  # matches configs/train/diffusion_denoiser.yaml


def linear_beta_schedule(
    num_timesteps: int = NUM_TIMESTEPS,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
) -> Tensor:
    """Linear ``beta`` schedule of shape ``(num_timesteps,)``; requires ``0 < beta < 0.5``."""
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
    if not 0.0 < beta_start <= beta_end < 0.5:
        raise ValueError(
            "betas must satisfy 0 < beta_start <= beta_end < 0.5, got "
            f"beta_start={beta_start}, beta_end={beta_end}"
        )
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def _extract(values: Tensor, t: Tensor | int, broadcast_to: Tensor) -> Tensor:
    """Gather ``values[t]`` and broadcast to ``broadcast_to``'s shape."""
    index = t if isinstance(t, Tensor) else torch.as_tensor(t, device=values.device)
    index = index.to(device=values.device, dtype=torch.long)
    gathered = values[index]
    while gathered.dim() < broadcast_to.dim():
        gathered = gathered.unsqueeze(-1)
    return gathered.to(dtype=broadcast_to.dtype)


class BernoulliDiffusionSchedule(nn.Module):
    """Discrete Bernoulli schedule for symmetric binary ``M``.

    ``t=0`` is least noised; ``t=T-1`` is near max-entropy. Buffers move with ``.to(device)``.
    """

    betas: Tensor
    q_bar_flip: Tensor
    q_bar_flip_prev: Tensor

    def __init__(
        self,
        num_timesteps: int = NUM_TIMESTEPS,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        # signal = 1 - 2β; cumprod → cumulative flip prob via q_bar_flip = (1 - signal_bar) / 2
        signal = 1.0 - 2.0 * betas
        signal_bar = torch.cumprod(signal, dim=0)
        q_bar_flip = 0.5 * (1.0 - signal_bar)
        signal_bar_prev = torch.cat([signal_bar.new_ones(1), signal_bar[:-1]])
        q_bar_flip_prev = 0.5 * (1.0 - signal_bar_prev)

        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("q_bar_flip", q_bar_flip.to(torch.float32))
        self.register_buffer("q_bar_flip_prev", q_bar_flip_prev.to(torch.float32))

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        mode: str = "uniform",
    ) -> Tensor:
        """Sample ``t ∈ [0, T)``. ``mode='high'`` biases toward large ``t`` (near prior)."""
        if mode not in ("uniform", "high"):
            raise ValueError(f"mode must be 'uniform' or 'high', got {mode!r}")
        sample_device: torch.device | str | None
        if generator is not None:
            sample_device = generator.device
        else:
            sample_device = device
        if mode == "uniform":
            timesteps = torch.randint(
                0, self.num_timesteps, (batch_size,), generator=generator, device=sample_device
            )
        else:
            # t = floor(T * (1 - (1-u)^2)) puts more mass near T-1
            u = torch.rand(batch_size, generator=generator, device=sample_device)
            timesteps = torch.floor((1.0 - (1.0 - u).square()) * self.num_timesteps).long()
            timesteps = timesteps.clamp(0, self.num_timesteps - 1)
        if device is not None:
            timesteps = timesteps.to(device)
        return timesteps

    def marginal_prob(self, m_true: Tensor, t: Tensor | int) -> Tensor:
        """``P(x_t = 1 | x_0)`` entrywise (CMD eq. 6)."""
        m = m_true.to(dtype=torch.get_default_dtype())
        flip = _extract(self.q_bar_flip, t, m)
        return m * (1.0 - flip) + (1.0 - m) * flip

    def q_sample(
        self,
        m_true: Tensor,
        t: Tensor | int,
        *,
        customer_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample ``m_t ~ q(x_t | x_0)``; symmetrize + zero diagonal (optional ``customer_mask``)."""
        prob_one = self.marginal_prob(m_true, t)
        if generator is not None and torch.device(generator.device).type != prob_one.device.type:
            noise = torch.rand(
                prob_one.shape,
                generator=generator,
                device=generator.device,
                dtype=torch.float32,
            ).to(device=prob_one.device, dtype=prob_one.dtype)
        else:
            noise = torch.rand(
                prob_one.shape, generator=generator, device=prob_one.device, dtype=prob_one.dtype
            )
        sample = (noise < prob_one).to(dtype=prob_one.dtype)
        return self._symmetrize(sample, customer_mask)

    def q_posterior_prob(self, m_t: Tensor, m_true: Tensor, t: Tensor | int) -> Tensor:
        """``P(x_{t-1} = 1 | x_t, x_0)`` (CMD eq. 8). At ``t=0`` this is ``m_true``."""
        m_t_f = m_t.to(dtype=torch.get_default_dtype())
        m0_f = m_true.to(dtype=torch.get_default_dtype())
        beta = _extract(self.betas, t, m_t_f)
        flip_prev = _extract(self.q_bar_flip_prev, t, m_t_f)

        marg_one = m0_f * (1.0 - flip_prev) + (1.0 - m0_f) * flip_prev
        marg_zero = 1.0 - marg_one
        fwd_one = m_t_f * (1.0 - beta) + (1.0 - m_t_f) * beta
        fwd_zero = (1.0 - m_t_f) * (1.0 - beta) + m_t_f * beta

        unnorm_one = fwd_one * marg_one
        unnorm_zero = fwd_zero * marg_zero
        return unnorm_one / (unnorm_one + unnorm_zero)

    @staticmethod
    def _symmetrize(matrix: Tensor, customer_mask: Tensor | None) -> Tensor:
        """Upper-triangle mirror, zero diagonal, optional customer mask."""
        upper = torch.triu(matrix, diagonal=1)
        symmetric = upper + upper.transpose(-1, -2)
        if customer_mask is not None:
            mask = customer_mask.to(dtype=symmetric.dtype)
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            symmetric = symmetric * pair_mask
        return symmetric
