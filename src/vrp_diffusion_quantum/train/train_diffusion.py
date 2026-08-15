"""Train the constraint-matrix denoiser (library + CLI).

Library: :func:`train_constraint_denoiser` and helpers.

CLI::

    python -m vrp_diffusion_quantum.train.train_diffusion \\
      --config configs/train/diffusion_denoiser.yaml

MLflow: ``mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db``
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as ff
import yaml
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from vrp_diffusion_quantum.data.dataset import (
    CVRPBatch,
    collate_batch,
    load_dataset,
    size_homogeneous_batches,
    size_homogeneous_chunks,
)
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.inference.predict_matrix import (
    evaluate_full_chain_sampling,
    select_examples_by_size,
)
from vrp_diffusion_quantum.metrics.matrix_metrics import (
    MatrixMetrics,
    MatrixPrediction,
    compute_matrix_metrics,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.utils.experiment import ExperimentTracker
from vrp_diffusion_quantum.utils.runtime import (
    capture_rng_state,
    default_mlflow_tracking_uri,
    resolve_device,
    restore_rng_state,
    seed_everything,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _ROOT / "configs" / "train" / "diffusion_denoiser.yaml"

__all__ = [
    "customer_tensors_from_batch",
    "diffusion_matrix_bce_loss",
    "evaluate_constraint_denoiser",
    "main",
    "save_denoiser_checkpoint",
    "train_constraint_denoiser",
]

EpochCallback = Callable[[dict[str, Any]], None]


def customer_tensors_from_batch(
    batch: CVRPBatch,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gather per-customer coords/demands (depot excluded) from a collated batch.

    Returns:
        ``(coords, demands, capacity)`` with shapes ``[B, max_n, 2]``, ``[B, max_n]``, ``[B]``.
        Padded customer slots are zeroed via ``batch.customer_mask``.
    """
    idx = batch.customer_node_indices.clamp(min=0)
    coords = torch.gather(batch.coords, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    demands = torch.gather(batch.demands, 1, idx)
    mask = batch.customer_mask
    return coords * mask.unsqueeze(-1), demands * mask, batch.capacity


def diffusion_matrix_bce_loss(
    logits: Tensor,
    m_true: Tensor,
    customer_mask: Tensor | None = None,
    *,
    weighted: bool = True,
    pos_weight_power: float = 0.5,
) -> Tensor:
    """BCE-with-logits on off-diagonal real pairs.

    When ``weighted``, uses ``pos_weight = (neg/pos) ** pos_weight_power``
    (default ``0.5`` = soft / sqrt WBCE).
    """
    if logits.shape != m_true.shape:
        raise ValueError(f"logits shape {logits.shape} != m_true shape {m_true.shape}")
    batch_size, n_customers, _ = logits.shape
    off_diagonal = ~torch.eye(n_customers, dtype=torch.bool, device=logits.device)
    pair_ok = off_diagonal.unsqueeze(0).expand(batch_size, -1, -1)
    if customer_mask is not None:
        real_pair = customer_mask.unsqueeze(-1) & customer_mask.unsqueeze(-2)
        pair_ok = pair_ok & real_pair
    if not bool(pair_ok.any()):
        raise ValueError("no off-diagonal real customer pairs to score")
    targets = m_true[pair_ok].float()
    logits_flat = logits[pair_ok]
    if not weighted:
        return ff.binary_cross_entropy_with_logits(logits_flat, targets)
    pos = targets.sum()
    neg = targets.numel() - pos
    ratio = (neg / pos.clamp(min=1.0)).detach()
    pos_weight = ratio ** float(pos_weight_power)
    return ff.binary_cross_entropy_with_logits(logits_flat, targets, pos_weight=pos_weight)


def _batches(
    examples: list[CVRPExample],
    batch_size: int,
    *,
    same_size: bool = False,
    generator: torch.Generator | None = None,
    shuffle: bool = True,
    augmentation: bool = False,
) -> Iterator[CVRPBatch]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if same_size or augmentation:
        yield from size_homogeneous_batches(
            examples,
            batch_size,
            generator=generator,
            shuffle=shuffle,
            augmentation=augmentation,
        )
        return
    for i in range(0, len(examples), batch_size):
        yield collate_batch(examples[i : i + batch_size])


def _example_chunks(
    examples: list[CVRPExample],
    batch_size: int,
    *,
    same_size: bool = False,
    generator: torch.Generator | None = None,
    shuffle: bool = True,
    augmentation: bool = False,
) -> Iterator[list[CVRPExample]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if same_size or augmentation:
        yield from size_homogeneous_chunks(
            examples,
            batch_size,
            generator=generator,
            shuffle=shuffle,
            augmentation=augmentation,
        )
        return
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


def _shuffle_and_maybe_augment_online(
    examples: list[CVRPExample],
    *,
    seed: int,
    epoch: int,
    online_augmentation: bool,
    shuffle: bool = True,
) -> list[CVRPExample]:
    """Optionally shuffle and select one label-preserving augmented view per example."""
    from vrp_diffusion_quantum.data.augment import AUGMENT_NUM, augment_example

    if shuffle:
        gen = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(examples), generator=gen)
        ordered = [examples[int(i)] for i in order]
    else:
        ordered = list(examples)
    if not online_augmentation:
        return ordered
    aug_gen = torch.Generator().manual_seed(seed + 17_000 + epoch)
    vs = torch.randint(0, AUGMENT_NUM, (len(ordered),), generator=aug_gen)
    return [augment_example(ex, int(v)) for ex, v in zip(ordered, vs.tolist(), strict=True)]


def _noise_batch(
    schedule: BernoulliDiffusionSchedule,
    batch: CVRPBatch,
    *,
    generator: torch.Generator | None = None,
    t_sample: str = "uniform",
) -> tuple[Tensor, Tensor]:
    """Sample per-example timesteps and a corrupted ``m_t`` for ``batch``."""
    batch_size = batch.constraint_matrix.shape[0]
    device = batch.constraint_matrix.device
    t = schedule.sample_timesteps(batch_size, device=device, generator=generator, mode=t_sample)
    m_t = schedule.q_sample(
        batch.constraint_matrix,
        t,
        customer_mask=batch.customer_mask,
        generator=generator,
    )
    return m_t, t


@torch.no_grad()
def evaluate_constraint_denoiser(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    examples: list[CVRPExample],
    *,
    batch_size: int = 8,
    generator: torch.Generator | None = None,
    same_size_batches: bool = False,
    weighted_bce: bool = True,
    pos_weight_power: float = 0.5,
    decision_threshold: float | None = None,
    adaptive_threshold: bool = True,
    t_sample: str = "uniform",
    mixed_precision: bool = False,
) -> tuple[float, MatrixMetrics]:
    """Return mean val loss and matrix metrics on ``examples``.

    ``t_sample`` should match training (``uniform`` or ``high``) so ``val_loss`` is on the
    same noise-time distribution as ``train_loss``.
    """
    if not examples:
        raise ValueError("cannot evaluate on an empty list of examples")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if t_sample not in ("uniform", "high"):
        raise ValueError(f"t_sample must be 'uniform' or 'high', got {t_sample!r}")

    model.eval()
    total_loss = 0.0
    num_batches = 0
    predictions: list[MatrixPrediction] = []

    model_device = next(model.parameters()).device
    amp_enabled = mixed_precision and model_device.type == "cuda"
    for chunk in _example_chunks(examples, batch_size, same_size=same_size_batches, shuffle=False):
        batch = collate_batch(chunk)
        if model_device.type != "cpu":
            batch = _batch_to_device(batch, model_device)
        coords, demands, capacity = customer_tensors_from_batch(batch)
        m_t, t = _noise_batch(schedule, batch, generator=generator, t_sample=t_sample)
        with torch.autocast(
            device_type=model_device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask)
            loss = diffusion_matrix_bce_loss(
                logits,
                batch.constraint_matrix,
                batch.customer_mask,
                weighted=weighted_bce,
                pos_weight_power=pos_weight_power,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite validation loss")
        total_loss += float(loss.item())
        num_batches += 1

        with torch.autocast(
            device_type=model_device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            m_prob = model.predict_proba(
                coords, demands, capacity, m_t, t, customer_mask=batch.customer_mask
            )
        for i, example in enumerate(chunk):
            n_customers = example.instance.n_customers
            predictions.append(
                MatrixPrediction.from_example(
                    example,
                    m_prob[i, :n_customers, :n_customers].detach().cpu().numpy().astype(np.float64),
                )
            )

    mean_loss = total_loss / max(num_batches, 1)
    return mean_loss, compute_matrix_metrics(
        predictions,
        threshold=decision_threshold,
        adaptive_threshold=adaptive_threshold,
    )


def save_denoiser_checkpoint(
    path: str | Path,
    *,
    model: ConstraintDenoiser,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    row: dict[str, Any],
    best_metric_name: str,
    best_metric_value: float,
    extra: dict[str, Any] | None = None,
    scaler: GradScaler | None = None,
) -> Path:
    """Write a resumable checkpoint ``.pt`` (model + optimizer + epoch metrics)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": dict(row),
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "rng_state": capture_rng_state(),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, target)
    return target


def train_constraint_denoiser(
    model: ConstraintDenoiser,
    schedule: BernoulliDiffusionSchedule,
    train_examples: list[CVRPExample],
    *,
    val_examples: list[CVRPExample] | None = None,
    num_epochs: int,
    learning_rate: float,
    batch_size: int = 8,
    seed: int = 0,
    device: torch.device | str | None = None,
    checkpoint_dir: str | Path | None = None,
    best_metric: str = "val_auc",
    minimize_best: bool = False,
    early_stop_patience: int = 0,
    on_epoch_end: EpochCallback | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
    sample_eval_examples: list[CVRPExample] | None = None,
    sample_eval_every: int | None = None,
    resume_checkpoint: str | Path | None = None,
    online_augmentation: bool = False,
    same_size_batches: bool = False,
    augmentation: bool = False,
    weighted_bce: bool = True,
    pos_weight_power: float = 0.5,
    decision_threshold: float | None = None,
    adaptive_threshold: bool = True,
    t_sample: str = "uniform",
    sample_step_stride: int = 1,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float | None = None,
    max_runtime_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Train ``model`` to reconstruct clean ``M`` from noisy ``M_t``."""
    if augmentation:
        online_augmentation = False
    expand_any = augmentation
    if t_sample not in ("uniform", "high"):
        raise ValueError(f"t_sample must be 'uniform' or 'high', got {t_sample!r}")
    if sample_step_stride < 1:
        raise ValueError(f"sample_step_stride must be >= 1, got {sample_step_stride}")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when set")
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive when set")
    if not train_examples:
        raise ValueError("cannot train on an empty list of examples")
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    resolved_val = val_examples if val_examples is not None else train_examples
    if not resolved_val:
        raise ValueError("cannot validate on an empty list of examples")

    if device is not None:
        model.to(device)
        schedule.to(device)
    model_device = next(model.parameters()).device
    if mixed_precision and model_device.type != "cuda":
        raise ValueError("mixed_precision requires a CUDA device")
    amp_enabled = mixed_precision and model_device.type == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_value: float | None = None
    epochs_without_improve = 0
    start_epoch = 0
    if resume_checkpoint is not None:
        from vrp_diffusion_quantum.models.gat_encoder import compat_layernorm_state_dict

        resume_path = Path(resume_checkpoint)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(compat_layernorm_state_dict(payload["model"]))
        if device is not None:
            model.to(device)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        if payload.get("rng_state"):
            restore_rng_state(payload["rng_state"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        if payload.get("best_metric_value") is not None:
            best_value = float(payload["best_metric_value"])
        logger.info(
            "resumed from %s at epoch=%d (next=%d) best_%s=%s",
            resume_path,
            start_epoch - 1,
            start_epoch,
            payload.get("best_metric_name", best_metric),
            best_value,
        )
        if start_epoch >= num_epochs:
            logger.info(
                "resume start_epoch=%d >= num_epochs=%d; nothing to train",
                start_epoch,
                num_epochs,
            )
            return history

    training_started = time.perf_counter()
    total_optimizer_steps = 0
    for epoch in range(start_epoch, num_epochs):
        epoch_started = time.perf_counter()
        if model_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model_device)
        model.train()
        train_generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        epoch_loss = 0.0
        num_batches = 0
        examples_seen = 0
        optimizer_steps = 0
        pending_microbatches = 0
        last_gradient_norm = float("nan")
        epoch_examples = _shuffle_and_maybe_augment_online(
            train_examples,
            seed=seed,
            epoch=epoch,
            online_augmentation=online_augmentation,
            shuffle=not (same_size_batches or expand_any),
        )
        batch_gen = torch.Generator().manual_seed(seed + 3_000 + epoch)

        optimizer.zero_grad(set_to_none=True)

        def optimizer_step(*, gradient_scale_correction: float = 1.0) -> float:
            scaler.unscale_(optimizer)
            if gradient_scale_correction != 1.0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_scale_correction)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm if gradient_clip_norm is not None else float("inf"),
                # GradScaler records non-finite gradients during unscale_ and skips
                # the optimizer step below while reducing its scale. Raising here
                # prevents that normal AMP recovery path from running.
                error_if_nonfinite=not amp_enabled,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return float(norm.detach().cpu())

        try:
            for batch in _batches(
                epoch_examples,
                batch_size,
                same_size=same_size_batches or expand_any,
                generator=batch_gen,
                shuffle=same_size_batches or expand_any,
                augmentation=augmentation,
            ):
                if device is not None:
                    batch = _batch_to_device(batch, device)
                coords, demands, capacity = customer_tensors_from_batch(batch)
                m_t, t = _noise_batch(schedule, batch, generator=train_generator, t_sample=t_sample)

                with torch.autocast(
                    device_type=model_device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(
                        coords,
                        demands,
                        capacity,
                        m_t,
                        t,
                        customer_mask=batch.customer_mask,
                    )
                    loss = diffusion_matrix_bce_loss(
                        logits,
                        batch.constraint_matrix,
                        batch.customer_mask,
                        weighted=weighted_bce,
                        pos_weight_power=pos_weight_power,
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite training loss at epoch={epoch}")
                scaled_loss = loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()  # type: ignore[no-untyped-call]
                pending_microbatches += 1

                epoch_loss += float(loss.item())
                num_batches += 1
                examples_seen += int(batch.coords.shape[0])
                if pending_microbatches == gradient_accumulation_steps:
                    last_gradient_norm = optimizer_step()
                    pending_microbatches = 0
                    optimizer_steps += 1
                    total_optimizer_steps += 1

            if pending_microbatches:
                # Correct the final short accumulation group back to its mean gradient.
                last_gradient_norm = optimizer_step(
                    gradient_scale_correction=(gradient_accumulation_steps / pending_microbatches)
                )
                optimizer_steps += 1
                total_optimizer_steps += 1
        except torch.OutOfMemoryError as exc:
            if model_device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory during diffusion training. Reduce training.batch_size, "
                "increase training.gradient_accumulation_steps, keep mixed_precision enabled, "
                "or disable x9 augmentation."
            ) from exc

        train_loss = epoch_loss / max(num_batches, 1)

        val_generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 + epoch)
        val_loss, val_metrics = evaluate_constraint_denoiser(
            model,
            schedule,
            resolved_val,
            batch_size=batch_size,
            generator=val_generator,
            same_size_batches=same_size_batches or expand_any,
            weighted_bce=weighted_bce,
            pos_weight_power=pos_weight_power,
            decision_threshold=decision_threshold,
            adaptive_threshold=adaptive_threshold,
            t_sample=t_sample,
            mixed_precision=mixed_precision,
        )
        epoch_runtime = time.perf_counter() - epoch_started
        total_runtime = time.perf_counter() - training_started
        peak_cuda_memory = (
            int(torch.cuda.max_memory_allocated(model_device)) if model_device.type == "cuda" else 0
        )
        peak_cuda_memory_reserved = (
            int(torch.cuda.max_memory_reserved(model_device)) if model_device.type == "cuda" else 0
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bce": val_metrics.bce,
            "val_auc": val_metrics.auc,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
            "val_accuracy": val_metrics.accuracy,
            "val_threshold": val_metrics.threshold,
            "val_calibration_error": val_metrics.calibration_error,
            "val_capacity_consistency": val_metrics.capacity_consistency,
            "epoch_runtime_seconds": epoch_runtime,
            "total_runtime_seconds": total_runtime,
            "train_examples_seen": examples_seen,
            "train_examples_per_second": examples_seen / max(epoch_runtime, 1e-9),
            "microbatches": num_batches,
            "optimizer_steps": optimizer_steps,
            "optimizer_steps_per_second": optimizer_steps / max(epoch_runtime, 1e-9),
            "total_optimizer_steps": total_optimizer_steps,
            "gradient_norm": last_gradient_norm,
            "mixed_precision": amp_enabled,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "peak_cuda_memory_bytes": peak_cuda_memory,
            "peak_cuda_memory_reserved_bytes": peak_cuda_memory_reserved,
        }

        run_sample = (
            sample_eval_examples is not None
            and sample_eval_every is not None
            and sample_eval_every > 0
            and ((epoch + 1) % sample_eval_every == 0 or epoch + 1 == num_epochs)
        )
        if run_sample:
            assert sample_eval_examples is not None
            sample_metrics = evaluate_full_chain_sampling(
                model,
                schedule,
                sample_eval_examples,
                device=device,
                seed=seed + 20_000 + epoch,
                threshold=0.5,
                adaptive_threshold=False,
                step_stride=sample_step_stride,
            )
            row.update(sample_metrics)
            logger.info(
                "epoch=%d sample_f1=%.4f sample_precision=%.4f sample_recall=%.4f (full T→0, n=%d)",
                epoch,
                sample_metrics["sample_f1"],
                sample_metrics["sample_precision"],
                sample_metrics["sample_recall"],
                sample_metrics["sample_num_examples"],
            )

        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f val_bce=%.6f "
            "val_auc=%s val_f1=%.4f val_acc=%.4f val_prec=%.4f val_rec=%.4f thr=%.2f",
            epoch,
            train_loss,
            val_loss,
            val_metrics.bce,
            val_metrics.auc,
            val_metrics.f1,
            val_metrics.accuracy,
            val_metrics.precision,
            val_metrics.recall,
            val_metrics.threshold,
        )

        metric_raw = row.get(best_metric)
        # sample_f1 only exists on sample_eval epochs — skip best/early-stop that epoch.
        track_best = metric_raw is not None
        if track_best and isinstance(metric_raw, (int, float)) and metric_raw == metric_raw:
            metric_value = float(metric_raw)
        elif track_best:
            # AUC can be None when undefined; skip best updates but still save last.pt.
            metric_value = float("inf") if minimize_best else float("-inf")
            track_best = False
        else:
            metric_value = float("inf") if minimize_best else float("-inf")

        is_best = False
        if track_best:
            if best_value is None:
                is_best = True
            elif minimize_best and metric_value < best_value:
                is_best = True
            elif not minimize_best and metric_value > best_value:
                is_best = True
            if is_best and metric_value not in (float("inf"), float("-inf")):
                best_value = metric_value
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

        callback_row = dict(row)
        if ckpt_dir is not None:
            tracked_best = best_value if best_value is not None else metric_value
            last_path = save_denoiser_checkpoint(
                ckpt_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                row=row,
                best_metric_name=best_metric,
                best_metric_value=tracked_best,
                extra=checkpoint_extra,
                scaler=scaler,
            )
            callback_row["checkpoint_last"] = str(last_path)
            if is_best and best_value is not None:
                best_path = save_denoiser_checkpoint(
                    ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    row=row,
                    best_metric_name=best_metric,
                    best_metric_value=best_value,
                    extra=checkpoint_extra,
                    scaler=scaler,
                )
                callback_row["checkpoint_best"] = str(best_path)
                callback_row["is_best"] = True
            else:
                callback_row["is_best"] = False

        if on_epoch_end is not None:
            on_epoch_end(callback_row)

        if (
            early_stop_patience > 0
            and epochs_without_improve >= early_stop_patience
            and best_value is not None
        ):
            logger.info(
                "early stop at epoch=%d (patience=%d, best_%s=%.6f)",
                epoch,
                early_stop_patience,
                best_metric,
                best_value,
            )
            break

        if max_runtime_seconds is not None and total_runtime >= max_runtime_seconds:
            logger.info(
                "runtime budget reached after epoch=%d: %.1fs >= %.1fs",
                epoch,
                total_runtime,
                max_runtime_seconds,
            )
            break

    return history


def _batch_to_device(batch: CVRPBatch, device: torch.device | str) -> CVRPBatch:
    """Move tensor fields of a :class:`CVRPBatch` onto ``device`` (metadata stays on CPU)."""
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConstraintDenoiser (noisy M_t → clean M).")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to last.pt/best.pt to continue training from epoch+1",
    )
    return parser.parse_args()


def _flatten_params(obj: object, *, prefix: str = "") -> dict[str, str | int | float | bool]:
    """Flatten nested config into MLflow-friendly scalar params."""
    out: dict[str, str | int | float | bool] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_params(value, prefix=name))
        return out
    if isinstance(obj, (list, tuple)):
        out[prefix or "value"] = ",".join(str(v) for v in obj)
        return out
    if isinstance(obj, bool):
        out[prefix or "value"] = obj
        return out
    if isinstance(obj, int | float):
        out[prefix or "value"] = obj
        return out
    if obj is None:
        out[prefix or "value"] = "null"
        return out
    out[prefix or "value"] = str(obj)
    return out


def main() -> None:
    """CLI entry: config → model → :func:`train_constraint_denoiser` (+ MLflow)."""
    args = _parse_args()
    cfg_path = args.config if args.config.is_absolute() else _ROOT / args.config
    config = yaml.safe_load(cfg_path.read_text())
    seed = int(config["seed"])
    config["reproducibility"] = seed_everything(
        seed,
        deterministic=bool(config.get("training", {}).get("deterministic", False)),
    )

    dataset_path_value = config["dataset"].get("path")
    if not dataset_path_value:
        raise ValueError(
            "dataset.path must be set; use a corpus-specific config such as "
            "configs/train/diffusion_denoiser_s7799.yaml"
        )
    dataset_path = _ROOT / dataset_path_value
    output_root = _ROOT / config["output"]["root"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    schedule_cfg = config.get("schedule", {})
    mlflow_cfg = config.get("mlflow", {})
    ckpt_cfg = config.get("checkpoint", {})

    examples = load_dataset(dataset_path)
    if not examples:
        raise ValueError(f"no examples found under {dataset_path}")

    val_cfg = config.get("validation", {})
    val_path = val_cfg.get("path")
    if val_path:
        val_examples = load_dataset(_ROOT / val_path)
        if not val_examples:
            raise ValueError(f"no validation examples found under {_ROOT / val_path}")
    else:
        val_examples = examples

    encoder_type = str(model_cfg.get("node_encoder_type", "linear"))
    if encoder_type not in ("linear", "gat"):
        raise ValueError(f"model.node_encoder_type must be 'linear' or 'gat', got {encoder_type!r}")
    model = ConstraintDenoiser(
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        time_embed_dim=int(model_cfg.get("time_embed_dim", 64)),
        node_encoder_type=encoder_type,  # type: ignore[arg-type]
        gat_num_layers=int(model_cfg.get("gat_num_layers", 5)),
        gat_num_heads=int(model_cfg.get("gat_num_heads", 4)),
        gat_dropout=float(model_cfg.get("gat_dropout", 0.0)),
        freeze_node_encoder=bool(model_cfg.get("freeze_node_encoder", False)),
    )
    gat_ckpt = model_cfg.get("gat_checkpoint")
    if gat_ckpt:
        gat_path = _ROOT / gat_ckpt if not Path(gat_ckpt).is_absolute() else Path(gat_ckpt)
        model.load_gat_pretrained(gat_path)
        if bool(model_cfg.get("freeze_node_encoder", False)):
            model.set_node_encoder_trainable(False)
    elif encoder_type == "gat" and bool(model_cfg.get("freeze_node_encoder", False)):
        raise ValueError(
            "model.node_encoder_type='gat' with freeze_node_encoder=true requires "
            "model.gat_checkpoint (run configs/train/gat_pretrain.yaml first)"
        )
    schedule = BernoulliDiffusionSchedule(
        num_timesteps=int(schedule_cfg.get("num_timesteps", 700)),
        beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
        beta_end=float(schedule_cfg.get("beta_end", 2e-2)),
    )
    device = resolve_device(train_cfg.get("device", "auto"))
    best_metric = str(ckpt_cfg.get("best_metric", "val_auc"))
    minimize_best = bool(ckpt_cfg.get("minimize", False))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0) or 0)
    adaptive_threshold = bool(train_cfg.get("adaptive_threshold", True))
    decision_threshold_raw = train_cfg.get("decision_threshold")
    decision_threshold = None if decision_threshold_raw is None else float(decision_threshold_raw)
    pos_weight_power = float(train_cfg.get("pos_weight_power", 0.5))
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True))
    sample_cfg = config.get("sample_eval") or {}
    sample_eval_every = int(sample_cfg["every"]) if sample_cfg.get("every") else None
    sample_eval_examples = None
    if sample_eval_every is not None:
        sample_eval_examples = select_examples_by_size(
            val_examples,
            sizes=[int(s) for s in sample_cfg.get("sizes", [20, 50, 100])],
            per_size=int(sample_cfg.get("per_size", 4)),
            seed=seed + 12345,
        )
        if not sample_eval_examples:
            raise ValueError(
                "sample_eval.every set but no matching val examples for requested sizes"
            )

    mlflow = None
    if mlflow_enabled:
        import mlflow as mlflow_mod

        mlflow = mlflow_mod
        tracking_uri = str(mlflow_cfg.get("tracking_uri") or default_mlflow_tracking_uri())
        if tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        experiment_name = str(mlflow_cfg.get("experiment_name") or config["experiment_name"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    with ExperimentTracker(
        output_root=output_root,
        experiment_name=config["experiment_name"],
        config=config,
        seed=seed,
        dataset_path=dataset_path,
    ) as tracker:
        checkpoint_dir = tracker.run_dir / "checkpoints"
        run_context = (
            mlflow.start_run(run_name=tracker.run_dir.name) if mlflow is not None else None
        )
        try:
            if mlflow is not None:
                flat = _flatten_params(config)
                for key, value in flat.items():
                    text = str(value)
                    mlflow.log_param(key[:250], text if len(text) <= 500 else text[:497] + "...")
                mlflow.log_param("run_dir", str(tracker.run_dir))
                mlflow.log_param("device", str(device))
                if tracker.dataset_hash is not None:
                    mlflow.log_param("dataset_hash", tracker.dataset_hash)

            skip_keys = {"checkpoint_last", "checkpoint_best", "is_best"}

            def on_epoch_end(row: dict[str, Any]) -> None:
                tracker.log_metric_row({k: v for k, v in row.items() if k not in skip_keys})
                if mlflow is None:
                    return
                step = int(row["epoch"])
                for key, value in row.items():
                    if key in skip_keys:
                        continue
                    if isinstance(value, bool):
                        mlflow.log_metric(key, float(value), step=step)
                    elif isinstance(value, (int, float)) and value == value:
                        mlflow.log_metric(key, float(value), step=step)

            history = train_constraint_denoiser(
                model,
                schedule,
                examples,
                val_examples=val_examples,
                num_epochs=int(train_cfg["epochs"]),
                learning_rate=float(train_cfg["learning_rate"]),
                batch_size=int(train_cfg.get("batch_size", 8)),
                seed=seed,
                device=device,
                checkpoint_dir=checkpoint_dir,
                best_metric=best_metric,
                minimize_best=minimize_best,
                early_stop_patience=early_stop_patience,
                on_epoch_end=on_epoch_end,
                checkpoint_extra={
                    "experiment_name": config["experiment_name"],
                    "seed": seed,
                    "model": model_cfg,
                    "schedule": schedule_cfg,
                    "decision_threshold": decision_threshold,
                    "adaptive_threshold": adaptive_threshold,
                    "pos_weight_power": pos_weight_power,
                    "training_runtime": {
                        "mixed_precision": bool(train_cfg.get("mixed_precision", False)),
                        "gradient_accumulation_steps": int(
                            train_cfg.get("gradient_accumulation_steps", 1)
                        ),
                        "gradient_clip_norm": train_cfg.get("gradient_clip_norm"),
                        "max_runtime_seconds": train_cfg.get("max_runtime_seconds"),
                    },
                },
                sample_eval_examples=sample_eval_examples,
                sample_eval_every=sample_eval_every,
                resume_checkpoint=(
                    None
                    if args.resume is None
                    else (args.resume if args.resume.is_absolute() else _ROOT / args.resume)
                ),
                online_augmentation=bool(train_cfg.get("online_augmentation", False)),
                same_size_batches=bool(train_cfg.get("same_size_batches", False)),
                augmentation=bool(train_cfg.get("augmentation", False)),
                weighted_bce=bool(train_cfg.get("weighted_bce", True)),
                pos_weight_power=pos_weight_power,
                decision_threshold=decision_threshold,
                adaptive_threshold=adaptive_threshold,
                t_sample=str(train_cfg.get("t_sample", "uniform")),
                sample_step_stride=int(sample_cfg.get("step_stride", 1)),
                mixed_precision=bool(train_cfg.get("mixed_precision", False)),
                gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
                gradient_clip_norm=(
                    None
                    if train_cfg.get("gradient_clip_norm") is None
                    else float(train_cfg["gradient_clip_norm"])
                ),
                max_runtime_seconds=(
                    None
                    if train_cfg.get("max_runtime_seconds") is None
                    else float(train_cfg["max_runtime_seconds"])
                ),
            )

            if not history:
                completed_metrics = {
                    "status": "already_complete",
                    "num_epochs_configured": int(train_cfg["epochs"]),
                    "resume_checkpoint": str(args.resume),
                    "dataset_hash": tracker.dataset_hash,
                }
                tracker.log_metrics(completed_metrics)
                tracker.logger.info(
                    "resume checkpoint already reached the configured epoch count; "
                    "no optimization steps were run"
                )
                print("checkpoint already reached the configured epoch count; nothing to train")
                return

            final = history[-1]
            summary_metrics = {
                "final_train_loss": final["train_loss"],
                "final_val_loss": final["val_loss"],
                "final_val_bce": final["val_bce"],
                "final_val_auc": final["val_auc"],
                "final_val_precision": final["val_precision"],
                "final_val_recall": final["val_recall"],
                "final_val_f1": final["val_f1"],
                "final_val_accuracy": final["val_accuracy"],
                "final_val_calibration_error": final["val_calibration_error"],
                "final_val_capacity_consistency": final["val_capacity_consistency"],
                "decision_threshold": final.get("val_threshold", decision_threshold),
                "adaptive_threshold": adaptive_threshold,
                "pos_weight_power": pos_weight_power,
                "num_epochs": len(history),
                "num_epochs_configured": int(train_cfg["epochs"]),
                "num_train_examples": len(examples),
                "num_val_examples": len(val_examples),
                "dataset_hash": tracker.dataset_hash,
                "train_loss_first": history[0]["train_loss"],
                "train_loss_decreased": final["train_loss"] < history[0]["train_loss"],
                "checkpoint_dir": str(checkpoint_dir),
                "best_metric": best_metric,
                "total_runtime_seconds": final["total_runtime_seconds"],
                "train_examples_per_second": final["train_examples_per_second"],
                "optimizer_steps_per_second": final["optimizer_steps_per_second"],
                "peak_cuda_memory_bytes": final["peak_cuda_memory_bytes"],
                "peak_cuda_memory_reserved_bytes": final["peak_cuda_memory_reserved_bytes"],
                "mixed_precision": final["mixed_precision"],
                "gradient_accumulation_steps": final["gradient_accumulation_steps"],
            }
            tracker.log_metrics(summary_metrics)
            if mlflow is not None:
                for key, value in summary_metrics.items():
                    if isinstance(value, bool):
                        mlflow.log_metric(f"summary_{key}", float(value))
                    elif isinstance(value, (int, float)) and value == value:
                        mlflow.log_metric(f"summary_{key}", float(value))
                for ckpt_name in ("best.pt", "last.pt"):
                    ckpt_path = checkpoint_dir / ckpt_name
                    if ckpt_path.is_file():
                        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                for name in ("config.yaml", "summary.csv", "metrics.json"):
                    path = tracker.run_dir / name
                    if path.is_file():
                        mlflow.log_artifact(str(path), artifact_path="experiment")

            tracker.logger.info(
                "diffusion denoiser training complete run_dir=%s checkpoints=%s mlflow=%s",
                tracker.run_dir,
                checkpoint_dir,
                mlflow_enabled,
            )
            print(
                f"final train_loss={final['train_loss']:.6f} "
                f"(first={history[0]['train_loss']:.6f}, decreased="
                f"{final['train_loss'] < history[0]['train_loss']})"
            )
            print(
                f"val: loss={final['val_loss']:.4f} bce={final['val_bce']:.4f} "
                f"auc={final['val_auc']} f1={final['val_f1']:.4f} "
                f"acc={final['val_accuracy']:.4f} "
                f"precision={final['val_precision']:.4f} recall={final['val_recall']:.4f} "
                f"thr={final['val_threshold']}"
            )
            print(f"checkpoints: {checkpoint_dir}")
            if mlflow_enabled:
                uri = str(mlflow_cfg.get("tracking_uri") or default_mlflow_tracking_uri())
                print(f"mlflow ui --backend-store-uri {uri}")
        finally:
            if run_context is not None and mlflow is not None:
                mlflow.end_run()


if __name__ == "__main__":
    main()
