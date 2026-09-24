"""REINFORCE training of the dual-pointer policy with a POMO multi-start baseline (P4.6).

Implements CMD Algorithm 1: for each instance the policy samples one trajectory per start node,
the mean tour length across starts becomes the baseline, and the advantage-weighted score function
is the gradient estimator. The diffusion prior stays frozen and only supplies ``M_hat``.

Usage::

    python -m vrp_diffusion_quantum.train.train_policy \\
      --config configs/policy/policy_reinforce.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch import Tensor

from vrp_diffusion_quantum.data.dataset import (
    CVRPBatch,
    IndexedJSONDataset,
    collate_batch,
    size_homogeneous_chunks,
)
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.inference.policy_support import (
    PriorProvider,
    batch_to_device,
    constraint_matrix_prior,
    denoiser_prior,
)
from vrp_diffusion_quantum.inference.predict_matrix import load_denoiser_checkpoint
from vrp_diffusion_quantum.models.decoder import (
    NSTART_CAP,
    CVRPPolicy,
    DecoderRollout,
    PolicyEncoding,
    actions_to_routes,
    nstart_count,
)
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.utils.alignment import (
    require_artifact_alignment,
    validate_alignment_config,
)
from vrp_diffusion_quantum.utils.experiment import ExperimentTracker
from vrp_diffusion_quantum.utils.feasibility import validate_routes
from vrp_diffusion_quantum.utils.runtime import (
    capture_rng_state,
    default_mlflow_tracking_uri,
    resolve_device,
    seed_everything,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _ROOT / "configs" / "policy" / "policy_reinforce.yaml"

__all__ = [
    "BaselineMode",
    "PriorProvider",
    "build_policy_from_config",
    "constraint_matrix_prior",
    "denoiser_prior",
    "evaluate_policy",
    "main",
    "policy_gradient_loss",
    "save_policy_checkpoint",
    "train_policy",
]

BaselineMode = Literal["multi_start", "greedy_rollout", "none"]
EpochCallback = Callable[[dict[str, Any]], None]


def policy_gradient_loss(
    cost: Tensor,
    log_probability: Tensor,
    *,
    num_starts: int,
    baseline_mode: BaselineMode = "multi_start",
    baseline_cost: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Self-critic REINFORCE loss (CMD Algorithm 1, lines 8-9).

    ``cost`` and ``log_probability`` are flat ``[batch * num_starts]`` tensors laid out so that
    consecutive ``num_starts`` entries belong to the same instance. Returns the scalar loss to
    minimize and the per-rollout advantage (already detached).
    """
    if cost.shape != log_probability.shape:
        raise ValueError(
            f"cost {tuple(cost.shape)} and log_probability {tuple(log_probability.shape)} "
            "must have the same shape"
        )
    if num_starts < 1:
        raise ValueError(f"num_starts must be >= 1, got {num_starts}")
    if cost.numel() % num_starts != 0:
        raise ValueError(f"cost size {cost.numel()} is not divisible by num_starts {num_starts}")

    grouped_cost = cost.view(-1, num_starts)
    grouped_log_probability = log_probability.view(-1, num_starts)
    if baseline_mode == "multi_start":
        if num_starts < 2:
            raise ValueError("the multi_start baseline needs num_starts >= 2")
        baseline = grouped_cost.mean(dim=1, keepdim=True)
    elif baseline_mode == "greedy_rollout":
        if baseline_cost is None:
            raise ValueError("the greedy_rollout baseline requires baseline_cost")
        baseline = baseline_cost.view(-1, 1)
    elif baseline_mode == "none":
        baseline = torch.zeros_like(grouped_cost[:, :1])
    else:
        raise ValueError(f"unsupported baseline_mode: {baseline_mode!r}")

    advantage = (grouped_cost - baseline).detach()
    loss = (advantage * grouped_log_probability).mean()
    return loss, advantage.reshape(-1)


def _encode_batch(
    policy: CVRPPolicy, batch: CVRPBatch, prior: PriorProvider | None
) -> PolicyEncoding:
    m_hat = prior(batch) if prior is not None else None
    return policy.encode(
        batch.coords,
        batch.demands,
        batch.capacity,
        batch.depot_index,
        batch.node_mask,
        m_hat=m_hat,
        customer_node_indices=batch.customer_node_indices,
        customer_mask=batch.customer_mask,
    )


def _rollout_feasibility(
    batch: CVRPBatch,
    rollout: DecoderRollout,
    chunk: Sequence[CVRPExample],
    num_starts: int,
) -> float:
    """Fraction of decoded rollouts that pass the independent route validator."""
    routes = actions_to_routes(
        rollout.actions,
        batch.depot_index.repeat_interleave(num_starts),
        batch.customer_node_indices.repeat_interleave(num_starts, dim=0),
        batch.customer_mask.repeat_interleave(num_starts, dim=0),
    )
    feasible = 0
    for index, route_set in enumerate(routes):
        report = validate_routes(chunk[index // num_starts].instance, route_set)
        feasible += int(report.feasible)
    return feasible / max(len(routes), 1)


def _mean_fusion_gate(encoding: PolicyEncoding) -> float:
    """Average fusion gate over real nodes, or NaN when the encoder has no gate.

    A value drifting toward zero means the masked local branch is being ignored, which silently
    turns the run into the "no diffusion prior" encoder ablation.
    """
    gate = encoding.fusion_gate
    if gate is None:
        return float("nan")
    real_nodes = encoding.node_mask.to(dtype=torch.bool).sum()
    denominator = real_nodes * gate.shape[-1]
    if int(denominator) == 0:
        return float("nan")
    return float((gate.detach().sum() / denominator).item())


@torch.no_grad()
def evaluate_policy(
    policy: CVRPPolicy,
    examples: Sequence[CVRPExample],
    *,
    batch_size: int = 8,
    num_starts: int | None = None,
    start_node_cap: int = NSTART_CAP,
    device: torch.device | str | None = None,
    prior: PriorProvider | None = None,
    check_feasibility: bool = True,
) -> dict[str, float]:
    """Greedy multi-start evaluation: mean cost, best-of-starts cost, and gap to the labels.

    ``num_starts=None`` follows CMD Algorithm 1 ``NStart`` (every customer, or the closest
    ``start_node_cap`` when ``N`` is larger), resolved per batch from the instance size.
    """
    if not examples:
        raise ValueError("cannot evaluate on an empty list of examples")
    policy.eval()
    total_cost = 0.0
    total_best_cost = 0.0
    total_reference = 0.0
    total_instances = 0
    total_starts = 0.0
    feasible_sum = 0.0
    feasible_batches = 0

    for chunk in size_homogeneous_chunks(list(examples), batch_size, shuffle=False):
        batch = _collate(chunk, device)
        encoding = _encode_batch(policy, batch, prior)
        starts = (
            nstart_count(encoding.node_mask, cap=start_node_cap)
            if num_starts is None
            else num_starts
        )
        rollout = policy.rollout(encoding, decode_mode="greedy", num_starts=starts)
        grouped = rollout.cost.view(len(chunk), starts)
        total_cost += float(grouped.mean(dim=1).sum().item())
        total_best_cost += float(grouped.min(dim=1).values.sum().item())
        total_reference += sum(float(example.solution.cost) for example in chunk)
        total_instances += len(chunk)
        total_starts += float(starts * len(chunk))
        if check_feasibility:
            feasible_sum += _rollout_feasibility(batch, rollout, chunk, starts)
            feasible_batches += 1

    mean_cost = total_cost / total_instances
    best_cost = total_best_cost / total_instances
    reference_cost = total_reference / total_instances
    gap = 100.0 * (best_cost - reference_cost) / reference_cost if reference_cost > 0 else math.nan
    return {
        "cost": mean_cost,
        "best_cost": best_cost,
        "reference_cost": reference_cost,
        "gap_percent": gap,
        "reward": -mean_cost,
        "feasible_rate": (feasible_sum / feasible_batches if feasible_batches else float("nan")),
        "num_instances": float(total_instances),
        "num_starts": total_starts / total_instances,
    }


def _collate(chunk: Sequence[CVRPExample], device: torch.device | str | None) -> CVRPBatch:
    batch = collate_batch(list(chunk))
    return batch if device is None else batch_to_device(batch, device)


def save_policy_checkpoint(
    path: str | Path,
    *,
    policy: CVRPPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    row: dict[str, Any],
    best_metric_name: str,
    best_metric_value: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a resumable policy checkpoint (weights + optimizer + epoch metrics)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": dict(row),
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "rng_state": capture_rng_state(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, target)
    return target


def train_policy(
    policy: CVRPPolicy,
    train_examples: Sequence[CVRPExample],
    *,
    val_examples: Sequence[CVRPExample] | None = None,
    num_epochs: int,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    batch_size: int = 8,
    num_starts: int | None = None,
    val_num_starts: int | None = None,
    start_node_cap: int = NSTART_CAP,
    baseline_mode: BaselineMode = "multi_start",
    prior: PriorProvider | None = None,
    entropy_weight: float = 0.0,
    gradient_clip_norm: float | None = 1.0,
    seed: int = 0,
    device: torch.device | str | None = None,
    checkpoint_dir: str | Path | None = None,
    best_metric: str = "val_best_cost",
    minimize_best: bool = True,
    early_stop_patience: int = 0,
    on_epoch_end: EpochCallback | None = None,
    checkpoint_extra: dict[str, Any] | None = None,
    max_runtime_seconds: float | None = None,
    check_feasibility: bool = True,
) -> list[dict[str, Any]]:
    """Train ``policy`` with REINFORCE and a POMO-style shared baseline.

    Trajectories are sampled (not greedy) during training so the action space keeps being
    explored, matching CMD Algorithm 1. Validation always decodes greedily.

    ``num_starts=None`` uses CMD Algorithm 1 ``NStart``: every customer if ``N <= start_node_cap``,
    otherwise the ``start_node_cap`` customers closest to the depot. That makes the shared baseline
    an average over genuinely distinct route structures, at a memory cost linear in the count, so
    lower ``batch_size`` rather than ``num_starts`` when constrained.
    """
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if not train_examples:
        raise ValueError("cannot train on an empty list of examples")
    if entropy_weight < 0:
        raise ValueError(f"entropy_weight must be >= 0, got {entropy_weight}")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when set")
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive when set")

    resolved_val = val_examples if val_examples is not None else train_examples
    if not resolved_val:
        raise ValueError("cannot validate on an empty list of examples")
    if device is not None:
        policy.to(device)
    policy_device = next(policy.parameters()).device
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    eval_starts = val_num_starts if val_num_starts is not None else num_starts

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_value: float | None = None
    epochs_without_improve = 0
    training_started = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_started = time.perf_counter()
        policy.train()
        if policy.architecture == "paper_cmd":
            policy.verify_paper_global_gat_frozen()
        chunk_generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        rollout_generator = torch.Generator(device="cpu").manual_seed(seed + 7919 * (epoch + 1))
        epoch_loss = 0.0
        epoch_cost = 0.0
        epoch_best_cost = 0.0
        epoch_entropy = 0.0
        epoch_advantage_std = 0.0
        epoch_gradient_norm = 0.0
        epoch_feasible = 0.0
        epoch_starts = 0.0
        epoch_fusion_gate = 0.0
        num_batches = 0

        for chunk in size_homogeneous_chunks(
            list(train_examples), batch_size, generator=chunk_generator
        ):
            batch = _collate(chunk, policy_device)
            encoding = _encode_batch(policy, batch, prior)
            starts = (
                nstart_count(encoding.node_mask, cap=start_node_cap)
                if num_starts is None
                else num_starts
            )
            rollout = policy.rollout(
                encoding,
                decode_mode="sampling",
                num_starts=starts,
                generator=rollout_generator,
            )
            baseline_cost: Tensor | None = None
            if baseline_mode == "greedy_rollout":
                with torch.no_grad():
                    greedy = policy.rollout(encoding, decode_mode="greedy", num_starts=1)
                baseline_cost = greedy.cost.detach()

            loss, advantage = policy_gradient_loss(
                rollout.cost,
                rollout.log_probability,
                num_starts=starts,
                baseline_mode=baseline_mode,
                baseline_cost=baseline_cost,
            )
            if entropy_weight > 0:
                loss = loss - entropy_weight * rollout.entropy.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite policy loss at epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            if policy.architecture == "paper_cmd":
                policy.verify_paper_global_gat_frozen()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                gradient_clip_norm if gradient_clip_norm is not None else float("inf"),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"non-finite gradient norm at epoch {epoch}")
            optimizer.step()

            grouped_cost = rollout.cost.detach().view(len(chunk), starts)
            epoch_loss += float(loss.item())
            epoch_cost += float(grouped_cost.mean().item())
            epoch_best_cost += float(grouped_cost.min(dim=1).values.mean().item())
            epoch_entropy += float(rollout.entropy.detach().mean().item())
            epoch_advantage_std += float(advantage.std().item()) if advantage.numel() > 1 else 0.0
            epoch_gradient_norm += float(gradient_norm)
            epoch_starts += float(starts)
            epoch_fusion_gate += _mean_fusion_gate(encoding)
            if check_feasibility:
                epoch_feasible += _rollout_feasibility(batch, rollout, chunk, starts)
            num_batches += 1

        if num_batches == 0:
            raise ValueError("no training batches were produced; check batch_size and dataset")

        validation = evaluate_policy(
            policy,
            resolved_val,
            batch_size=batch_size,
            num_starts=eval_starts,
            start_node_cap=start_node_cap,
            device=policy_device,
            prior=prior,
            check_feasibility=check_feasibility,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": epoch_loss / num_batches,
            "train_cost": epoch_cost / num_batches,
            "train_best_cost": epoch_best_cost / num_batches,
            "train_reward": -(epoch_cost / num_batches),
            "train_entropy": epoch_entropy / num_batches,
            "train_advantage_std": epoch_advantage_std / num_batches,
            "train_feasibility_rate": (
                epoch_feasible / num_batches if check_feasibility else float("nan")
            ),
            "gradient_norm": epoch_gradient_norm / num_batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "num_starts": epoch_starts / num_batches,
            "fusion_gate_mean": epoch_fusion_gate / num_batches,
            "baseline_mode": baseline_mode,
            "val_num_starts": validation["num_starts"],
            "val_cost": validation["cost"],
            "val_best_cost": validation["best_cost"],
            "val_reward": validation["reward"],
            "val_gap_percent": validation["gap_percent"],
            "val_feasibility_rate": validation["feasible_rate"],
            "epoch_runtime_seconds": time.perf_counter() - epoch_started,
            "runtime_seconds": time.perf_counter() - training_started,
            "seed": seed,
        }

        if best_metric not in row:
            raise KeyError(f"best_metric {best_metric!r} is not one of {sorted(row)}")
        current = float(row[best_metric])
        improved = (
            best_value is None
            or (minimize_best and current < best_value)
            or (not minimize_best and current > best_value)
        )
        if improved:
            best_value = current
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        row["is_best"] = improved

        if ckpt_dir is not None:
            save_policy_checkpoint(
                ckpt_dir / "last.pt",
                policy=policy,
                optimizer=optimizer,
                epoch=epoch,
                row=row,
                best_metric_name=best_metric,
                best_metric_value=float(best_value if best_value is not None else current),
                extra=checkpoint_extra,
            )
            if improved:
                save_policy_checkpoint(
                    ckpt_dir / "best.pt",
                    policy=policy,
                    optimizer=optimizer,
                    epoch=epoch,
                    row=row,
                    best_metric_name=best_metric,
                    best_metric_value=float(current),
                    extra=checkpoint_extra,
                )

        logger.info(
            "epoch=%d train_loss=%.4f train_cost=%.4f val_cost=%.4f val_best_cost=%.4f "
            "val_gap=%.2f%% entropy=%.4f",
            epoch,
            row["train_loss"],
            row["train_cost"],
            row["val_cost"],
            row["val_best_cost"],
            row["val_gap_percent"],
            row["train_entropy"],
        )
        history.append(row)
        if on_epoch_end is not None:
            on_epoch_end(row)

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            logger.info(
                "early stopping after %d epochs without improvement", epochs_without_improve
            )
            break
        if max_runtime_seconds is not None:
            if time.perf_counter() - training_started >= max_runtime_seconds:
                logger.info("stopping: runtime budget of %.0fs reached", max_runtime_seconds)
                break

    return history


def build_policy_from_config(
    model_cfg: dict[str, Any],
    *,
    require_pretrained_global: bool = True,
) -> CVRPPolicy:
    """Instantiate :class:`CVRPPolicy` from the ``model`` block of a policy config."""
    architecture = str(model_cfg.get("architecture", "ours_robust"))
    if architecture not in {"ours_robust", "paper_cmd"}:
        raise ValueError(
            f"model.architecture must be 'ours_robust' or 'paper_cmd', got {architecture!r}"
        )
    global_checkpoint = model_cfg.get("global_gat_checkpoint")
    checkpoint_path: Path | None = None
    if global_checkpoint and require_pretrained_global:
        checkpoint_path = Path(str(global_checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = _ROOT / checkpoint_path
    if architecture == "paper_cmd" and checkpoint_path is None and require_pretrained_global:
        raise ValueError("paper_cmd requires model.global_gat_checkpoint")

    policy = CVRPPolicy(
        architecture=architecture,  # type: ignore[arg-type]
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        global_num_layers=int(model_cfg.get("global_num_layers", 5)),
        global_num_heads=int(model_cfg.get("global_num_heads", 8)),
        local_num_layers=int(model_cfg.get("local_num_layers", 5)),
        local_num_heads=int(model_cfg.get("local_num_heads", 8)),
        feed_forward_dim=int(model_cfg.get("feed_forward_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        decoder_num_heads=int(model_cfg.get("decoder_num_heads", 8)),
        clip_constant=float(model_cfg.get("clip_constant", 10.0)),
        adjacency_mode=str(model_cfg.get("adjacency_mode", "hard")),  # type: ignore[arg-type]
        local_threshold=float(model_cfg.get("local_threshold", 0.5)),
        use_local_encoder=bool(model_cfg.get("use_local_encoder", True)),
        use_local_pointer=bool(model_cfg.get("use_local_pointer", True)),
        use_global_pointer=bool(model_cfg.get("use_global_pointer", True)),
        use_savings_bias=bool(model_cfg.get("use_savings_bias", True)),
        use_context_perception=bool(model_cfg.get("use_context_perception", True)),
        savings_weight=float(model_cfg.get("savings_weight", 1.0)),
        savings_epsilon=float(model_cfg.get("savings_epsilon", 1e-2)),
        global_gat_checkpoint=checkpoint_path,
        allow_uninitialized_global_gat=not require_pretrained_global,
    )
    if architecture == "paper_cmd":
        if require_pretrained_global:
            require_artifact_alignment(
                policy.paper_global_gat_checkpoint_payload,
                expected_track="paper_cmd",
                artifact_name="model.global_gat_checkpoint",
            )
        policy.verify_paper_global_gat_frozen()
    return policy


def _build_prior(config: dict[str, Any], device: torch.device) -> PriorProvider | None:
    """Resolve the ``prior`` config block into a provider of ``M_hat``."""
    prior_cfg = config.get("prior") or {}
    source = str(prior_cfg.get("source", "denoiser"))
    if source == "none":
        return None
    if source == "true_matrix":
        return constraint_matrix_prior
    if source != "denoiser":
        raise ValueError(f"prior.source must be denoiser/true_matrix/none, got {source!r}")

    checkpoint = prior_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("prior.source='denoiser' requires prior.checkpoint")
    ckpt_path = Path(str(checkpoint))
    if not ckpt_path.is_absolute():
        ckpt_path = _ROOT / ckpt_path
    denoiser, payload = load_denoiser_checkpoint(ckpt_path, device=device)
    alignment_track = str((config.get("alignment") or {}).get("track", "legacy_unspecified"))
    if alignment_track == "paper_cmd":
        require_artifact_alignment(
            payload,
            expected_track="paper_cmd",
            artifact_name="prior.checkpoint",
        )
    schedule_cfg = (payload.get("extra") or {}).get("schedule") or {}
    schedule = BernoulliDiffusionSchedule(
        num_timesteps=int(schedule_cfg.get("num_timesteps", 700)),
        beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
        beta_end=float(schedule_cfg.get("beta_end", 2e-2)),
    ).to(device)
    return denoiser_prior(
        denoiser,
        schedule,
        num_inference_steps=(
            int(prior_cfg["num_inference_steps"]) if prior_cfg.get("num_inference_steps") else None
        ),
        step_stride=int(prior_cfg.get("step_stride", 1)),
        threshold=float(prior_cfg.get("threshold", 0.5)),
        seed=int(config["seed"]),
        use_probabilities=bool(prior_cfg.get("use_probabilities", True)),
        sampler=str(prior_cfg.get("sampler", "one_step_approx")),  # type: ignore[arg-type]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CMD dual-pointer policy with REINFORCE + POMO baseline."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    """CLI entry: config → policy → :func:`train_policy` (+ MLflow)."""
    args = _parse_args()
    cfg_path = args.config if args.config.is_absolute() else _ROOT / args.config
    config = yaml.safe_load(cfg_path.read_text())
    alignment = validate_alignment_config(config, component="policy")
    config["alignment"] = alignment.as_dict()
    seed = int(config["seed"])
    train_cfg = config["training"]
    config["reproducibility"] = seed_everything(
        seed, deterministic=bool(train_cfg.get("deterministic", False))
    )

    dataset_path_value = config["dataset"].get("path")
    if not dataset_path_value:
        raise ValueError("dataset.path must be set to a materialized CVRP split")
    dataset_path = _ROOT / dataset_path_value
    train_examples: Sequence[CVRPExample] = IndexedJSONDataset(
        dataset_path, cache_size=int(config["dataset"].get("cache_size", 0))
    )
    if not train_examples:
        raise ValueError(f"no examples found under {dataset_path}")

    val_cfg = config.get("validation") or {}
    val_path = val_cfg.get("path")
    val_examples: Sequence[CVRPExample]
    if val_path:
        val_examples = IndexedJSONDataset(
            _ROOT / val_path, cache_size=int(val_cfg.get("cache_size", 0))
        )
        if not val_examples:
            raise ValueError(f"no validation examples found under {_ROOT / val_path}")
    else:
        val_examples = train_examples

    device = resolve_device(train_cfg.get("device", "auto"))
    policy = build_policy_from_config(config["model"]).to(device)
    prior = _build_prior(config, device)
    if policy.use_local_encoder and prior is None:
        raise ValueError(
            "model.use_local_encoder=true needs a prior; set prior.source or disable the "
            "local encoder for the no-diffusion ablation"
        )

    mlflow_cfg = config.get("mlflow") or {}
    mlflow = None
    if bool(mlflow_cfg.get("enabled", True)):
        import mlflow as mlflow_mod

        mlflow = mlflow_mod
        tracking_uri = str(mlflow_cfg.get("tracking_uri") or default_mlflow_tracking_uri())
        if tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(str(mlflow_cfg.get("experiment_name") or config["experiment_name"]))

    with ExperimentTracker(
        output_root=_ROOT / config["output"]["root"],
        experiment_name=config["experiment_name"],
        config=config,
        seed=seed,
        dataset_path=dataset_path,
    ) as tracker:
        run_context = (
            mlflow.start_run(run_name=tracker.run_dir.name) if mlflow is not None else None
        )
        try:
            skip_keys = {"is_best", "baseline_mode"}

            def on_epoch_end(row: dict[str, Any]) -> None:
                tracker.log_metric_row(row)
                if mlflow is None:
                    return
                step = int(row["epoch"])
                for key, value in row.items():
                    if key in skip_keys or isinstance(value, str):
                        continue
                    if isinstance(value, (int, float, bool)) and value == value:
                        mlflow.log_metric(key, float(value), step=step)

            history = train_policy(
                policy,
                train_examples,
                val_examples=val_examples,
                num_epochs=int(train_cfg["epochs"]),
                learning_rate=float(train_cfg["learning_rate"]),
                weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
                batch_size=int(train_cfg.get("batch_size", 8)),
                num_starts=(int(train_cfg["num_starts"]) if train_cfg.get("num_starts") else None),
                val_num_starts=(
                    int(train_cfg["val_num_starts"]) if train_cfg.get("val_num_starts") else None
                ),
                start_node_cap=int(train_cfg.get("start_node_cap", NSTART_CAP)),
                baseline_mode=str(train_cfg.get("baseline", "multi_start")),  # type: ignore[arg-type]
                prior=prior,
                entropy_weight=float(train_cfg.get("entropy_weight", 0.0)),
                gradient_clip_norm=float(train_cfg.get("gradient_clip_norm", 1.0)),
                seed=seed,
                device=device,
                checkpoint_dir=tracker.run_dir / "checkpoints",
                best_metric=str(config.get("checkpoint", {}).get("best_metric", "val_best_cost")),
                minimize_best=bool(config.get("checkpoint", {}).get("minimize", True)),
                early_stop_patience=int(train_cfg.get("early_stop_patience", 0) or 0),
                on_epoch_end=on_epoch_end,
                checkpoint_extra={
                    "experiment_name": config["experiment_name"],
                    "seed": seed,
                    "alignment": config["alignment"],
                    "model": config["model"],
                    "prior": config.get("prior"),
                },
                max_runtime_seconds=(
                    float(train_cfg["max_runtime_seconds"])
                    if train_cfg.get("max_runtime_seconds")
                    else None
                ),
                check_feasibility=bool(train_cfg.get("check_feasibility", True)),
            )
            if history:
                tracker.log_metrics({f"final_{k}": v for k, v in history[-1].items()})
        finally:
            if run_context is not None:
                run_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
