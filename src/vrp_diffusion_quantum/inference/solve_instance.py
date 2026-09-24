"""Solve CVRP instances with a trained dual-pointer policy (CMD Algorithm 2, P4.7).

Four decoding controls compose into one best-of-k search:

* ``decode_mode="greedy"`` takes the arg-max action, ``"sampling"`` draws from the policy;
* ``num_starts`` decodes in parallel from several start customers (CMD ``NStart``);
* ``num_samples`` repeats a stochastic rollout, which only helps in sampling mode;
* ``num_augmentations`` replays the instance under distance-preserving geometric symmetries.

Every candidate is re-validated and re-costed against the *original* instance, so the reported
route set is feasible by independent check rather than by construction alone.

Usage::

    python -m vrp_diffusion_quantum.inference.solve_instance \\
      --checkpoint outputs/policy/<run>/checkpoints/best.pt --data-dir data/processed/test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from vrp_diffusion_quantum.data.augment import AUGMENT_NUM
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance
from vrp_diffusion_quantum.eval.confidence import bootstrap_mean_interval
from vrp_diffusion_quantum.inference.policy_support import (
    PriorProvider,
    augment_instance,
    batch_to_device,
    collate_instances,
)
from vrp_diffusion_quantum.models.decoder import (
    NSTART_CAP,
    CVRPPolicy,
    DecodeMode,
    actions_to_routes,
    nstart_count,
)
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]

__all__ = [
    "InstanceSolution",
    "load_policy_checkpoint",
    "main",
    "solve_instance",
    "solve_instances",
    "summarize_solutions",
]


@dataclass(frozen=True)
class InstanceSolution:
    """Best feasible route set found for one instance, plus the search settings that found it."""

    instance_id: str
    n_customers: int
    routes: list[list[int]]
    cost: float
    feasible: bool
    num_vehicles: int
    method: str
    num_starts: int
    num_samples: int
    num_augmentations: int
    num_candidates: int
    best_augmentation: int
    best_start: int
    runtime_seconds: float
    seed: int
    inference_batch_size: int = 1
    reference_cost: float | None = None
    gap_to_reference: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_policy_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[CVRPPolicy, dict[str, Any]]:
    """Rebuild a :class:`CVRPPolicy` from a training checkpoint; return ``(policy, payload)``."""
    from vrp_diffusion_quantum.train.train_policy import build_policy_from_config

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = (payload.get("extra") or {}).get("model") or {}
    if str(model_cfg.get("architecture", "ours_robust")) == "paper_cmd":
        from vrp_diffusion_quantum.utils.alignment import require_artifact_alignment

        require_artifact_alignment(
            payload,
            expected_track="paper_cmd",
            artifact_name="policy checkpoint",
        )
    # The policy checkpoint itself contains the frozen GAT weights, so loading remains
    # self-contained even if the original pretraining checkpoint has moved.
    policy = build_policy_from_config(model_cfg, require_pretrained_global=False)
    policy.load_state_dict(payload["model"])
    if policy.architecture == "paper_cmd":
        policy.verify_paper_global_gat_frozen()
    policy.eval()
    if device is not None:
        policy.to(device)
    return policy, payload


def _method_name(decode_mode: DecodeMode, num_starts: int, num_augmentations: int) -> str:
    parts = [f"policy_{decode_mode}"]
    if num_starts > 1:
        parts.append(f"start{num_starts}")
    if num_augmentations > 1:
        parts.append(f"aug{num_augmentations}")
    return "_".join(parts)


@torch.no_grad()
def solve_instance(
    policy: CVRPPolicy,
    instance: CVRPInstance,
    *,
    prior: PriorProvider | None = None,
    decode_mode: DecodeMode = "greedy",
    num_starts: int | None = None,
    start_node_cap: int = NSTART_CAP,
    num_samples: int = 1,
    num_augmentations: int = 1,
    seed: int = 0,
    device: torch.device | str | None = None,
    reference_cost: float | None = None,
) -> InstanceSolution:
    """Return the cheapest feasible route set the policy finds under the given search budget.

    ``num_starts=None`` applies the paper's ``NStart`` rule (CMD Algorithm 2 line 2).
    """
    if num_starts is not None and num_starts < 1:
        raise ValueError(f"num_starts must be >= 1, got {num_starts}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if not 1 <= num_augmentations <= AUGMENT_NUM:
        raise ValueError(f"num_augmentations must be in 1..{AUGMENT_NUM}, got {num_augmentations}")
    if decode_mode == "greedy" and num_samples > 1:
        raise ValueError("greedy decoding is deterministic; num_samples > 1 needs sampling mode")

    started = time.perf_counter()
    variants = [augment_instance(instance, variant) for variant in range(num_augmentations)]
    batch = collate_instances(variants)
    if device is not None:
        batch = batch_to_device(batch, device)

    m_hat = prior(batch) if prior is not None else None
    encoding = policy.encode(
        batch.coords,
        batch.demands,
        batch.capacity,
        batch.depot_index,
        batch.node_mask,
        m_hat=m_hat,
        customer_node_indices=batch.customer_node_indices,
        customer_mask=batch.customer_mask,
    )

    if num_starts is None:
        num_starts = nstart_count(encoding.node_mask, cap=start_node_cap)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates: list[tuple[float, int, int, list[list[int]]]] = []
    for _ in range(num_samples):
        rollout = policy.rollout(
            encoding,
            decode_mode=decode_mode,
            num_starts=num_starts,
            generator=generator,
        )
        routes = actions_to_routes(
            rollout.actions,
            batch.depot_index.repeat_interleave(num_starts),
            batch.customer_node_indices.repeat_interleave(num_starts, dim=0),
            batch.customer_mask.repeat_interleave(num_starts, dim=0),
        )
        costs = rollout.cost.detach().cpu().tolist()
        for index, route_set in enumerate(routes):
            candidates.append(
                (float(costs[index]), index // num_starts, index % num_starts, route_set)
            )

    # Geometric symmetries preserve distances, so a variant's rollout cost ranks candidates for
    # the original instance; the winner is still re-costed and re-validated on the original.
    candidates.sort(key=lambda candidate: candidate[0])
    best: tuple[float, int, int, list[list[int]]] | None = None
    for candidate in candidates:
        if validate_routes(instance, candidate[3]).feasible:
            best = candidate
            break

    elapsed = time.perf_counter() - started
    method = _method_name(decode_mode, num_starts, num_augmentations)
    if best is None:
        logger.warning(
            "no feasible route set for %s across %d candidates",
            instance.instance_id,
            len(candidates),
        )
        return InstanceSolution(
            instance_id=instance.instance_id,
            n_customers=instance.n_customers,
            routes=[],
            cost=math.inf,
            feasible=False,
            num_vehicles=0,
            method=method,
            num_starts=num_starts,
            num_samples=num_samples,
            num_augmentations=num_augmentations,
            num_candidates=len(candidates),
            best_augmentation=-1,
            best_start=-1,
            runtime_seconds=elapsed,
            seed=seed,
            reference_cost=reference_cost,
            gap_to_reference=None,
        )

    _, augmentation, start, best_routes = best
    cost = route_cost(instance, best_routes)
    gap = None
    if reference_cost is not None and reference_cost > 0:
        gap = 100.0 * (cost - reference_cost) / reference_cost
    return InstanceSolution(
        instance_id=instance.instance_id,
        n_customers=instance.n_customers,
        routes=best_routes,
        cost=cost,
        feasible=True,
        num_vehicles=len(best_routes),
        method=method,
        num_starts=num_starts,
        num_samples=num_samples,
        num_augmentations=num_augmentations,
        num_candidates=len(candidates),
        best_augmentation=augmentation,
        best_start=start,
        runtime_seconds=elapsed,
        seed=seed,
        reference_cost=reference_cost,
        gap_to_reference=gap,
    )


def solve_instances(
    policy: CVRPPolicy,
    examples: Sequence[CVRPExample | CVRPInstance],
    *,
    prior: PriorProvider | None = None,
    decode_mode: DecodeMode = "greedy",
    num_starts: int | None = None,
    start_node_cap: int = NSTART_CAP,
    num_samples: int = 1,
    num_augmentations: int = 1,
    seed: int = 0,
    device: torch.device | str | None = None,
    batch_size: int = 1,
) -> list[InstanceSolution]:
    """Solve instances, optionally batching same-size greedy rollouts.

    Batch size one preserves the historical per-instance execution and stochastic seed stream.
    Larger batches are restricted to one greedy sample because independent per-instance sampling
    generators cannot be represented by the decoder's current batched generator interface.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_starts is not None and num_starts < 1:
        raise ValueError(f"num_starts must be >= 1, got {num_starts}")
    if not 1 <= num_augmentations <= AUGMENT_NUM:
        raise ValueError(f"num_augmentations must be in 1..{AUGMENT_NUM}, got {num_augmentations}")
    if batch_size > 1 and (decode_mode != "greedy" or num_samples != 1):
        raise ValueError("batch_size > 1 currently requires greedy decoding with num_samples=1")
    if batch_size == 1:
        return _solve_instances_sequential(
            policy,
            examples,
            prior=prior,
            decode_mode=decode_mode,
            num_starts=num_starts,
            start_node_cap=start_node_cap,
            num_samples=num_samples,
            num_augmentations=num_augmentations,
            seed=seed,
            device=device,
        )

    normalized: list[tuple[int, CVRPInstance, float | None]] = []
    for index, item in enumerate(examples):
        if isinstance(item, CVRPExample):
            normalized.append((index, item.instance, float(item.solution.cost)))
        else:
            normalized.append((index, item, None))
    indexed_by_size: dict[int, list[tuple[int, CVRPInstance, float | None]]] = {}
    for normalized_item in normalized:
        indexed_by_size.setdefault(normalized_item[1].n_customers, []).append(normalized_item)

    output: list[InstanceSolution | None] = [None] * len(normalized)
    for same_size in indexed_by_size.values():
        for start in range(0, len(same_size), batch_size):
            chunk = same_size[start : start + batch_size]
            started = time.perf_counter()
            variants = [
                augment_instance(instance, variant)
                for _, instance, _ in chunk
                for variant in range(num_augmentations)
            ]
            batch = collate_instances(variants)
            if device is not None:
                batch = batch_to_device(batch, device)
            m_hat = prior(batch) if prior is not None else None
            encoding = policy.encode(
                batch.coords,
                batch.demands,
                batch.capacity,
                batch.depot_index,
                batch.node_mask,
                m_hat=m_hat,
                customer_node_indices=batch.customer_node_indices,
                customer_mask=batch.customer_mask,
            )
            resolved_starts = (
                nstart_count(encoding.node_mask, cap=start_node_cap)
                if num_starts is None
                else num_starts
            )
            rollout = policy.rollout(
                encoding,
                decode_mode="greedy",
                num_starts=resolved_starts,
            )
            route_sets = actions_to_routes(
                rollout.actions,
                batch.depot_index.repeat_interleave(resolved_starts),
                batch.customer_node_indices.repeat_interleave(resolved_starts, dim=0),
                batch.customer_mask.repeat_interleave(resolved_starts, dim=0),
            )
            costs = rollout.cost.detach().cpu().tolist()
            elapsed_per_instance = (time.perf_counter() - started) / len(chunk)
            for chunk_index, (original_index, instance, reference) in enumerate(chunk):
                candidates: list[tuple[float, int, int, list[list[int]]]] = []
                first_variant = chunk_index * num_augmentations
                for augmentation in range(num_augmentations):
                    variant_index = first_variant + augmentation
                    for start_index in range(resolved_starts):
                        result_index = variant_index * resolved_starts + start_index
                        candidates.append(
                            (
                                float(costs[result_index]),
                                augmentation,
                                start_index,
                                route_sets[result_index],
                            )
                        )
                output[original_index] = _solution_from_candidates(
                    instance,
                    candidates,
                    decode_mode="greedy",
                    num_starts=resolved_starts,
                    num_samples=1,
                    num_augmentations=num_augmentations,
                    runtime_seconds=elapsed_per_instance,
                    seed=seed + original_index,
                    reference_cost=reference,
                    inference_batch_size=len(chunk),
                )
    if any(solution is None for solution in output):
        raise RuntimeError("internal error: batched policy inference did not produce every result")
    return [solution for solution in output if solution is not None]


def _solve_instances_sequential(
    policy: CVRPPolicy,
    examples: Sequence[CVRPExample | CVRPInstance],
    *,
    prior: PriorProvider | None,
    decode_mode: DecodeMode,
    num_starts: int | None,
    start_node_cap: int,
    num_samples: int,
    num_augmentations: int,
    seed: int,
    device: torch.device | str | None,
) -> list[InstanceSolution]:
    solutions: list[InstanceSolution] = []
    for index, item in enumerate(examples):
        if isinstance(item, CVRPExample):
            instance = item.instance
            reference: float | None = float(item.solution.cost)
        else:
            instance = item
            reference = None
        solutions.append(
            solve_instance(
                policy,
                instance,
                prior=prior,
                decode_mode=decode_mode,
                num_starts=num_starts,
                start_node_cap=start_node_cap,
                num_samples=num_samples,
                num_augmentations=num_augmentations,
                seed=seed + index,
                device=device,
                reference_cost=reference,
            )
        )
    return solutions


def _solution_from_candidates(
    instance: CVRPInstance,
    candidates: list[tuple[float, int, int, list[list[int]]]],
    *,
    decode_mode: DecodeMode,
    num_starts: int,
    num_samples: int,
    num_augmentations: int,
    runtime_seconds: float,
    seed: int,
    reference_cost: float | None,
    inference_batch_size: int,
) -> InstanceSolution:
    """Validate ranked candidates and build one result for batched policy inference."""
    candidates.sort(key=lambda candidate: candidate[0])
    best = next(
        (candidate for candidate in candidates if validate_routes(instance, candidate[3]).feasible),
        None,
    )
    method = _method_name(decode_mode, num_starts, num_augmentations)
    if best is None:
        return InstanceSolution(
            instance_id=instance.instance_id,
            n_customers=instance.n_customers,
            routes=[],
            cost=math.inf,
            feasible=False,
            num_vehicles=0,
            method=method,
            num_starts=num_starts,
            num_samples=num_samples,
            num_augmentations=num_augmentations,
            num_candidates=len(candidates),
            best_augmentation=-1,
            best_start=-1,
            runtime_seconds=runtime_seconds,
            seed=seed,
            inference_batch_size=inference_batch_size,
            reference_cost=reference_cost,
        )
    _, augmentation, start, routes = best
    cost = route_cost(instance, routes)
    gap = (
        100.0 * (cost - reference_cost) / reference_cost
        if reference_cost is not None and reference_cost > 0
        else None
    )
    return InstanceSolution(
        instance_id=instance.instance_id,
        n_customers=instance.n_customers,
        routes=routes,
        cost=cost,
        feasible=True,
        num_vehicles=len(routes),
        method=method,
        num_starts=num_starts,
        num_samples=num_samples,
        num_augmentations=num_augmentations,
        num_candidates=len(candidates),
        best_augmentation=augmentation,
        best_start=start,
        runtime_seconds=runtime_seconds,
        seed=seed,
        inference_batch_size=inference_batch_size,
        reference_cost=reference_cost,
        gap_to_reference=gap,
    )


def summarize_solutions(
    solutions: Sequence[InstanceSolution],
    *,
    confidence_level: float | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, float]:
    """Aggregate solved instances without hiding infeasible ones."""
    if not solutions:
        raise ValueError("cannot summarize an empty solution list")
    feasible = [solution for solution in solutions if solution.feasible]
    with_gap = [
        solution
        for solution in feasible
        if solution.gap_to_reference is not None and math.isfinite(solution.gap_to_reference)
    ]
    summary = {
        "num_instances": float(len(solutions)),
        "feasible_count": float(len(feasible)),
        "feasibility_rate": len(feasible) / len(solutions),
        "mean_cost": (
            sum(solution.cost for solution in feasible) / len(feasible) if feasible else math.nan
        ),
        "mean_gap_to_reference": (
            sum(float(s.gap_to_reference or 0.0) for s in with_gap) / len(with_gap)
            if with_gap
            else math.nan
        ),
        "mean_num_vehicles": (
            sum(solution.num_vehicles for solution in feasible) / len(feasible)
            if feasible
            else math.nan
        ),
        "mean_runtime_seconds": sum(s.runtime_seconds for s in solutions) / len(solutions),
        "instances_per_second": len(solutions)
        / max(sum(s.runtime_seconds for s in solutions), 1e-12),
        "mean_inference_batch_size": sum(s.inference_batch_size for s in solutions)
        / len(solutions),
        "mean_num_candidates": sum(s.num_candidates for s in solutions) / len(solutions),
    }
    if confidence_level is not None and with_gap:
        interval = bootstrap_mean_interval(
            [float(solution.gap_to_reference or 0.0) for solution in with_gap],
            confidence_level=confidence_level,
            num_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        summary.update(
            {
                "mean_gap_to_reference_ci_lower": interval.lower,
                "mean_gap_to_reference_ci_upper": interval.upper,
                "mean_gap_to_reference_ci_confidence_level": interval.confidence_level,
                "mean_gap_to_reference_ci_num_resamples": float(interval.num_resamples),
                "mean_gap_to_reference_ci_num_observations": float(interval.num_observations),
                "mean_gap_to_reference_ci_seed": float(interval.seed),
            }
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve CVRP instances with a trained policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--per-size", type=int, default=8)
    parser.add_argument("--decode-mode", choices=["greedy", "sampling"], default="greedy")
    parser.add_argument(
        "--num-starts",
        type=int,
        default=None,
        help="start nodes per instance; omit to use the paper's NStart rule",
    )
    parser.add_argument("--start-node-cap", type=int, default=NSTART_CAP)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-augmentations", type=int, default=AUGMENT_NUM)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--denoiser-checkpoint", type=Path, default=None)
    parser.add_argument("--prior-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI: best-of-k policy decoding over a test split, writing metrics and per-instance rows."""
    from vrp_diffusion_quantum.data.dataset import load_examples_by_size
    from vrp_diffusion_quantum.inference.policy_support import denoiser_prior
    from vrp_diffusion_quantum.inference.predict_matrix import (
        load_denoiser_checkpoint,
        select_examples_by_size,
    )
    from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
    from vrp_diffusion_quantum.utils.experiment import ExperimentTracker
    from vrp_diffusion_quantum.utils.runtime import resolve_device, seed_everything

    args = _parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else _ROOT / args.checkpoint
    data_dir = args.data_dir if args.data_dir.is_absolute() else _ROOT / args.data_dir
    if not checkpoint.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {checkpoint}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    policy, _ = load_policy_checkpoint(checkpoint, device=device)
    prior: PriorProvider | None = None
    if args.denoiser_checkpoint is not None:
        denoiser_path = (
            args.denoiser_checkpoint
            if args.denoiser_checkpoint.is_absolute()
            else _ROOT / args.denoiser_checkpoint
        )
        denoiser, payload = load_denoiser_checkpoint(denoiser_path, device=device)
        schedule_cfg = (payload.get("extra") or {}).get("schedule") or {}
        schedule = BernoulliDiffusionSchedule(
            num_timesteps=int(schedule_cfg.get("num_timesteps", 700)),
            beta_start=float(schedule_cfg.get("beta_start", 1e-4)),
            beta_end=float(schedule_cfg.get("beta_end", 2e-2)),
        ).to(device)
        prior = denoiser_prior(
            denoiser,
            schedule,
            num_inference_steps=args.prior_inference_steps,
            seed=args.seed,
        )
    if policy.use_local_encoder and prior is None:
        raise ValueError("this policy uses the local encoder; pass --denoiser-checkpoint")

    pool = load_examples_by_size(data_dir, list(args.sizes))
    if not pool:
        raise ValueError(f"no examples for sizes {args.sizes} under {data_dir}")
    examples = select_examples_by_size(
        pool, sizes=args.sizes, per_size=args.per_size, seed=args.seed
    )

    config = {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "sizes": list(args.sizes),
        "per_size": args.per_size,
        "decode_mode": args.decode_mode,
        "num_starts": args.num_starts,
        "start_node_cap": args.start_node_cap,
        "num_samples": args.num_samples,
        "num_augmentations": args.num_augmentations,
        "batch_size": args.batch_size,
        "confidence_level": args.confidence_level,
        "bootstrap_resamples": args.bootstrap_resamples,
        "denoiser_checkpoint": str(args.denoiser_checkpoint or ""),
        "device": str(device),
        "seed": args.seed,
    }
    output_root = args.output_dir or (_ROOT / "outputs" / "inference")
    with ExperimentTracker(
        output_root=output_root if output_root.is_absolute() else _ROOT / output_root,
        experiment_name="solve_instance",
        config=config,
        seed=args.seed,
        dataset_path=data_dir,
    ) as tracker:
        solutions = solve_instances(
            policy,
            examples,
            prior=prior,
            decode_mode=args.decode_mode,
            num_starts=args.num_starts,
            start_node_cap=args.start_node_cap,
            num_samples=args.num_samples,
            num_augmentations=args.num_augmentations,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
        )
        for solution in solutions:
            row = solution.to_dict()
            row.pop("routes")
            row["config_name"] = "solve_instance"
            tracker.log_metric_row(row)
        summary = summarize_solutions(
            solutions,
            confidence_level=args.confidence_level,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.seed,
        )
        tracker.log_metrics(summary)
        routes_path = tracker.run_dir / "routes.json"
        routes_path.write_text(
            json.dumps(
                {solution.instance_id: solution.routes for solution in solutions},
                indent=2,
            )
        )
        print(yaml.safe_dump(summary, sort_keys=True))
        print(f"wrote {tracker.run_dir}")


if __name__ == "__main__":
    main()
