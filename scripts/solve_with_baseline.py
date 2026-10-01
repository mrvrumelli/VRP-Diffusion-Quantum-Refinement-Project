"""Solve instances end to end with the frozen per-size baseline (prior + policy).

For every instance: regenerate the diffusion prior with the per-size denoiser
(``posterior_mixture_v2``, graph-ID-derived seed), decode the per-size policy greedily from
``--starts`` start nodes, keep the best start's routes, and record cost, feasibility, the gap to
the reference, and separate prior and policy timings (CUDA-synchronised, model loading excluded).
The prior probability matrix is saved per instance for refinement experiments.

``--once`` refuses to run if the output directory already holds results; use it for the reserved
test set, which must be scored exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vrp_diffusion_quantum.data.dataset import collate_batch, load_example
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.inference.policy_support import batch_to_device, denoiser_prior
from vrp_diffusion_quantum.inference.predict_matrix import load_denoiser_checkpoint
from vrp_diffusion_quantum.inference.solve_instance import load_policy_checkpoint
from vrp_diffusion_quantum.models.decoder import CVRPPolicy, actions_to_routes
from vrp_diffusion_quantum.models.diffusion import schedule_from_config
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

LOGGER = logging.getLogger("solve_with_baseline")
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _graph_seed(seed: int, instance_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{instance_id}".encode()).digest()[:4], "big")


def _solve_one(
    example: CVRPExample,
    policy: CVRPPolicy,
    prior_model: tuple[Any, Any],
    *,
    starts: int,
    steps: int,
    seed: int,
) -> tuple[list[list[int]], float, bool, np.ndarray, float, float]:
    denoiser, schedule = prior_model
    instance_id = example.instance.instance_id
    batch = batch_to_device(collate_batch([example]), "cuda")
    provider = denoiser_prior(
        denoiser,
        schedule,
        num_inference_steps=steps,
        seed=_graph_seed(seed, instance_id),
        sampler="posterior_mixture_v2",
        use_probabilities=True,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        m_prob = provider(batch)
        torch.cuda.synchronize()
        prior_seconds = time.perf_counter() - started
        started = time.perf_counter()
        encoding = policy.encode(
            batch.coords,
            batch.demands,
            batch.capacity,
            batch.depot_index,
            batch.node_mask,
            m_hat=m_prob,
            customer_node_indices=batch.customer_node_indices,
            customer_mask=batch.customer_mask,
        )
        rollout = policy.rollout(encoding, decode_mode="greedy", num_starts=starts)
        torch.cuda.synchronize()
        policy_seconds = time.perf_counter() - started
    costs = rollout.cost.view(starts)
    best = int(torch.argmin(costs).item())
    all_routes = actions_to_routes(
        rollout.actions,
        batch.depot_index.repeat_interleave(starts),
        batch.customer_node_indices.repeat_interleave(starts, dim=0),
        batch.customer_mask.repeat_interleave(starts, dim=0),
    )
    routes = all_routes[best]
    feasible = validate_routes(example.instance, routes).feasible
    cost = route_cost(example.instance, routes)
    n = example.instance.n_customers
    matrix = m_prob[0, :n, :n].detach().float().cpu().numpy()
    return routes, cost, feasible, matrix, prior_seconds, policy_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True, help="manifest with 'examples'")
    parser.add_argument("--assignment", type=Path, required=True, help="size -> policy/prior")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="optional strengthened references")
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--once", action="store_true", help="refuse to run twice")
    parser.add_argument("--warmup", type=int, default=2, help="untimed warm-up solves per size")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.once and output.exists() and any(output.glob("*.json")):
        raise SystemExit(f"{output} already holds results; --once forbids a second scoring")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / args.examples).read_text())
    entries = manifest["examples"]
    assignment = json.loads((ROOT / args.assignment).read_text())
    references: dict[str, float] = {}
    if args.references:
        refs = json.loads((ROOT / args.references).read_text())
        references = {
            row["instance_id"]: float(row["reference_cost"]) for row in refs["references"]
        }

    models: dict[int, tuple[CVRPPolicy, tuple[Any, Any]]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for size_key, spec in assignment.items():
        size = int(size_key)
        policy_path, prior_path = ROOT / spec["policy"], ROOT / spec["prior"]
        policy, _ = load_policy_checkpoint(policy_path, device="cuda")
        policy.eval()
        denoiser, payload = load_denoiser_checkpoint(prior_path, device="cuda")
        schedule = schedule_from_config((payload.get("extra") or {}).get("schedule")).to("cuda")
        models[size] = (policy, (denoiser, schedule))
        artifacts[size_key] = {
            "policy": spec["policy"],
            "policy_sha256": _sha256(policy_path),
            "prior": spec["prior"],
            "prior_sha256": _sha256(prior_path),
        }

    for size, (policy, prior_model) in models.items():
        warm = [e for e in entries if int(e.get("n_customers", e.get("size", 0))) == size]
        for entry in warm[: args.warmup]:
            _solve_one(
                load_example(ROOT / entry["file"]),
                policy,
                prior_model,
                starts=args.starts,
                steps=args.steps,
                seed=args.seed + 999,
            )

    rows = []
    for entry in entries:
        example = load_example(ROOT / entry["file"])
        size = example.instance.n_customers
        policy, prior_model = models[size]
        routes, cost, feasible, matrix, prior_s, policy_s = _solve_one(
            example, policy, prior_model, starts=args.starts, steps=args.steps, seed=args.seed
        )
        instance_id = example.instance.instance_id
        label_cost = route_cost(example.instance, example.solution.routes)
        reference = references.get(instance_id, label_cost)
        row = {
            "instance_id": instance_id,
            "file": entry["file"],
            "n_customers": size,
            "routes": routes,
            "cost": cost,
            "feasible": feasible,
            "reference_cost": reference,
            "label_cost": label_cost,
            "gap_percent": 100.0 * (cost - reference) / reference,
            "prior_seconds": prior_s,
            "policy_seconds": policy_s,
            "total_seconds": prior_s + policy_s,
        }
        rows.append(row)
        (output / f"{instance_id}.json").write_text(json.dumps(row) + "\n")
        np.savez_compressed(output / f"{instance_id}.m_prob.npz", m_prob=matrix.astype(np.float32))
        LOGGER.info(
            "%s gap=%.2f%% time=%.3fs", instance_id, row["gap_percent"], row["total_seconds"]
        )

    summary: dict[str, Any] = {
        "examples": str(args.examples),
        "assignment": artifacts,
        "starts": args.starts,
        "steps": args.steps,
        "seed": args.seed,
        "sampler": "posterior_mixture_v2",
        "references": str(args.references) if args.references else "instance labels",
        "by_size": {},
    }
    for size in sorted({row["n_customers"] for row in rows}):
        sel = [row for row in rows if row["n_customers"] == size]
        summary["by_size"][str(size)] = {
            "instances": len(sel),
            "feasible": sum(row["feasible"] for row in sel),
            "mean_gap_percent": float(np.mean([row["gap_percent"] for row in sel])),
            "mean_prior_seconds": float(np.mean([row["prior_seconds"] for row in sel])),
            "mean_policy_seconds": float(np.mean([row["policy_seconds"] for row in sel])),
            "mean_total_seconds": float(np.mean([row["total_seconds"] for row in sel])),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["by_size"], indent=1))


if __name__ == "__main__":
    main()
