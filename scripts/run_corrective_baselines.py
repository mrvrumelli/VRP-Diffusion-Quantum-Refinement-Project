"""Identical-panel OR baselines and matched-prior policy comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from run_corrective_evaluation import dataset_ids, sha256
from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.data.generate_cvrp import CVRPInstance as SolverInstance
from vrp_diffusion_quantum.data.solve_cvrp import solve_instance
from vrp_diffusion_quantum.eval.comparison import (
    InstanceResult,
    assert_disjoint_sources,
    paired_bootstrap,
)
from vrp_diffusion_quantum.inference.policy_support import PriorProvider, denoiser_prior
from vrp_diffusion_quantum.inference.predict_matrix import load_denoiser_checkpoint
from vrp_diffusion_quantum.inference.solve_instance import load_policy_checkpoint
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.train.train_policy import evaluate_policy
from vrp_diffusion_quantum.utils.feasibility import route_cost, validate_routes

LOGGER = logging.getLogger(__name__)


def fixed_prior(matrix: torch.Tensor) -> PriorProvider:
    """Bind the already-generated prior for a single-instance decoder comparison."""
    return lambda batch: matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="store_true")
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--matched-time", action="store_true")
    args = parser.parse_args()
    if args.matched_time and args.policy:
        parser.error("--matched-time runs solvers against completed policy end-to-end timings")
    root = Path("outputs/corrective_20260930")
    panel = json.loads((root / "panel.json").read_text())
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(root / "baselines.log")],
    )
    torch.set_num_threads(2)
    methods = ["pyvrp", "ortools"]
    policies = {
        "policy_n20": "policy_reinforce_s7799_n20_heldout_cuda_20260828T081136097007Z",
        "policy_n50": "policy_reinforce_s7799_n50_heldout_cuda_20260828T082044076031Z",
        "policy_n100": "policy_reinforce_s7799_n100_heldout_cuda_20260828T090345301408Z",
        "policy_starts16_n100": "policy_reinforce_s7799_n100_starts16_cuda_20260924T165104398879Z",
        "policy_starts16_seed4332_n100": (
            "policy_reinforce_s7799_n100_starts16_seed4332_cuda_20260924T191919208891Z"
        ),
        "policy_starts16_seed4333_n100": (
            "policy_reinforce_s7799_n100_starts16_seed4333_cuda_20260924T214845128519Z"
        ),
    }
    if args.policy:
        methods = list(policies)
        if args.end_to_end:
            methods = [method for method in methods if method.startswith("policy_n")]
    for method in methods:
        policy = None
        checkpoint = None
        if args.policy:
            checkpoint = Path("outputs/policy") / policies[method] / "checkpoints/best.pt"
            config = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text())
            sources = dataset_ids(Path(config["dataset"]["path"]))
            assert_disjoint_sources(
                sources,
                {entry["instance_id"] for entry in panel["examples"]},
            )
            (root / f"{method}_training_inventory.json").write_text(
                json.dumps(
                    {
                        "checkpoint": checkpoint.as_posix(),
                        "checkpoint_sha256": sha256(checkpoint),
                        "config_sha256": sha256(checkpoint.parent.parent / "config.yaml"),
                        "training_path": config["dataset"]["path"],
                        "unique_sources": len(sources),
                        "source_ids": sorted(sources),
                        "panel_overlap": 0,
                        "prior_checkpoint": config["prior"]["checkpoint"],
                    },
                    indent=2,
                )
                + "\n"
            )
            size = int(method.split("_n")[-1])
            expected_prior = Path(panel["checkpoints"][f"champion_n{size}"]["path"])
            if Path(config["prior"]["checkpoint"]).resolve() != expected_prior.resolve():
                raise ValueError("policy was trained against a different prior checkpoint")
            policy, _ = load_policy_checkpoint(checkpoint, device="cuda")
            if args.end_to_end:
                denoiser, denoiser_payload = load_denoiser_checkpoint(expected_prior, device="cuda")
                schedule = BernoulliDiffusionSchedule(**denoiser_payload["extra"]["schedule"]).to(
                    "cuda"
                )
        budgets = [1, 8, 16] if args.policy else ([16] if args.matched_time else [1, 3])
        for budget in budgets:
            name = f"{method}_{'starts' if args.policy else 'seconds'}{budget}"
            if args.matched_time:
                name = f"{method}_matched_policy16"
            if args.end_to_end:
                name += "_e2e"
            output = root / name
            output.mkdir(exist_ok=True)
            records = []
            for entry in panel["examples"]:
                if args.policy and entry["n_customers"] != size:
                    continue
                path = Path(entry["file"])
                if sha256(path) != entry["sha256"]:
                    raise ValueError("panel example changed")
                example = load_example(path)
                identity = entry["instance_id"]
                record_path = output / f"{identity}.json"
                if record_path.exists():
                    records.append(InstanceResult(**json.loads(record_path.read_text())))
                    continue
                prior_seconds = 0.0
                if args.policy:
                    prior_dir = root / f"champion_n{size}_posterior_mixture_v2_steps50_seed0"
                    prior_array = np.load(prior_dir / f"{identity}.npz")["probability"]
                    prior_tensor = torch.tensor(prior_array, device="cuda").unsqueeze(0)
                    prior_seconds = json.loads((prior_dir / f"{identity}.json").read_text())[
                        "runtime_seconds"
                    ]
                    torch.cuda.synchronize()
                start = time.perf_counter()
                if policy is not None:
                    provider = fixed_prior(prior_tensor)
                    if args.end_to_end:
                        graph_seed = int.from_bytes(
                            hashlib.sha256(f"0:{identity}".encode()).digest()[:4], "big"
                        )
                        provider = denoiser_prior(
                            denoiser,
                            schedule,
                            num_inference_steps=50,
                            seed=graph_seed,
                            sampler="posterior_mixture_v2",
                            use_probabilities=True,
                        )
                    metrics = evaluate_policy(
                        policy,
                        [example],
                        batch_size=1,
                        num_starts=budget,
                        device="cuda",
                        prior=provider,
                    )
                    torch.cuda.synchronize()
                    cost = metrics["best_cost"]
                    feasible = metrics["feasible_rate"] == 1.0
                else:
                    instance = example.instance
                    time_limit = float(budget)
                    if args.matched_time:
                        reference_timing = root / (
                            f"policy_n{entry['n_customers']}_starts16_e2e/{identity}.json"
                        )
                        time_limit = float(
                            json.loads(reference_timing.read_text())["runtime_seconds"]
                        )
                    solved = solve_instance(
                        SolverInstance(
                            coords=instance.coords,
                            demands=instance.demands,
                            capacity=instance.capacity,
                            depot_index=instance.depot_index,
                        ),
                        solver=method,
                        time_limit=time_limit,
                        seed=int.from_bytes(
                            hashlib.sha256(f"0:{identity}".encode()).digest()[:4], "big"
                        ),
                        instance_id=0,
                    )  # type: ignore[arg-type]
                    lookup = {
                        node: customer
                        for customer, node in enumerate(instance.customer_node_indices())
                    }
                    routes = [[lookup[node] for node in route] for route in solved.routes]
                    cost = route_cost(instance, routes)
                    feasible = solved.feasible and validate_routes(instance, routes).feasible
                elapsed = time.perf_counter() - start
                record = InstanceResult(
                    identity,
                    entry["n_customers"],
                    0,
                    0,
                    0,
                    cost,
                    route_cost(example.instance, example.solution.routes),
                    feasible,
                    elapsed,
                )
                record_path.write_text(json.dumps(asdict(record), indent=2) + "\n")
                if args.matched_time:
                    (output / f"{identity}.timing.json").write_text(
                        json.dumps(
                            {
                                "policy_end_to_end_budget_seconds": time_limit,
                                "solver_search_limit_seconds": time_limit,
                                "solver_elapsed_seconds": elapsed,
                                "accounting": (
                                    "search cap matches policy elapsed; solver setup is extra"
                                ),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                if args.policy and not args.end_to_end:
                    # The cached prior timing also includes diffusion's clustering decode;
                    # keep it separate rather than mislabeling this sum as policy end-to-end.
                    (output / f"{identity}.timing.json").write_text(
                        json.dumps(
                            {
                                "policy_decode_seconds": elapsed,
                                "cached_diffusion_plus_clustering_seconds": prior_seconds,
                            }
                        )
                    )
                records.append(record)
            summary = {
                "method": name,
                "num_instances": len(records),
                "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                "feasibility_rate": sum(record.feasible for record in records) / len(records),
                "runtime_seconds": sum(record.runtime_seconds for record in records),
            }
            for metric in ("mean_instance_gap_percent", "ratio_total_gap_percent"):
                summary[metric] = asdict(paired_bootstrap(records, metric=metric))  # type: ignore[arg-type]
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            LOGGER.info("%s gap=%.3f", name, summary["mean_instance_gap_percent"]["estimate"])


if __name__ == "__main__":
    main()
