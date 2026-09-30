"""Bounded sanity and matched diagnostic training; these are not paper reproductions."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from vrp_diffusion_quantum.data.dataset import collate_batch, load_example
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.inference.policy_support import (
    batch_to_device,
    sample_constraint_matrix_batch,
)
from vrp_diffusion_quantum.inference.predict_matrix import evaluate_full_chain_sampling
from vrp_diffusion_quantum.metrics.matrix_metrics import MatrixPrediction, compute_matrix_metrics
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.train.train_diffusion import (
    customer_tensors_from_batch,
    diffusion_matrix_bce_loss,
    save_denoiser_checkpoint,
)

LOGGER = logging.getLogger(__name__)


class OracleDenoiser(ConstraintDenoiser):
    """A diagnostic clean-state oracle, independent of corrupted inputs."""

    def __init__(self, target: Tensor) -> None:
        super().__init__(hidden_dim=8, num_layers=1, time_embed_dim=8)
        self.register_buffer("target", target)

    def predict_proba(
        self,
        customer_coords: Tensor,
        customer_demands: Tensor,
        capacity: Tensor,
        m_t: Tensor,
        t: Tensor,
        *,
        customer_mask: Tensor | None = None,
    ) -> Tensor:
        return self.target.expand_as(m_t)


@torch.no_grad()
def noisy_metrics(
    model: ConstraintDenoiser, schedule: BernoulliDiffusionSchedule, examples: list[CVRPExample]
) -> dict[str, Any]:
    """Fixed-threshold denoising metrics at four declared corruption levels."""
    model.eval()
    result = {}
    for timestep in (0, 250, 500, 999):
        predictions = []
        for index, example in enumerate(examples):
            batch = batch_to_device(collate_batch([example]), "cuda")
            coords, demands, capacity = customer_tensors_from_batch(
                batch, coordinate_frame=model.coordinate_frame
            )
            noise = schedule.q_sample(
                batch.constraint_matrix,
                timestep,
                generator=torch.Generator().manual_seed(8017 + index),
            )
            prob = model.predict_proba(
                coords,
                demands,
                capacity,
                noise,
                torch.tensor([timestep], device="cuda"),
                customer_mask=batch.customer_mask,
            )
            predictions.append(MatrixPrediction.from_example(example, prob[0].cpu().numpy()))
        result[str(timestep)] = asdict(
            compute_matrix_metrics(predictions, threshold=0.5, adaptive_threshold=False)
        )
    return result


def run(arm: str, seed: int, updates: int, output: Path, single_size: int | None = None) -> None:
    run_dir = output / f"{arm}_seed{seed}"
    run_dir.mkdir(exist_ok=True)
    if (run_dir / "summary.json").exists():
        LOGGER.info("Already complete: %s", run_dir)
        return
    tiny = arm == "tiny"
    sizes = (20,) if tiny else ((single_size,) if single_size else (20, 50, 100))
    train_by_size = {}
    for size in sizes:
        source = Path(f"data/processed/corrective_20260930/curve_n{size}_500")
        paths = sorted(source.glob("cvrp*.json"))
        rng = np.random.default_rng(np.random.SeedSequence([93711, size]))
        selected: dict[str, CVRPExample] = {}
        for index in rng.permutation(len(paths)):
            example = load_example(paths[int(index)])
            selected.setdefault(example.instance.instance_id, example)
            if len(selected) == (1 if tiny else 32):
                break
        train_by_size[size] = list(selected.values())
    panel = json.loads(Path("outputs/corrective_20260930/panel.json").read_text())
    validation = [
        load_example(Path(entry["file"]))
        for entry in panel["examples"]
        if entry["n_customers"] in sizes
    ]
    train_examples = [example for examples in train_by_size.values() for example in examples]
    assert not {e.instance.instance_id for e in train_examples} & {
        e.instance.instance_id for e in validation
    }
    torch.manual_seed(seed)
    model_config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "time_embed_dim": 64,
        "node_encoder_type": "gat",
        "gat_num_layers": 3,
        "gat_num_heads": 4,
        "freeze_node_encoder": False,
        "normalization": "batch_norm" if arm.startswith("bn_") else "layer_norm",
        "edge_input_features": "noisy_matrix_distance",
        "coordinate_frame": "depot_relative" if arm == "depot" else "absolute",
    }
    model = ConstraintDenoiser(**model_config).to("cuda")  # type: ignore[arg-type]
    schedule = BernoulliDiffusionSchedule(1000).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    config = {
        "arm": arm,
        "seed": seed,
        "updates": updates,
        "sizes": sizes,
        "batch_size": 1 if tiny else 4,
        "model": model_config,
        "schedule": {"num_timesteps": 1000},
        "learning_rate": 0.0003,
        "weighted_bce": arm != "unweighted",
        "pos_weight_power": 1.0,
        "augmentation": False,
        "precision": "float32",
        "sampler": "posterior_mixture_v2",
        "checkpoint_selection": "final fixed update; no validation-based early stopping",
        "train_sources": [e.instance.instance_id for e in train_examples],
        "validation_sources": [e.instance.instance_id for e in validation],
        "claim": "reduced-width diagnostic; jointly trained GAT; not a paper reproduction",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.perf_counter()
    history = []
    batches = []
    epoch = 0
    for update in range(updates):
        if not batches:
            epoch += 1
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
            for size in sizes:
                examples = train_by_size[size]
                order = rng.permutation(len(examples))
                batches.extend(
                    [
                        [examples[int(i)] for i in order[j : j + config["batch_size"]]]
                        for j in range(0, len(order), config["batch_size"])
                    ]
                )
            if arm != "bn_sorted":
                rng.shuffle(batches)
        chunk = batches.pop(0)
        batch = batch_to_device(collate_batch(chunk), "cuda")
        coords, demands, capacity = customer_tensors_from_batch(
            batch, coordinate_frame=model.coordinate_frame
        )
        generator = torch.Generator().manual_seed(seed * 100000 + update)
        timesteps = schedule.sample_timesteps(len(chunk), device="cuda", generator=generator)
        corrupted = schedule.q_sample(batch.constraint_matrix, timesteps, generator=generator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            coords, demands, capacity, corrupted, timesteps, customer_mask=batch.customer_mask
        )
        loss = diffusion_matrix_bce_loss(
            logits,
            batch.constraint_matrix,
            batch.customer_mask,
            weighted=arm != "unweighted",
            pos_weight_power=1.0,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite diagnostic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 100 == 0 or update == updates - 1:
            row = {"update": update + 1, "loss": float(loss.detach())}
            history.append(row)
            LOGGER.info("%s update=%d loss=%.5f", run_dir.name, update + 1, row["loss"])
    summary: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "history": history,
        "training_seconds": time.perf_counter() - start,
    }
    summary["noisy_train"] = noisy_metrics(model, schedule, train_examples[:8])
    summary["noisy_validation"] = noisy_metrics(model, schedule, validation)
    for label, examples in (("train", train_examples[:8]), ("validation", validation)):
        summary[f"generated_{label}"] = evaluate_full_chain_sampling(
            model,
            schedule,
            examples,
            device="cuda",
            seed=9001,
            num_inference_steps=50,
            sampler="posterior_mixture_v2",
            batch_size=1,
        )
    if tiny:
        batch = batch_to_device(collate_batch(train_examples), "cuda")
        coords, demands, capacity = customer_tensors_from_batch(batch)
        oracle = OracleDenoiser(batch.constraint_matrix).to("cuda")
        summary["oracle_exact"] = {}
        for steps in (1, 50, 1000):
            result = sample_constraint_matrix_batch(
                oracle,
                schedule,
                coords=coords,
                demands=demands,
                capacity=capacity,
                customer_mask=batch.customer_mask,
                generator=torch.Generator().manual_seed(123),
                num_inference_steps=steps,
                sampler="posterior_mixture_v2",
            )
            summary["oracle_exact"][str(steps)] = bool(
                torch.equal(result.m_hat, batch.constraint_matrix)
            )
        summary["tiny_fit_gate_passed"] = summary["generated_train"]["sample_f1"] >= 0.95 and all(
            summary["oracle_exact"].values()
        )
    save_denoiser_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        row=history[-1],
        best_metric_name="fixed_final_update",
        best_metric_value=float(updates),
        extra={"model": model_config, "schedule": {"num_timesteps": 1000}},
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info(
        "Completed %s generated train/val F1=%.4f/%.4f",
        run_dir.name,
        summary["generated_train"]["sample_f1"],
        summary["generated_validation"]["sample_f1"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["tiny", "baseline", "depot", "unweighted", "bn_sorted", "bn_shuffled"],
        default=["tiny"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[4331])
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--single-size", type=int, choices=[20, 50, 100])
    parser.add_argument("--output", type=Path, default=Path("outputs/corrective_learning_20260930"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log")],
    )
    torch.set_num_threads(2)
    for arm in args.arms:
        for seed in args.seeds:
            run(arm, seed, args.updates, args.output, args.single_size)


if __name__ == "__main__":
    main()
