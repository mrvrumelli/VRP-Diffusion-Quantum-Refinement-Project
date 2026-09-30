"""Separate held-out calibration from scored graphs; compare posterior and final-only shifts."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from run_corrective_evaluation import evaluate
from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    load_denoiser_checkpoint,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule


class ProbabilityShift(ConstraintDenoiser):
    """Frozen post-hoc size-specific logit intercept; no checkpoint parameters change."""

    def __init__(self, base: ConstraintDenoiser, shifts: dict[int, float]) -> None:
        torch.nn.Module.__init__(self)
        self.base = base
        self.shifts = shifts
        self.coordinate_frame = base.coordinate_frame

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
        original = self.base.predict_proba(
            customer_coords, customer_demands, capacity, m_t, t, customer_mask=customer_mask
        )
        return torch.sigmoid(
            torch.logit(original.clamp(1e-6, 1 - 1e-6)) + self.shifts[m_t.shape[-1]]
        )


@torch.no_grad()
def main() -> None:
    output = Path("outputs/corrective_calibration_20260930")
    output.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    torch.set_num_threads(2)
    source_panel = json.loads(Path("outputs/corrective_20260930/panel.json").read_text())
    excluded = {
        entry["instance_id"]
        for entry in json.loads(
            Path("outputs/corrective_confirmation_20260930/panel.json").read_text()
        )["examples"]
    }
    # Calibration never uses the scored diagnostic or expanded development panels.
    examples = []
    for size in (20, 50, 100):
        choices = [
            load_example(path)
            for path in sorted(
                Path("data/processed/s7799_val100_policy_v1").glob(f"cvrp{size}_*.json")
            )
        ]
        choices = [example for example in choices if example.instance.instance_id not in excluded]
        rng = np.random.default_rng(np.random.SeedSequence([93831, size]))
        examples.extend(choices[int(index)] for index in rng.permutation(len(choices))[:8])
    checkpoint = source_panel["checkpoints"]["full"]
    base, _ = load_denoiser_checkpoint(checkpoint["path"], device="cuda")
    schedule = BernoulliDiffusionSchedule(1000).to("cuda")
    shifts = {}
    for size in (20, 50, 100):
        all_logits, all_targets = [], []
        for index, example in enumerate(examples):
            if example.instance.n_customers != size:
                continue
            coords, demands, capacity, target, mask = example_to_model_inputs(
                example, device="cuda"
            )
            off_diag = ~torch.eye(size, device="cuda", dtype=torch.bool)
            for timestep in (0, 250, 500, 999):
                corrupted = schedule.q_sample(
                    target, timestep, generator=torch.Generator().manual_seed(71 + index)
                )
                logits = base(
                    coords,
                    demands,
                    capacity,
                    corrupted,
                    torch.tensor([timestep], device="cuda"),
                    customer_mask=mask,
                )
                all_logits.append(logits[0][off_diag].cpu().numpy())
                all_targets.append(target[0][off_diag].cpu().numpy())
        logits = np.concatenate(all_logits)
        target_array = np.concatenate(all_targets)
        candidates = np.linspace(-4, 4, 161)
        losses = [
            np.mean(np.logaddexp(0, logits + shift) - target_array * (logits + shift))
            for shift in candidates
        ]
        shifts[size] = float(candidates[int(np.argmin(losses))])
    panel = copy.deepcopy(source_panel)
    panel["checkpoints"] = {"full_shift_inside": checkpoint}
    panel["calibration"] = {
        "seed": 93831,
        "source_ids": [example.instance.instance_id for example in examples],
        "shifts": shifts,
        "criterion": (
            "unweighted BCE pooled over t=0,250,500,999 within size; intercept grid [-4,4] step .05"
        ),
    }
    (output / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    shifted = ProbabilityShift(base, shifts)
    for seed in (0, 1):
        evaluate(
            output,
            panel,
            "full_shift_inside",
            50,
            "posterior_mixture_v2",
            seed,
            model_override=shifted,
        )
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for entry in source_panel["examples"]:
            archive = np.load(
                Path("outputs/corrective_20260930")
                / f"full_posterior_mixture_v2_steps50_seed{seed}"
                / f"{entry['instance_id']}.npz"
            )
            probability, truth = archive["probability"], archive["target"]
            off_diag = ~np.eye(len(truth), dtype=bool)
            # The same intercept at the final decision is equivalent to this threshold.
            threshold = 1 / (1 + np.exp(shifts[entry["n_customers"]]))
            predicted = (probability >= threshold) & off_diag
            target_mask = (truth >= 0.5) & off_diag
            counts["tp"] += int((predicted & target_mask).sum())
            counts["fp"] += int((predicted & ~target_mask).sum())
            counts["fn"] += int((~predicted & target_mask).sum())
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        (output / f"final_only_seed{seed}.json").write_text(
            json.dumps(
                {
                    **counts,
                    "f1": 2 * counts["tp"] / denominator,
                    "note": (
                        "final classification only; reverse probabilities and routing unchanged"
                    ),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
