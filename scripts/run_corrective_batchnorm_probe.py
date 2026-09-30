"""Compare cloned train/eval outputs across sizes and noise without changing checkpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from run_corrective_evaluation import sha256
from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    load_denoiser_checkpoint,
)
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(2)
    panel = json.loads(Path("outputs/corrective_20260930/panel.json").read_text())
    rows = []
    for checkpoint in sorted(Path("outputs/corrective_learning_20260930").glob("bn_*/last.pt")):
        model, _ = load_denoiser_checkpoint(checkpoint, device="cpu")
        before = {name: value.clone() for name, value in model.state_dict().items()}
        schedule = BernoulliDiffusionSchedule(1000)
        for size in (20, 50, 100):
            entry = next(row for row in panel["examples"] if row["n_customers"] == size)
            example = load_example(Path(entry["file"]))
            coords, demands, capacity, target, mask = example_to_model_inputs(example, device="cpu")
            off_diagonal = ~torch.eye(size, dtype=torch.bool)
            for timestep in (0, 250, 500, 999):
                t = torch.tensor([timestep])
                noisy = schedule.q_sample(target, t, generator=torch.Generator().manual_seed(9017))
                model.eval()
                eval_prob = model.predict_proba(
                    coords, demands, capacity, noisy, t, customer_mask=mask
                )[0][off_diagonal]
                clone = copy.deepcopy(model)
                clone.train()
                train_prob = clone.predict_proba(
                    coords, demands, capacity, noisy, t, customer_mask=mask
                )[0][off_diagonal]
                rows.append(
                    {
                        "model": checkpoint.parent.name,
                        "checkpoint_sha256": sha256(checkpoint),
                        "instance_id": example.instance.instance_id,
                        "size": size,
                        "t": timestep,
                        "batch_size": 1,
                        "mean_abs_probability_difference": float(
                            (eval_prob - train_prob).abs().mean()
                        ),
                        "eval_positive_fraction": float((eval_prob >= 0.5).float().mean()),
                        "train_mode_positive_fraction": float((train_prob >= 0.5).float().mean()),
                    }
                )
        if not all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()):
            raise ValueError("diagnostic modified original persistent model state")
    output = Path("outputs/corrective_learning_20260930/batchnorm_all_noise_probe.json")
    output.write_text(json.dumps({"original_state_unchanged": True, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
