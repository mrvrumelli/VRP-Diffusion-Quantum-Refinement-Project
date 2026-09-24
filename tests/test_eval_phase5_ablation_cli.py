"""CPU-only end-to-end CLI test for the Phase 5 M ablation (scripts/eval_phase5_ablation.py).

Proves the full wiring -- all 8 arms, both dataset pools, partial per-size diffusion coverage,
and provenance reporting -- without any GPU or real checkpoint, following the subprocess +
tmp_path pattern already used by ``tests/test_pretrain_gat_cli.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from vrp_diffusion_quantum.data.dataset import make_example, save_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.train.train_diffusion import save_denoiser_checkpoint


def _example(n_customers: int, seed: int) -> CVRPExample:
    rng = np.random.default_rng(seed)
    coords = np.vstack(([0.5, 0.5], rng.random((n_customers, 2))))
    demands = np.concatenate([[0.0], np.ones(n_customers)])
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=float(n_customers),
        depot_index=0,
        instance_id=f"cvrp{n_customers}_{seed}",
        n_customers=n_customers,
        seed=seed,
        generator_settings={},
    )
    mid = max(1, n_customers // 2)
    routes = [route for route in (list(range(mid)), list(range(mid, n_customers))) if route]
    solution = LabeledSolution(
        routes=routes,
        cost=1.0,
        num_vehicles=len(routes),
        feasible=True,
        solver_name="unit",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def _write_examples(directory: Path, sizes_and_seeds: list[tuple[int, int]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for n, seed in sizes_and_seeds:
        example = _example(n, seed)
        save_example(example, directory / f"cvrp{n}_{seed}.json")


def _write_diffusion_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    save_denoiser_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        row={"val_loss": 0.1},
        best_metric_name="val_loss",
        best_metric_value=0.1,
        extra={
            "model": {"hidden_dim": 8, "num_layers": 1, "time_embed_dim": 8},
            "schedule": {"num_timesteps": 4, "beta_start": 1.0e-4, "beta_end": 2.0e-2},
        },
    )


def test_eval_phase5_ablation_cli_all_arms(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    train_dir = tmp_path / "train"
    selection_dir = tmp_path / "selection"
    test_dir = tmp_path / "test"
    accepted_dir = tmp_path / "accepted"

    _write_examples(train_dir, [(n, seed) for n in (20, 50) for seed in range(6)])
    _write_examples(selection_dir, [(n, seed) for n in (20, 50) for seed in range(100, 103)])
    _write_examples(test_dir, [(n, seed) for n in (20, 50) for seed in range(200, 203)])
    # Same (n, seed) pairs as some train examples => identical instance_ids => real overlap.
    _write_examples(accepted_dir, [(n, seed) for n in (20, 50) for seed in range(2)])

    pooled_ckpt = tmp_path / "pooled.pt"
    persize20_ckpt = tmp_path / "persize20.pt"
    _write_diffusion_checkpoint(pooled_ckpt)
    _write_diffusion_checkpoint(persize20_ckpt)

    output_root = tmp_path / "outputs"
    config_path = tmp_path / "phase5.yaml"
    config = {
        "experiment_name": "phase5_ablation_cli_test",
        "seed": 0,
        "device": "cpu",
        "dataset": {
            "train_path": str(train_dir),
            "selection_path": str(selection_dir),
            "test_path": str(test_dir),
            "accepted_matrix_examples_path": str(accepted_dir),
        },
        "max_train_examples": 0,
        "matrix_predictor": {
            "hidden_dim": 8,
            "epochs": 1,
            "learning_rate": 0.01,
            "augmentation": False,
            "weighted_bce": True,
            "pos_weight_power": 0.5,
        },
        "diffusion": {
            "step_stride": 1,
            "modes": ["one_shot", "full_chain"],
            "pooled": {"checkpoint": str(pooled_ckpt)},
            # 50/100 left null on purpose: only n20 has a per-size checkpoint configured.
            "persize": {"checkpoints": {20: str(persize20_ckpt), 50: None, 100: None}},
        },
        "baselines": {"random_seed": 7},
        "eval": {
            "selection_per_size": 3,
            "per_size": 3,
            "sizes": [20, 50],
            "seed": 0,
        },
        "output": {"root": str(output_root)},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    completed = subprocess.run(
        [sys.executable, "scripts/eval_phase5_ablation.py", "--config", str(config_path)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    run_dir = next(output_root.glob("phase5_ablation_cli_test_*"))
    assert (run_dir / "ablation_table.csv").is_file()
    assert (run_dir / "ablation_table.md").is_file()

    metrics = json.loads((run_dir / "ablation_metrics.json").read_text())
    rows = metrics["rows"]
    methods = {row["method"] for row in rows}
    assert methods == {
        "no_M",
        "random_M",
        "ground_truth_M",
        "supervised_M",
        "diffusion_M_pooled_one_shot",
        "diffusion_M_pooled_full_chain",
        "diffusion_M_persize_one_shot",
        "diffusion_M_persize_full_chain",
    }

    ground_truth_row = next(row for row in rows if row["method"] == "ground_truth_M")
    assert ground_truth_row["f1"] == 1.0

    # Only n20 has a per-size checkpoint: the persize rows must be restricted to n20 (n50's
    # per-size breakdown columns are absent, not silently zeroed or crashing).
    for mode in ("one_shot", "full_chain"):
        persize_row = next(row for row in rows if row["method"] == f"diffusion_M_persize_{mode}")
        assert "f1_n20" in persize_row
        assert "f1_n50" not in persize_row

    provenance = metrics["provenance"]
    overlap = provenance["accepted_subset_train_overlap"]
    assert overlap["overlap_count"] > 0
    assert overlap["pool_b_size"] == 4
    assert provenance["diffusion_persize_available_sizes"]["one_shot"] == [20]
    assert provenance["diffusion_persize_available_sizes"]["full_chain"] == [20]
