"""CPU integration test for the CUDA-ready GAT pretraining entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from vrp_diffusion_quantum.data.dataset import make_example, save_example
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution


def _example(seed: int) -> CVRPExample:
    rng = np.random.default_rng(seed)
    instance = CVRPInstance(
        coords=np.vstack(([0.5, 0.5], rng.random((6, 2)))),
        demands=np.array([0.0, 1, 1, 1, 1, 1, 1]),
        capacity=6.0,
        depot_index=0,
        instance_id=f"gat_cli_{seed}",
        n_customers=6,
        seed=seed,
        generator_settings={},
    )
    solution = LabeledSolution(
        routes=[[0, 1, 2], [3, 4, 5]],
        cost=1.0,
        num_vehicles=2,
        feasible=True,
        solver_name="unit",
        time_budget=None,
        seed=seed,
        runtime_seconds=0.0,
    )
    return make_example(instance, solution)


def test_gat_pretrain_cli_writes_checkpoint_and_runtime_metrics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    for seed in range(2):
        save_example(_example(seed), train_dir / f"example_{seed}.json")
        save_example(_example(seed + 10), val_dir / f"example_{seed}.json")

    output_root = tmp_path / "outputs"
    config_path = tmp_path / "gat.yaml"
    config = {
        "experiment_name": "gat_cli_unit",
        "seed": 0,
        "dataset": {"path": str(train_dir)},
        "validation": {"path": str(val_dir)},
        "output": {"root": str(output_root)},
        "model": {
            "hidden_dim": 8,
            "gat_num_layers": 1,
            "gat_num_heads": 2,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.01,
            "batch_size": 1,
            "device": "cpu",
            "mixed_precision": False,
            "gradient_accumulation_steps": 2,
            "gradient_clip_norm": 1.0,
            "same_size_batches": True,
            "weighted_bce": True,
        },
        "mlflow": {"enabled": False},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    completed = subprocess.run(
        [sys.executable, "scripts/pretrain_gat_encoder.py", "--config", str(config_path)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    run_dir = next(output_root.glob("gat_cli_unit_*"))
    assert (run_dir / "checkpoints" / "gat_encoder_best.pt").is_file()
    summary = (run_dir / "summary.csv").read_text()
    assert "train_examples_per_second" in summary
    assert "peak_cuda_memory_reserved_bytes" in summary
