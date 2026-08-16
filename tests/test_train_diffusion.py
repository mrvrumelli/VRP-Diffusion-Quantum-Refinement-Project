"""Tests for constraint-denoiser diffusion training (task P3.3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vrp_diffusion_quantum.data.consensus_targets import build_consensus_targets
from vrp_diffusion_quantum.data.dataset import (
    IndexedJSONDataset,
    collate_batch,
    make_example,
    save_example,
)
from vrp_diffusion_quantum.data.types import CVRPExample, CVRPInstance, LabeledSolution
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule
from vrp_diffusion_quantum.train.train_diffusion import (
    diffusion_matrix_bce_loss,
    evaluate_constraint_denoiser,
    train_constraint_denoiser,
)


def _example(n_customers: int, *, seed: int) -> CVRPExample:
    rng = np.random.default_rng(seed)
    coords = np.vstack([[0.5, 0.5], rng.random((n_customers, 2))])
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
    # Two routes so M is not fully connected (non-trivial target).
    mid = n_customers // 2
    routes = [list(range(mid)), list(range(mid, n_customers))]
    routes = [route for route in routes if route]
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


def test_diffusion_matrix_bce_loss_ignores_diagonal_and_padding() -> None:
    logits = torch.zeros(1, 3, 3)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 0] = 10.0
    m_true = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    # Only first two customers are real; pair (0,2) must not affect the loss.
    customer_mask = torch.tensor([[True, True, False]])
    logits[0, 0, 2] = -100.0
    logits[0, 2, 0] = -100.0
    loss = diffusion_matrix_bce_loss(logits, m_true, customer_mask)
    assert torch.isfinite(loss)
    assert loss.item() < 0.1


def test_diffusion_loss_confidence_mask_ignores_disputed_pair() -> None:
    logits = torch.tensor([[[0.0, -100.0, 10.0], [-100.0, 0.0, 10.0], [10.0, 10.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.5, 1.0], [0.5, 0.0, 1.0], [1.0, 1.0, 0.0]]])
    confidence = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]])
    loss = diffusion_matrix_bce_loss(
        logits,
        target,
        torch.ones(1, 3, dtype=torch.bool),
        weighted=False,
        pair_weights=confidence,
    )
    assert loss.item() < 0.001


def test_train_constraint_denoiser_rejects_empty() -> None:
    torch.manual_seed(0)
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1)
    schedule = BernoulliDiffusionSchedule(num_timesteps=10)
    with pytest.raises(ValueError, match="empty"):
        train_constraint_denoiser(
            model, schedule, [], num_epochs=1, learning_rate=0.01, batch_size=1
        )


def test_train_constraint_denoiser_loss_decreases() -> None:
    torch.manual_seed(0)
    examples = [_example(5, seed=0), _example(6, seed=1), _example(5, seed=2)]
    model = ConstraintDenoiser(hidden_dim=32, num_layers=2, time_embed_dim=32)
    schedule = BernoulliDiffusionSchedule(num_timesteps=100)

    history = train_constraint_denoiser(
        model,
        schedule,
        examples,
        num_epochs=40,
        learning_rate=0.01,
        batch_size=2,
        seed=0,
    )

    assert len(history) == 40
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    # Validation metrics are present every epoch (P3.3 done criterion).
    for row in history:
        assert "val_loss" in row
        assert "val_bce" in row
        assert "val_f1" in row
        assert "val_auc" in row
        assert row["val_loss"] >= 0.0
        assert row["val_bce"] >= 0.0


def test_evaluate_constraint_denoiser_returns_metrics() -> None:
    torch.manual_seed(0)
    examples = [_example(5, seed=0), _example(5, seed=1)]
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=50)
    val_loss, metrics = evaluate_constraint_denoiser(
        model,
        schedule,
        examples,
        batch_size=2,
        generator=torch.Generator().manual_seed(0),
        t_sample="high",
    )
    assert val_loss >= 0.0
    assert metrics.num_pairs > 0
    assert 0.0 <= metrics.f1 <= 1.0


def test_evaluate_constraint_denoiser_rejects_bad_t_sample() -> None:
    examples = [_example(5, seed=0)]
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    schedule = BernoulliDiffusionSchedule(num_timesteps=10)
    try:
        evaluate_constraint_denoiser(model, schedule, examples, t_sample="bogus")
    except ValueError as exc:
        assert "t_sample" in str(exc)
    else:
        raise AssertionError("expected ValueError for bad t_sample")


def test_train_constraint_denoiser_writes_checkpoints(tmp_path: Path) -> None:
    torch.manual_seed(0)
    examples = [_example(5, seed=0), _example(5, seed=1)]
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=50)
    ckpt_dir = tmp_path / "checkpoints"

    history = train_constraint_denoiser(
        model,
        schedule,
        examples,
        num_epochs=3,
        learning_rate=0.02,
        batch_size=2,
        seed=0,
        checkpoint_dir=ckpt_dir,
        best_metric="val_loss",
        minimize_best=True,
    )

    assert len(history) == 3
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    assert last_path.is_file()
    assert best_path.is_file()
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    assert "model" in payload and "optimizer" in payload
    assert "scaler" in payload and "rng_state" in payload
    assert payload["best_metric_name"] == "val_loss"
    assert int(payload["epoch"]) >= 0


def test_train_constraint_denoiser_accumulates_and_logs_runtime() -> None:
    examples = [_example(5, seed=i) for i in range(3)]
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    schedule = BernoulliDiffusionSchedule(num_timesteps=10)

    history = train_constraint_denoiser(
        model,
        schedule,
        examples,
        num_epochs=1,
        learning_rate=0.01,
        batch_size=2,
        seed=0,
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
    )

    row = history[0]
    assert row["microbatches"] == 2
    assert row["optimizer_steps"] == 1
    assert row["gradient_accumulation_steps"] == 2
    assert row["epoch_runtime_seconds"] > 0
    assert row["train_examples_per_second"] > 0
    assert np.isfinite(row["gradient_norm"])


def test_full_chain_checkpoint_selection_uses_fixed_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = [_example(4, seed=0)]
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    schedule = BernoulliDiffusionSchedule(num_timesteps=3)
    observed_seeds: list[int] = []

    def fake_sample_eval(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        observed_seeds.append(int(kwargs["seed"]))
        value = 20.0 - len(observed_seeds)
        return {
            "sample_f1": 0.5,
            "sample_precision": 0.5,
            "sample_recall": 0.5,
            "sample_num_examples": 1,
            "route_mean_cost_gap_percent": value,
        }

    monkeypatch.setattr(
        "vrp_diffusion_quantum.train.train_diffusion.evaluate_full_chain_sampling",
        fake_sample_eval,
    )
    ckpt_dir = tmp_path / "checkpoints"
    history = train_constraint_denoiser(
        model,
        schedule,
        examples,
        num_epochs=2,
        learning_rate=0.01,
        sample_eval_examples=examples,
        sample_eval_every=1,
        sample_eval_seed=777,
        checkpoint_dir=ckpt_dir,
        best_metric="route_mean_cost_gap_percent",
        minimize_best=True,
    )

    assert observed_seeds == [777, 777]
    assert history[-1]["route_mean_cost_gap_percent"] == 18.0
    payload = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    assert payload["epoch"] == 1
    assert payload["best_metric_name"] == "route_mean_cost_gap_percent"


def test_stochastic_references_keep_one_update_per_source() -> None:
    first = _example(4, seed=0)
    alternative = _example(4, seed=1)
    alternative.instance.instance_id = first.instance.instance_id
    alternative.instance.generator_settings["training_label_policy"] = {
        "source_file": "shared.json",
        "candidate_route_hash": "b",
    }
    first.instance.generator_settings["training_label_policy"] = {
        "source_file": "shared.json",
        "candidate_route_hash": "a",
    }
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    history = train_constraint_denoiser(
        model,
        BernoulliDiffusionSchedule(num_timesteps=3),
        [first, alternative],
        num_epochs=2,
        learning_rate=0.01,
        batch_size=1,
        stochastic_references=True,
    )
    assert [row["train_examples_seen"] for row in history] == [1, 1]
    assert [row["unique_train_sources"] for row in history] == [1, 1]


def test_consensus_target_training_keeps_hard_examples_unchanged() -> None:
    first = _example(4, seed=0)
    second = _example(4, seed=1)
    for item, route_hash in ((first, "a"), (second, "b")):
        item.instance.instance_id = "shared"
        item.instance.generator_settings["training_label_policy"] = {
            "source_file": "shared.json",
            "candidate_route_hash": route_hash,
        }
    original = first.constraint_matrix.copy()
    targets = build_consensus_targets([first, second])
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    history = train_constraint_denoiser(
        model,
        BernoulliDiffusionSchedule(num_timesteps=3),
        [first],
        num_epochs=1,
        learning_rate=0.01,
        consensus_targets=targets,
        consensus_use_confidence=True,
    )
    assert history[0]["consensus_use_confidence"] is True
    np.testing.assert_array_equal(first.constraint_matrix, original)


def test_training_streams_indexed_json_dataset(tmp_path: Path) -> None:
    for index in range(3):
        save_example(_example(4, seed=index), tmp_path / f"cvrp4_{index}.json")
    dataset = IndexedJSONDataset(tmp_path, cache_size=0)
    model = ConstraintDenoiser(hidden_dim=8, num_layers=1, time_embed_dim=8)
    history = train_constraint_denoiser(
        model,
        BernoulliDiffusionSchedule(num_timesteps=3),
        dataset,
        val_examples=dataset,
        num_epochs=1,
        learning_rate=0.01,
        batch_size=2,
        same_size_batches=True,
    )
    assert history[0]["train_examples_seen"] == 3
    assert dataset._cache == {}  # cache_size=0 never retains parsed matrices


def test_train_diffusion_script_logs_metrics(tmp_path: Path) -> None:
    """End-to-end: ExperimentTracker writes summary.csv and metrics.json with val fields."""
    from vrp_diffusion_quantum.data.dataset import load_dataset, save_example
    from vrp_diffusion_quantum.utils.experiment import ExperimentTracker

    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    for i, example in enumerate([_example(4, seed=i) for i in range(2)]):
        save_example(example, dataset_dir / f"ex_{i}.json")

    config = {
        "experiment_name": "unit_diffusion",
        "seed": 0,
        "dataset": {"name": "unit", "path": str(dataset_dir)},
        "output": {"root": str(tmp_path / "out")},
        "model": {"hidden_dim": 16, "num_layers": 1, "time_embed_dim": 16},
        "schedule": {"num_timesteps": 50},
        "training": {"epochs": 5, "learning_rate": 0.02, "batch_size": 2},
    }

    torch.manual_seed(0)
    examples = load_dataset(dataset_dir)
    model = ConstraintDenoiser(hidden_dim=16, num_layers=1, time_embed_dim=16)
    schedule = BernoulliDiffusionSchedule(num_timesteps=50)
    ckpt_dir = tmp_path / "checkpoints"

    with ExperimentTracker(
        output_root=tmp_path / "out",
        experiment_name="unit_diffusion",
        config=config,
        seed=0,
        dataset_path=dataset_dir,
    ) as tracker:

        def _log_row(row: dict) -> None:
            tracker.log_metric_row(
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"checkpoint_last", "checkpoint_best", "is_best"}
                }
            )

        history = train_constraint_denoiser(
            model,
            schedule,
            examples,
            num_epochs=5,
            learning_rate=0.02,
            batch_size=2,
            seed=0,
            checkpoint_dir=ckpt_dir,
            on_epoch_end=_log_row,
        )
        tracker.log_metrics(
            {
                "final_train_loss": history[-1]["train_loss"],
                "train_loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
            }
        )
        run_dir = tracker.run_dir

    summary = run_dir / "summary.csv"
    metrics_path = run_dir / "metrics.json"
    assert summary.is_file()
    assert metrics_path.is_file()
    assert (ckpt_dir / "best.pt").is_file()
    summary_text = summary.read_text()
    assert "train_loss" in summary_text
    assert "val_bce" in summary_text
    assert "val_f1" in summary_text
    metrics = json.loads(metrics_path.read_text())
    assert metrics["train_loss_decreased"] is True

    batch = collate_batch([_example(4, seed=0), _example(6, seed=1)])
    assert batch.constraint_matrix.shape[1] == 6


def test_train_diffusion_cli_logs_mlflow(tmp_path: Path) -> None:
    """CLI script writes checkpoints and an MLflow file-store run."""
    import mlflow

    from vrp_diffusion_quantum.data.dataset import save_example

    root = Path(__file__).resolve().parents[1]
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    for i, example in enumerate([_example(4, seed=i) for i in range(2)]):
        save_example(example, dataset_dir / f"ex_{i}.json")

    mlflow_db = tmp_path / "mlflow.db"
    out_root = tmp_path / "out"
    config_path = tmp_path / "cfg.yaml"
    tracking_uri = f"sqlite:///{mlflow_db.resolve().as_posix()}"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: unit_mlflow",
                "seed: 0",
                "dataset:",
                "  name: unit",
                f"  path: {dataset_dir.resolve().as_posix()}",
                "validation: {}",
                "output:",
                f"  root: {out_root.resolve().as_posix()}",
                "model:",
                "  hidden_dim: 16",
                "  num_layers: 1",
                "  time_embed_dim: 16",
                "schedule:",
                "  num_timesteps: 40",
                "training:",
                "  epochs: 3",
                "  learning_rate: 0.02",
                "  batch_size: 2",
                "  device: cpu",
                "checkpoint:",
                "  best_metric: val_loss",
                "  minimize: true",
                "mlflow:",
                "  enabled: true",
                f"  tracking_uri: {tracking_uri}",
                "  experiment_name: unit_mlflow",
                "",
            ]
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "vrp_diffusion_quantum.train.train_diffusion",
        "--config",
        str(config_path),
    ]
    result = subprocess.run(cmd, cwd=root, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    run_dirs = list(out_root.glob("unit_mlflow_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "checkpoints" / "best.pt").is_file()
    assert (run_dirs[0] / "summary.csv").is_file()

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=["unit_mlflow"])
    assert len(runs) >= 1
    assert (
        "metrics.train_loss" in runs.columns or "metrics.summary_final_train_loss" in runs.columns
    )


def test_train_diffusion_cli_completed_resume_exits_cleanly(tmp_path: Path) -> None:
    """Resuming a checkpoint that reached the configured epochs is a successful no-op."""
    from vrp_diffusion_quantum.data.dataset import save_example

    root = Path(__file__).resolve().parents[1]
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    save_example(_example(4, seed=0), dataset_dir / "ex.json")
    output_root = tmp_path / "out"
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: completed_resume",
                "seed: 0",
                "dataset:",
                f"  path: {dataset_dir.resolve().as_posix()}",
                "validation: {}",
                "output:",
                f"  root: {output_root.resolve().as_posix()}",
                "model:",
                "  hidden_dim: 8",
                "  num_layers: 1",
                "  time_embed_dim: 8",
                "schedule:",
                "  num_timesteps: 4",
                "training:",
                "  epochs: 1",
                "  learning_rate: 0.01",
                "  batch_size: 1",
                "  device: cpu",
                "checkpoint:",
                "  best_metric: val_loss",
                "  minimize: true",
                "mlflow:",
                "  enabled: false",
                "",
            ]
        )
    )
    command = [
        sys.executable,
        "-m",
        "vrp_diffusion_quantum.train.train_diffusion",
        "--config",
        str(config_path),
    ]
    first = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + "\n" + first.stderr
    first_run = next(output_root.glob("completed_resume_*"))
    checkpoint = first_run / "checkpoints" / "last.pt"

    resumed = subprocess.run(
        [*command, "--resume", str(checkpoint)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stdout + "\n" + resumed.stderr
    assert "nothing to train" in resumed.stdout
    run_dirs = sorted(output_root.glob("completed_resume_*"))
    assert len(run_dirs) == 2
    metrics = json.loads((run_dirs[-1] / "metrics.json").read_text())
    assert metrics["status"] == "already_complete"
