import json
from pathlib import Path

from vrp_diffusion_quantum.utils.experiment import ExperimentTracker, hash_dataset


def test_hash_dataset_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("node_id,x,y,demand\n0,0.0,0.0,0\n")

    first_hash = hash_dataset(dataset)
    second_hash = hash_dataset(dataset)
    assert first_hash == second_hash

    dataset.write_text("node_id,x,y,demand\n0,0.0,0.0,1\n")
    assert hash_dataset(dataset) != first_hash


def test_hash_dataset_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    try:
        hash_dataset(missing)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing dataset path")


def test_experiment_tracker_writes_full_output_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("node_id,x,y,demand\n0,0.0,0.0,0\n1,1.0,0.0,3\n")

    output_root = tmp_path / "outputs"
    config = {"training": {"epochs": 2}}

    with ExperimentTracker(
        output_root=output_root,
        experiment_name="unit_test_experiment",
        config=config,
        seed=42,
        dataset_path=dataset,
    ) as tracker:
        run_dir = tracker.run_dir
        assert run_dir.parent == output_root
        assert run_dir.name.startswith("unit_test_experiment_")

        tracker.log_metric_row({"epoch": 1, "train_loss": 1.0})
        tracker.log_metric_row({"epoch": 2, "train_loss": 0.5})
        tracker.log_metrics({"final_train_loss": 0.5})

        plot_path = tracker.plots_dir / "dummy.png"
        plot_path.write_bytes(b"not a real png, just a placeholder")

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "seed.txt").read_text().strip() == "42"
    assert (run_dir / "dataset_hash.txt").read_text().strip() == hash_dataset(dataset)
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "plots" / "dummy.png").is_file()

    summary_lines = (run_dir / "summary.csv").read_text().strip().splitlines()
    assert summary_lines[0] == "epoch,train_loss"
    assert len(summary_lines) == 3

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics == {"final_train_loss": 0.5}


def test_experiment_tracker_without_dataset_path_skips_dataset_hash(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    with ExperimentTracker(
        output_root=output_root,
        experiment_name="no_dataset_experiment",
        config={},
        seed=0,
    ) as tracker:
        run_dir = tracker.run_dir

    assert tracker.dataset_hash is None
    assert not (run_dir / "dataset_hash.txt").exists()


def test_experiment_tracker_allows_same_experiment_started_quickly(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    with ExperimentTracker(
        output_root=output_root,
        experiment_name="collision_test",
        config={},
        seed=0,
    ) as first_tracker:
        first_run_dir = first_tracker.run_dir

    with ExperimentTracker(
        output_root=output_root,
        experiment_name="collision_test",
        config={},
        seed=0,
    ) as second_tracker:
        second_run_dir = second_tracker.run_dir

    assert first_run_dir != second_run_dir
    assert first_run_dir.is_dir()
    assert second_run_dir.is_dir()
