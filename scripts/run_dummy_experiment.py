"""Run a tiny dummy experiment to prove the experiment-tracking template works end to end.

Produces a full experiment output directory (config, dataset hash, seed, metric table, plots,
logs) under outputs/train/, per docs/coding_standards.md section 6. This script trains nothing
real; it exists only to exercise ExperimentTracker.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from vrp_diffusion_quantum.utils.experiment import ExperimentTracker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "train" / "dummy_experiment.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["seed"])
    rng = random.Random(seed)

    dataset_path = ROOT / config["dataset"]["path"]
    output_root = ROOT / config["output"]["root"]
    num_epochs = int(config["training"]["epochs"])

    with ExperimentTracker(
        output_root=output_root,
        experiment_name=config["experiment_name"],
        config=config,
        seed=seed,
        dataset_path=dataset_path,
    ) as tracker:
        losses = []
        for epoch in range(1, num_epochs + 1):
            loss = 1.0 / epoch + rng.uniform(-0.01, 0.01)
            losses.append(loss)
            tracker.log_metric_row({"epoch": epoch, "train_loss": round(loss, 6)})

        tracker.log_metrics(
            {
                "final_train_loss": round(losses[-1], 6),
                "num_epochs": num_epochs,
                "dataset_hash": tracker.dataset_hash,
            }
        )

        fig, ax = plt.subplots()
        ax.plot(range(1, num_epochs + 1), losses, marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train_loss")
        ax.set_title(config["experiment_name"])
        fig.savefig(tracker.plots_dir / "train_loss.png")
        plt.close(fig)

        tracker.logger.info("dummy experiment complete run_dir=%s", tracker.run_dir)


if __name__ == "__main__":
    main()
