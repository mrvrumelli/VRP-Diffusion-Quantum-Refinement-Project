"""Train MatrixPredictor on a small CVRP dataset (task P2.1) and score it (task P2.2).

Proves the supervised constraint-matrix predictor trains end to end, outputs valid M
probabilities, and that matrix metrics run on a validation batch, per AGENTS.md Phase 2's done
criteria. On data/samples/sanity_cvrp (2 hand-crafted examples, evaluated on the same examples
used for training) this only proves the code path runs -- it is not a meaningful AUC/F1 result
on held-out data, which needs a real dataset and train/val split from P1.1/P1.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from vrp_diffusion_quantum.data.dataset import load_dataset
from vrp_diffusion_quantum.metrics.matrix_metrics import MatrixPrediction, compute_matrix_metrics
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor, train_matrix_predictor
from vrp_diffusion_quantum.utils.experiment import ExperimentTracker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "train" / "matrix_predictor_sanity.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["seed"])
    torch.manual_seed(seed)

    dataset_path = ROOT / config["dataset"]["path"]
    output_root = ROOT / config["output"]["root"]
    hidden_dim = int(config["model"]["hidden_dim"])
    num_epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])

    examples = load_dataset(dataset_path)
    if not examples:
        raise ValueError(f"no examples found under {dataset_path}")

    model = MatrixPredictor(hidden_dim=hidden_dim)

    with ExperimentTracker(
        output_root=output_root,
        experiment_name=config["experiment_name"],
        config=config,
        seed=seed,
        dataset_path=dataset_path,
    ) as tracker:
        epoch_losses = train_matrix_predictor(
            model, examples, num_epochs=num_epochs, learning_rate=learning_rate
        )
        for epoch, loss in enumerate(epoch_losses):
            tracker.log_metric_row({"epoch": epoch, "matrix_bce_loss": loss})

        model.eval()
        with torch.no_grad():
            predictions = [
                MatrixPrediction.from_example(
                    example,
                    model(
                        torch.from_numpy(example.instance.customer_coords()).float(),
                        torch.from_numpy(example.instance.customer_demands()).float(),
                        example.instance.capacity,
                    ).numpy(),
                )
                for example in examples
            ]
        matrix_metrics = compute_matrix_metrics(predictions)

        tracker.log_metrics(
            {
                "final_matrix_bce_loss": epoch_losses[-1],
                "num_epochs": num_epochs,
                "num_examples": len(examples),
                "dataset_hash": tracker.dataset_hash,
                "val_bce": matrix_metrics.bce,
                "val_auc": matrix_metrics.auc,
                "val_precision": matrix_metrics.precision,
                "val_recall": matrix_metrics.recall,
                "val_f1": matrix_metrics.f1,
                "val_calibration_error": matrix_metrics.calibration_error,
                "val_capacity_consistency": matrix_metrics.capacity_consistency,
            }
        )
        tracker.logger.info("matrix predictor training complete run_dir=%s", tracker.run_dir)

    print(f"final matrix_bce_loss={epoch_losses[-1]:.6f} over {len(examples)} example(s)")
    print(
        f"val: bce={matrix_metrics.bce:.4f} auc={matrix_metrics.auc} "
        f"precision={matrix_metrics.precision:.4f} recall={matrix_metrics.recall:.4f} "
        f"f1={matrix_metrics.f1:.4f} calibration_error={matrix_metrics.calibration_error:.4f} "
        f"capacity_consistency={matrix_metrics.capacity_consistency:.4f}"
    )


if __name__ == "__main__":
    main()
