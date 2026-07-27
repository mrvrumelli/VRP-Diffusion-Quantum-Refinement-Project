"""Train the supervised M predictor and evaluate it on deterministic held-out data."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from vrp_diffusion_quantum.data.dataset import load_dataset, split_examples
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.metrics.matrix_metrics import (
    MatrixMetrics,
    MatrixPrediction,
    compute_matrix_metrics,
)
from vrp_diffusion_quantum.models.matrix_baselines import predict_heuristic_matrices
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor, train_matrix_predictor
from vrp_diffusion_quantum.utils.experiment import ExperimentTracker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "train" / "matrix_predictor_sanity.yaml"


def configure_plot_cache() -> None:
    """Keep Matplotlib cache files outside the repository."""
    cache_root = Path(tempfile.gettempdir()) / "vrp_diffusion_quantum_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def model_predictions(
    model: MatrixPredictor, examples: list[CVRPExample]
) -> list[MatrixPrediction]:
    """Run a trained model on examples and return metric-ready predictions."""
    model.eval()
    with torch.no_grad():
        return [
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


def heuristic_predictions(
    examples: list[CVRPExample], *, random_seed: int
) -> dict[str, list[MatrixPrediction]]:
    """Build metric-ready predictions for every deterministic heuristic baseline."""
    predictions: dict[str, list[MatrixPrediction]] = {}
    for example_index, example in enumerate(examples):
        example_seed = int(
            np.random.SeedSequence([random_seed, example_index]).generate_state(1)[0]
        )
        for name, m_prob in predict_heuristic_matrices(example, random_seed=example_seed).items():
            predictions.setdefault(name, []).append(MatrixPrediction.from_example(example, m_prob))
    return predictions


def prefixed_metrics(prefix: str, metrics: MatrixMetrics) -> dict[str, object]:
    """Flatten MatrixMetrics under a method/split prefix for metrics.json."""
    return {f"{prefix}_{name}": value for name, value in asdict(metrics).items()}


def main() -> None:
    configure_plot_cache()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["seed"])
    torch.manual_seed(seed)

    dataset_path = ROOT / config["dataset"]["path"]
    output_root = ROOT / config["output"]["root"]
    hidden_dim = int(config["model"]["hidden_dim"])
    num_epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    split_config = config.get("split", {})
    validation_fraction = float(split_config.get("validation_fraction", 0.2))
    test_fraction = float(split_config.get("test_fraction", 0.0))
    evaluation_config = config.get("evaluation", {})
    threshold = float(evaluation_config.get("threshold", 0.5))
    num_calibration_bins = int(evaluation_config.get("num_calibration_bins", 10))
    random_baseline_seed = int(evaluation_config.get("random_baseline_seed", seed))
    max_plots = int(evaluation_config.get("max_plots", 20))

    examples = load_dataset(dataset_path)
    if not examples:
        raise ValueError(f"no examples found under {dataset_path}")
    split = split_examples(
        examples,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if not split.validation:
        raise ValueError("training config must request a non-empty validation split")

    model = MatrixPredictor(hidden_dim=hidden_dim)
    training_started = time.perf_counter()

    with ExperimentTracker(
        output_root=output_root,
        experiment_name=config["experiment_name"],
        config=config,
        seed=seed,
        dataset_path=dataset_path,
    ) as tracker:
        epoch_losses = train_matrix_predictor(
            model, split.train, num_epochs=num_epochs, learning_rate=learning_rate
        )
        for epoch, loss in enumerate(epoch_losses):
            tracker.log_metric_row({"epoch": epoch, "matrix_bce_loss": loss})

        validation_predictions = model_predictions(model, split.validation)
        validation_metrics = compute_matrix_metrics(
            validation_predictions,
            threshold=threshold,
            num_calibration_bins=num_calibration_bins,
        )
        metrics: dict[str, object] = {
            "final_train_loss": epoch_losses[-1],
            "validation_loss": validation_metrics.bce,
            "num_epochs": num_epochs,
            "num_examples": len(examples),
            "num_train_examples": len(split.train),
            "num_validation_examples": len(split.validation),
            "num_test_examples": len(split.test),
            "dataset_hash": tracker.dataset_hash,
            "learning_rate": learning_rate,
            "seed": seed,
            "random_baseline_seed": random_baseline_seed,
            "feasibility_rate": sum(example.solution.feasible for example in examples)
            / len(examples),
        }
        metrics.update(prefixed_metrics("validation_model", validation_metrics))

        baseline_predictions = heuristic_predictions(
            split.validation,
            random_seed=random_baseline_seed,
        )
        for baseline_name, predictions in baseline_predictions.items():
            baseline_metrics = compute_matrix_metrics(
                predictions,
                threshold=threshold,
                num_calibration_bins=num_calibration_bins,
            )
            metrics.update(
                prefixed_metrics(f"validation_baseline_{baseline_name}", baseline_metrics)
            )

        if split.test:
            test_metrics = compute_matrix_metrics(
                model_predictions(model, split.test),
                threshold=threshold,
                num_calibration_bins=num_calibration_bins,
            )
            metrics.update(prefixed_metrics("test_model", test_metrics))

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "hidden_dim": hidden_dim,
                "seed": seed,
                "dataset_hash": tracker.dataset_hash,
            },
            tracker.run_dir / "model.pt",
        )

        if max_plots > 0:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from vrp_diffusion_quantum.eval.visualize import plot_matrix_comparison

            for example, prediction in zip(
                split.validation[:max_plots],
                validation_predictions[:max_plots],
                strict=True,
            ):
                fig = plot_matrix_comparison(
                    prediction.m_true,
                    prediction.m_prob,
                    title=example.instance.instance_id,
                )
                fig.savefig(tracker.plots_dir / f"{example.instance.instance_id}.png")
                plt.close(fig)

        metrics["runtime_seconds"] = time.perf_counter() - training_started
        tracker.log_metrics(metrics)
        tracker.logger.info("matrix predictor training complete run_dir=%s", tracker.run_dir)

    print(f"final train_loss={epoch_losses[-1]:.6f} over {len(split.train)} training example(s)")
    print(
        f"validation: bce={validation_metrics.bce:.4f} auc={validation_metrics.auc} "
        f"precision={validation_metrics.precision:.4f} recall={validation_metrics.recall:.4f} "
        f"f1={validation_metrics.f1:.4f} "
        f"calibration_error={validation_metrics.calibration_error:.4f} "
        f"capacity_consistency={validation_metrics.capacity_consistency:.4f}"
    )


if __name__ == "__main__":
    main()
