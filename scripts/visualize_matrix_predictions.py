"""Train on Phase 1 examples and plot predicted versus true M on held-out examples."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import torch

from vrp_diffusion_quantum.data.dataset import load_dataset, split_examples
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor, train_matrix_predictor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "data" / "samples" / "sanity_cvrp"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "matrix_prediction_examples"

logger = logging.getLogger(__name__)


def configure_plot_cache() -> None:
    """Keep Matplotlib cache files outside the repository."""
    cache_root = Path(tempfile.gettempdir()) / "vrp_diffusion_quantum_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Phase 1 examples/ directory containing CVRPExample JSON files",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-plots", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    configure_plot_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrp_diffusion_quantum.eval.visualize import plot_matrix_comparison

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    examples = load_dataset(args.dataset_dir)
    if not examples:
        raise ValueError(f"no examples found under {args.dataset_dir}")
    split = split_examples(
        examples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if not split.validation:
        raise ValueError("--validation-fraction must produce a non-empty held-out split")

    torch.manual_seed(args.seed)
    model = MatrixPredictor(hidden_dim=args.hidden_dim)
    train_matrix_predictor(
        model,
        split.train,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    held_out_examples = split.validation[: args.max_plots]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for example in held_out_examples:
            m_prob = model(
                torch.from_numpy(example.instance.customer_coords()).float(),
                torch.from_numpy(example.instance.customer_demands()).float(),
                example.instance.capacity,
            ).numpy()
            fig = plot_matrix_comparison(
                example.constraint_matrix,
                m_prob,
                title=(
                    f"{example.instance.instance_id} (n_customers={example.instance.n_customers})"
                ),
            )
            output_path = args.output_dir / f"{example.instance.instance_id}.png"
            fig.savefig(output_path)
            plt.close(fig)
            logger.info("saved held-out comparison instance_id=%s", example.instance.instance_id)

    print(
        f"saved {len(held_out_examples)} held-out plot(s) to {args.output_dir}; "
        f"trained on {len(split.train)} Phase 1 example(s)"
    )


if __name__ == "__main__":
    main()
