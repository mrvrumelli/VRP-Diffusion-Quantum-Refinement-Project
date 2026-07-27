"""Save routes + constraint-matrix sanity-check plots for a dataset directory (task P1.5).

For each example under `--dataset-dir`, saves one PNG (routes next to `constraint_matrix`) under
`--output-dir` so a human can check by eye that the two agree, per `AGENTS.md` Phase 1's done
criterion for this task.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "data" / "samples" / "sanity_cvrp"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "sanity_plots"

logger = logging.getLogger(__name__)


def configure_plot_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "vrp_diffusion_quantum_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Plot at most this many examples (default: all examples in the dataset).",
    )
    return parser.parse_args()


def main() -> None:
    configure_plot_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrp_diffusion_quantum.data.dataset import load_dataset
    from vrp_diffusion_quantum.eval.visualize import plot_example_sanity_check

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    examples = load_dataset(args.dataset_dir)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError(f"no examples found under {args.dataset_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for example in examples:
        fig = plot_example_sanity_check(example)
        output_path = args.output_dir / f"{example.instance.instance_id}.png"
        fig.savefig(output_path)
        plt.close(fig)
        logger.info(
            "saved sanity plot instance_id=%s path=%s", example.instance.instance_id, output_path
        )

    print(f"saved {len(examples)} sanity plot(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
