"""Validate R/C/RC stress-set separation and save durable plots/metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrp_diffusion_quantum.data.generate_cvrp import load_dataset
from vrp_diffusion_quantum.eval.spatial import spatial_metrics, summarize_spatial_metrics
from vrp_diffusion_quantum.utils.experiment import hash_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-dir", type=Path, required=True)
    parser.add_argument("--c-dir", type=Path, required=True)
    parser.add_argument("--rc-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--silhouette-sample", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    regimes = {"R": args.r_dir, "C": args.c_dir, "RC": args.rc_dir}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    representatives: dict[tuple[str, int], np.ndarray] = {}
    for regime, root in regimes.items():
        for size in (20, 50, 100):
            dataset = load_dataset(root / f"cvrp{size}")
            per_instance = []
            nearest = []
            for index in range(len(dataset)):
                metrics = spatial_metrics(
                    dataset.coords[index, 1:],
                    max_clusters=8 if index < args.silhouette_sample else 1,
                )
                if index >= args.silhouette_sample:
                    metrics.pop("best_kmeans_silhouette")
                per_instance.append(metrics)
                nearest.append(metrics["mean_nearest_neighbor"])
            base_rows = [
                {key: value for key, value in row.items() if key != "best_kmeans_silhouette"}
                for row in per_instance
            ]
            summary = summarize_spatial_metrics(base_rows)
            silhouettes = [
                row["best_kmeans_silhouette"]
                for row in per_instance
                if "best_kmeans_silhouette" in row
            ]
            summary["best_kmeans_silhouette_mean"] = float(np.mean(silhouettes))
            summary["best_kmeans_silhouette_std"] = float(np.std(silhouettes, ddof=1))
            nearest_array = np.asarray(nearest)
            representative_index = int(np.argmin(np.abs(nearest_array - np.median(nearest_array))))
            representatives[(regime, size)] = dataset.coords[representative_index]
            rows.append(
                {
                    "regime": regime,
                    "n_customers": size,
                    "num_instances": len(dataset),
                    "representative_index": representative_index,
                    **summary,
                }
            )

    with (args.output_dir / "spatial_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "inputs": {
            regime: {"path": path.as_posix(), "sha256": hash_dataset(path)}
            for regime, path in regimes.items()
        },
        "silhouette_sample_per_size": args.silhouette_sample,
        "rows": rows,
    }
    (args.output_dir / "spatial_metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for row, regime in enumerate(("R", "C", "RC")):
        for column, size in enumerate((20, 50, 100)):
            coords = representatives[(regime, size)]
            ax = axes[row, column]
            ax.scatter(coords[1:, 0], coords[1:, 1], s=8, alpha=0.75)
            ax.scatter(coords[0, 0], coords[0, 1], marker="*", s=70, c="red")
            ax.set_title(f"{regime} / CVRP{size}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.output_dir / "representative_instances.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
