"""Create a standalone figure showing the scaling stop and OOD route comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    destination = Path("docs/evidence/corrective_20260930")
    results = json.loads((destination / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for seed, color in ((4331, "#2166ac"), (4332, "#b2182b")):
        intervals = [
            results["curve_runs"][f"{count}_{seed}"]["evaluation"]["mean_instance_gap_percent"]
            for count in (500, 1000)
        ]
        values = np.array([row["estimate"] for row in intervals])
        errors = np.array(
            [
                [row["estimate"] - row["lower"] for row in intervals],
                [row["upper"] - row["estimate"] for row in intervals],
            ]
        )
        axes[0].errorbar(
            [500, 1000],
            values,
            yerr=errors,
            marker="o",
            capsize=4,
            color=color,
            label=f"Training seed {seed}",
        )
    axes[0].set(
        title="N100: five-epoch source-count curve",
        xlabel="Distinct training sources",
        ylabel="Mean instance-relative route gap (%)",
        xticks=[500, 1000],
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    names = list(results["ood_cells"])
    intervals = [results["ood_cells"][name]["paired_gap"] for name in names]
    values = np.array([row["estimate"] for row in intervals])
    errors = np.array(
        [
            [row["estimate"] - row["lower"] for row in intervals],
            [row["upper"] - row["estimate"] for row in intervals],
        ]
    )
    axes[1].errorbar(
        values, np.arange(len(names)), xerr=errors, fmt="o", capsize=3, color="#6a3d9a"
    )
    axes[1].axvline(0, color="0.5", linestyle="--", linewidth=1)
    axes[1].set(
        yticks=np.arange(len(names)),
        yticklabels=names,
        title="OOD: merged minus champion",
        xlabel="Route-gap difference (percentage points)",
    )
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Development evidence: lower route gap is better\n"
        "95% graph-bootstrap intervals; corrected sampler, 50 steps",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"route_evidence.{suffix}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
