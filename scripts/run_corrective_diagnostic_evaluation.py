"""Export graph-level uncertainty for the completed small learning diagnostics."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from run_corrective_evaluation import evaluate, sha256
from vrp_diffusion_quantum.eval.comparison import assert_disjoint_sources


def main() -> None:
    output = Path("outputs/corrective_diagnostic_evaluation_20260930")
    output.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    panel = json.loads(Path("outputs/corrective_20260930/panel.json").read_text())
    panel["checkpoints"] = {}
    for root, prefix in (
        (Path("outputs/corrective_learning_20260930"), "mixed"),
        (Path("outputs/corrective_normalization_n100_20260930"), "single"),
    ):
        for checkpoint in sorted(root.glob("*/last.pt")):
            if checkpoint.parent.name.startswith("tiny"):
                continue
            config = json.loads((checkpoint.parent / "config.json").read_text())
            assert_disjoint_sources(
                set(config["train_sources"]), {entry["instance_id"] for entry in panel["examples"]}
            )
            name = f"{prefix}_{checkpoint.parent.name}"
            if prefix == "single":
                name += "_n100"
            panel["checkpoints"][name] = {
                "path": checkpoint.as_posix(),
                "sha256": sha256(checkpoint),
                "config": config,
            }
    panel["sampling_note"] = (
        "Additional graph-ID-seeded scoring, seed 9001; earlier diagnostic summaries used "
        "position-derived seeds. Compare arms only within the same scoring protocol."
    )
    (output / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    for name in panel["checkpoints"]:
        evaluate(output, panel, name, 50, "posterior_mixture_v2", 9001)


if __name__ == "__main__":
    main()
