"""Bounded nine-cell R/C/RC checkpoint comparison on stable held-out labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from run_corrective_evaluation import checkpoint_registry, dataset_ids, evaluate, sha256
from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.eval.comparison import assert_disjoint_sources


def main() -> None:
    output = Path("outputs/corrective_ood_20260930")
    output.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    torch.set_num_threads(2)
    panel_path = output / "panel.json"
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
    else:
        registry = {
            key: path for key, path in checkpoint_registry().items() if key.startswith("champion")
        }
        for size, stamp in {20: "204918783220", 50: "211322935348", 100: "213809850801"}.items():
            registry[f"merged_n{size}"] = Path(
                f"outputs/train/diffusion_denoiser_s7799_rcfull_persize_n{size}_cuda_20260824T{stamp}Z/checkpoints/best.pt"
            ).resolve()
        train_ids: set[str] = set()
        inventories = {}
        pending = list(registry.values())
        seen = set()
        while pending:
            checkpoint = pending.pop()
            if checkpoint in seen:
                continue
            seen.add(checkpoint)
            config = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text())
            directory = config["dataset"]["path"]
            if directory not in inventories:
                ids = dataset_ids(Path(directory))
                inventories[directory] = sorted(ids)
                train_ids.update(ids)
            if config.get("model", {}).get("gat_checkpoint"):
                pending.append(Path(config["model"]["gat_checkpoint"]).resolve())
        entries = []
        coverage = {}
        source = Path("outputs/label_audit/rc_full_eval/accepted_matrix_examples")
        for regime_index, regime in enumerate(("r", "c", "rc")):
            for size in (20, 50, 100):
                paths = sorted(source.glob(f"{regime}_cvrp{size}_*.json"))
                coverage[f"{regime}_n{size}"] = {
                    "accepted": len(paths),
                    "original": 1000,
                    "evaluated": 8,
                }
                rng = np.random.default_rng(np.random.SeedSequence([93829, regime_index, size]))
                for index in rng.permutation(len(paths))[:8]:
                    path = paths[int(index)]
                    example = load_example(path)
                    assert_disjoint_sources(train_ids, {example.instance.instance_id})
                    entries.append(
                        {
                            "file": path.as_posix(),
                            "sha256": sha256(path),
                            "instance_id": example.instance.instance_id,
                            "n_customers": size,
                            "regime": regime,
                        }
                    )
        if len(entries) != 72:
            raise ValueError("incomplete nine-cell panel")
        panel = {
            "purpose": "bounded OOD development comparison; stable labels selected before sampling",
            "selection_seed": 93829,
            "batch_size": 1,
            "precision": "float32",
            "per_cell": 8,
            "coverage": coverage,
            "training_inventories": inventories,
            "checkpoints": {
                key: {"path": path.as_posix(), "sha256": sha256(path)}
                for key, path in registry.items()
            },
            "examples": entries,
        }
        panel_path.write_text(json.dumps(panel, indent=2) + "\n")
    for name in panel["checkpoints"]:
        evaluate(output, panel, name, 50, "posterior_mixture_v2", 0)


if __name__ == "__main__":
    main()
