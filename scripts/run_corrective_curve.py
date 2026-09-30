"""Execute the four predeclared source-count runs, then apply their paired stop rule."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from run_corrective_evaluation import evaluate, sha256
from vrp_diffusion_quantum.eval.comparison import (
    InstanceResult,
    assert_disjoint_sources,
    paired_bootstrap,
)
from vrp_diffusion_quantum.models.gat_encoder import compat_layernorm_state_dict


def main() -> None:
    root = Path("outputs/corrective_curve_20260930")
    root.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(root / "run.log")],
    )
    torch.set_num_threads(2)
    panel = json.loads(Path("outputs/corrective_confirmation_20260930/panel.json").read_text())
    panel["checkpoints"] = {}
    for count in (500, 1000):
        dataset_path = Path(f"data/processed/corrective_20260930/curve_n100_{count}")
        manifest_path = dataset_path / "subset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        identities = {entry["instance_id"] for entry in manifest["examples"]}
        assert_disjoint_sources(identities, {entry["instance_id"] for entry in panel["examples"]})
        if len(identities) != count:
            raise ValueError("curve source count differs from declared protocol")
        panel["training_inventories"][dataset_path.as_posix()] = {
            "unique_sources": count,
            "source_ids": sorted(identities),
            "manifest_sha256": sha256(manifest_path),
        }
        for seed in (4331, 4332):
            config_name = f"corrective_curve_n100_{count}_s{seed}"
            paths = list(root.glob(f"{config_name}_*/checkpoints/last.pt"))
            if not paths:
                log_path = root / f"{config_name}.log"
                with log_path.open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "vrp_diffusion_quantum.train.train_diffusion",
                            "--config",
                            f"configs/train/{config_name}.yaml",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                    )
                paths = list(root.glob(f"{config_name}_*/checkpoints/last.pt"))
            if len(paths) != 1:
                raise ValueError("ambiguous/missing curve run; inspect artifacts before resuming")
            checkpoint = paths[0]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["epoch"] != 4:  # Trainer checkpoints use zero-based epoch indices.
                raise ValueError("curve run did not complete the declared five epochs")
            config_path = checkpoint.parent.parent / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            gat_path = Path(config["model"]["gat_checkpoint"])
            gat_payload = torch.load(gat_path, map_location="cpu", weights_only=False)
            expected = compat_layernorm_state_dict(gat_payload["encoder"])
            actual = {
                key.removeprefix("node_encoder."): value
                for key, value in payload["model"].items()
                if key.startswith("node_encoder.")
            }
            if expected.keys() != actual.keys() or not all(
                torch.equal(value, actual[key]) for key, value in expected.items()
            ):
                raise ValueError("frozen GAT changed during source-count training")
            name = f"curve{count}_s{seed}_n100"
            panel["checkpoints"][name] = {
                "path": checkpoint.as_posix(),
                "sha256": sha256(checkpoint),
                "config_sha256": sha256(config_path),
                "frozen_gat_sha256": sha256(gat_path),
                "frozen_gat_exactly_preserved": True,
            }
            logging.info("Completed training %s", config_name)
    (root / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    for name in panel["checkpoints"]:
        evaluate(root, panel, name, 50, "posterior_mixture_v2", 9001)
    decision = {"protocol": "docs/corrective_curve_protocol_2026-09-30.md", "seed_results": {}}
    for seed in (4331, 4332):
        panels = {}
        for count in (500, 1000):
            directory = root / f"curve{count}_s{seed}_n100_posterior_mixture_v2_steps50_seed9001"
            panels[count] = [
                InstanceResult(**json.loads(path.read_text()))
                for path in directory.glob("*.json")
                if path.name != "summary.json"
            ]
        interval = paired_bootstrap(
            panels[1000], panels[500], metric="mean_instance_gap_percent", num_resamples=10000
        )
        decision["seed_results"][str(seed)] = {
            "paired_gap_difference": asdict(interval),
            "passes": interval.estimate <= -1 and interval.upper < 0,
        }
    decision["expand"] = all(result["passes"] for result in decision["seed_results"].values())
    decision["action"] = (
        "audit reserved test references before finalist evaluation"
        if decision["expand"]
        else "stop this bounded curve; no independently justified finalist; preserve reserved test"
    )
    (root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    logging.info("Curve decision: %s", decision["action"])


if __name__ == "__main__":
    main()
