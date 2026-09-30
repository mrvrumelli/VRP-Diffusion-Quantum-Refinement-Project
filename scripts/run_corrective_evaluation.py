"""Bounded, resumable checkpoint comparisons with graph records and source audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.eval.comparison import (
    InstanceResult,
    assert_disjoint_sources,
    metric_value,
    paired_bootstrap,
)
from vrp_diffusion_quantum.eval.routing import evaluate_decoded_matrix
from vrp_diffusion_quantum.inference.policy_support import sample_constraint_matrix_batch
from vrp_diffusion_quantum.inference.predict_matrix import (
    example_to_model_inputs,
    load_denoiser_checkpoint,
)
from vrp_diffusion_quantum.models.constraint_denoiser import ConstraintDenoiser
from vrp_diffusion_quantum.models.diffusion import BernoulliDiffusionSchedule

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    """Hash a file without allocating its full contents."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checkpoint_registry() -> dict[str, Path]:
    """Explicit old checkpoints: never silently choose the newest model."""
    registry: dict[str, Path] = {}
    for name, stamp in {
        "full": "20260928T090607233005Z",
        "pilot": "20260925T003139960628Z",
        "unfiltered": "20260928T122603886201Z",
        "n20only": "20260929T054533775177Z",
    }.items():
        registry[name] = ROOT / (
            f"outputs/paper_cmd/diffusion_denoiser_paper_cmd_{name}_{stamp}/checkpoints/best.pt"
        )
    for size, stamp in {20: "145222731562", 50: "150806474465", 100: "152405895038"}.items():
        registry[f"champion_n{size}"] = ROOT / (
            f"outputs/train/diffusion_denoiser_s7799_stochastic_persize_n{size}_cuda_"
            f"20260816T{stamp}Z/checkpoints/best.pt"
        )
    for size, count in ((20, 1000), (50, 500), (100, 1000)):
        matches = list(
            (ROOT / "outputs/train").glob(
                f"diffusion_denoiser_s7799_stochastic_task5_curve_n{size}_{count}_cuda_*/checkpoints/best.pt"
            )
        )
        if len(matches) != 1:
            raise ValueError(f"ambiguous/missing Task 5 checkpoint: {size}/{count}")
        registry[f"candidate_n{size}"] = matches[0]
    return registry


def dataset_ids(directory: Path) -> set[str]:
    """Use an explicit split manifest, otherwise read actual source IDs from examples."""
    manifest = directory.parent / "split_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        split = payload.get("splits", {}).get(directory.name)
        if split is not None:
            return {entry["instance_id"] for entry in split["examples"]}
    return {
        load_example(path).instance.instance_id
        for path in directory.glob("*.json")
        if not path.name.endswith("manifest.json")
    }


def build_panel(output: Path, per_size: int, seed: int) -> dict[str, Any]:
    """Audit the union of denoiser/GAT training sources before selecting development graphs."""
    registry = checkpoint_registry()
    pending = list(registry.values())
    seen: set[Path] = set()
    inventories: dict[str, Any] = {}
    training: set[str] = set()
    while pending:
        checkpoint = pending.pop()
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        config = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text())
        directory = ROOT / config["dataset"]["path"]
        key = directory.relative_to(ROOT).as_posix()
        if key not in inventories:
            ids = dataset_ids(directory)
            training.update(ids)
            inventories[key] = {"unique_sources": len(ids), "source_ids": sorted(ids)}
            LOGGER.info("Audited %s: %d sources", key, len(ids))
        gat = config.get("model", {}).get("gat_checkpoint")
        if gat:
            pending.append(ROOT / gat)
    source = ROOT / "data/processed/s7799_val100_policy_v1"
    candidates = [
        path for path in sorted(source.glob("*.json")) if not path.name.endswith("manifest.json")
    ]
    by_size: dict[int, list[Path]] = {}
    for path in candidates:
        example = load_example(path)
        assert_disjoint_sources(training, {example.instance.instance_id})
        by_size.setdefault(example.instance.n_customers, []).append(path)
    entries = []
    for size, paths in sorted(by_size.items()):
        rng = np.random.default_rng(np.random.SeedSequence([seed, size]))
        for index in rng.permutation(len(paths))[:per_size]:
            path = paths[int(index)]
            example = load_example(path)
            entries.append(
                {
                    "file": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                    "instance_id": example.instance.instance_id,
                    "n_customers": size,
                    "label_solver": example.solution.solver_name,
                }
            )
    payload = {
        "schema_version": 1,
        "purpose": "development; previously used for model selection; NOT untouched test",
        "selection_seed": seed,
        "per_size": per_size,
        "primary_f1": "pooled ordered off-diagonal positive-class F1 at fixed threshold 0.5",
        "primary_route_metric": "mean_instance_gap_percent",
        "batch_size": 1,
        "precision": "float32",
        "training_inventories": inventories,
        "checkpoints": {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for key, path in registry.items()
        },
        "gat_and_denoiser_checkpoints": [
            path.relative_to(ROOT).as_posix() for path in sorted(seen)
        ],
        "examples": entries,
    }
    (output / "panel.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def evaluate(
    output: Path,
    panel: dict[str, Any],
    name: str,
    steps: int,
    sampler: str,
    seed: int,
    *,
    model_override: ConstraintDenoiser | None = None,
) -> None:
    """Persist per-instance counts, predictions, costs and hashes before aggregation."""
    run = output / f"{name}_{sampler}_steps{steps}_seed{seed}"
    run.mkdir(exist_ok=True)
    checkpoint = ROOT / panel["checkpoints"][name]["path"]
    if sha256(checkpoint) != panel["checkpoints"][name]["sha256"]:
        raise ValueError("checkpoint changed since panel was frozen")
    model, payload = load_denoiser_checkpoint(checkpoint, device="cuda")
    if model_override is not None:
        model = model_override
    config = (payload.get("extra") or {}).get("schedule") or {}
    schedule = BernoulliDiffusionSchedule(**config).to("cuda")
    sizes = (
        {int(name.split("_n")[-1])}
        if "_n" in name
        else ({20} if name == "n20only" else {20, 50, 100})
    )
    records = []
    for entry in panel["examples"]:
        if entry["n_customers"] not in sizes:
            continue
        path = ROOT / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("panel example changed")
        identity = entry["instance_id"]
        record_path = run / f"{identity}.json"
        if record_path.exists():
            records.append(InstanceResult(**json.loads(record_path.read_text())))
            continue
        example = load_example(path)
        coords, demands, capacity, target, mask = example_to_model_inputs(
            example, device="cuda", coordinate_frame=model.coordinate_frame
        )
        graph_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:4], "big"
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = sample_constraint_matrix_batch(
            model,
            schedule,
            coords=coords,
            demands=demands,
            capacity=capacity,
            customer_mask=mask,
            generator=torch.Generator().manual_seed(graph_seed),
            num_inference_steps=steps,
            sampler=sampler,  # type: ignore[arg-type]
        )
        torch.cuda.synchronize()
        hard = prediction.m_hat[0].cpu().numpy()
        soft = prediction.m_prob[0].cpu().numpy()
        truth = target[0].cpu().numpy()
        off_diagonal = ~np.eye(len(hard), dtype=bool)
        positive = (hard >= 0.5) & off_diagonal
        actual = (truth >= 0.5) & off_diagonal
        routing = evaluate_decoded_matrix(example, soft)
        record = InstanceResult(
            identity,
            example.instance.n_customers,
            int((positive & actual).sum()),
            int((positive & ~actual).sum()),
            int((~positive & actual).sum()),
            routing.decoded_cost,
            routing.reference_cost,
            routing.feasible,
            time.perf_counter() - start,
        )
        np.savez_compressed(run / f"{identity}.npz", probability=soft, target=truth)
        record_path.write_text(json.dumps(asdict(record), indent=2) + "\n")
        records.append(record)
        LOGGER.info("%s %d/%d", run.name, len(records), len(panel["examples"]))
    summary: dict[str, Any] = {"checkpoint": name, "sampler": sampler, "steps": steps, "seed": seed}
    for metric in ("f1", "mean_instance_gap_percent", "ratio_total_gap_percent"):
        summary[metric] = asdict(paired_bootstrap(records, metric=metric))  # type: ignore[arg-type]
    summary["f1_by_size"] = {
        str(size): metric_value([record for record in records if record.n_customers == size], "f1")
        for size in sorted(sizes)
    }
    summary["equal_size_mean_f1"] = float(np.mean(list(summary["f1_by_size"].values())))
    summary["feasibility_rate"] = sum(record.feasible for record in records) / len(records)
    summary["total_runtime_seconds"] = sum(record.runtime_seconds for record in records)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info(
        "Completed %s F1=%.6f gap=%.3f",
        run.name,
        summary["f1"]["estimate"],
        summary["mean_instance_gap_percent"]["estimate"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/corrective_20260930"))
    parser.add_argument("--models", nargs="+", default=["pilot", "full", "unfiltered", "n20only"])
    parser.add_argument("--steps", nargs="+", type=int, default=[50])
    parser.add_argument(
        "--samplers", nargs="+", default=["skipped_posterior", "posterior_mixture_v2"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--per-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log")],
    )
    torch.set_num_threads(2)
    panel_path = args.output / "panel.json"
    panel = (
        json.loads(panel_path.read_text())
        if panel_path.exists()
        else build_panel(args.output, args.per_size, 93817)
    )
    if panel["per_size"] != args.per_size:
        raise ValueError("use a new output directory to change panel size")
    for name in args.models:
        for steps in args.steps:
            for sampler in args.samplers:
                for seed in args.seeds:
                    evaluate(args.output, panel, name, steps, sampler, seed)


if __name__ == "__main__":
    main()
