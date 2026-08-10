"""Pretrain GAT encoder on same-route ``M``; save encoder-only ckpt.

python scripts/pretrain_gat_encoder.py --config configs/train/gat_pretrain.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from vrp_diffusion_quantum.data.augment import AUGMENT_NUM, augment_example
from vrp_diffusion_quantum.data.dataset import collate_batch, load_dataset, size_homogeneous_chunks
from vrp_diffusion_quantum.metrics.matrix_metrics import MatrixPrediction, compute_matrix_metrics
from vrp_diffusion_quantum.models.gat_encoder import (
    GATConstraintPretrainer,
    save_gat_encoder_checkpoint,
)
from vrp_diffusion_quantum.train.train_diffusion import (
    customer_tensors_from_batch,
    diffusion_matrix_bce_loss,
)
from vrp_diffusion_quantum.utils.experiment import ExperimentTracker
from vrp_diffusion_quantum.utils.runtime import default_mlflow_tracking_uri, resolve_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "train" / "gat_pretrain.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["seed"])
    torch.manual_seed(seed)

    train_path = ROOT / config["dataset"]["path"]
    val_path = ROOT / config["validation"]["path"]
    train_examples = load_dataset(train_path)
    val_examples = load_dataset(val_path)
    if not train_examples:
        raise ValueError(f"no train examples under {train_path}")
    if not val_examples:
        raise ValueError(f"no val examples under {val_path}")

    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config.get("mlflow") or {}
    device = resolve_device(train_cfg.get("device", "auto"))
    model = GATConstraintPretrainer(
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        gat_num_layers=int(model_cfg.get("gat_num_layers", 5)),
        gat_num_heads=int(model_cfg.get("gat_num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    augmentation = bool(train_cfg.get("augmentation", False))
    online_augmentation = bool(train_cfg.get("online_augmentation", False)) and not augmentation
    same_size_batches = bool(train_cfg.get("same_size_batches", False))
    weighted_bce = bool(train_cfg.get("weighted_bce", False))
    pos_weight_power = float(train_cfg.get("pos_weight_power", 0.5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg["learning_rate"])
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unsupported optimizer {optimizer_name!r}")
    batch_size = int(train_cfg.get("batch_size", 32))
    num_epochs = int(train_cfg["epochs"])
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0) or 0)
    adaptive_threshold = bool(train_cfg.get("adaptive_threshold", True))
    decision_threshold = train_cfg.get("decision_threshold")
    decision_threshold = None if decision_threshold is None else float(decision_threshold)
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True))

    mlflow = None
    if mlflow_enabled:
        import mlflow as mlflow_mod

        mlflow = mlflow_mod
        tracking_uri = str(mlflow_cfg.get("tracking_uri") or default_mlflow_tracking_uri())
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(str(mlflow_cfg.get("experiment_name") or config["experiment_name"]))

    with ExperimentTracker(
        output_root=ROOT / config["output"]["root"],
        experiment_name=config["experiment_name"],
        config=config,
        seed=seed,
        dataset_path=train_path,
    ) as tracker:
        best_auc = float("-inf")
        epochs_without_improve = 0
        best_path = tracker.run_dir / "checkpoints" / "gat_encoder_best.pt"
        last_path = tracker.run_dir / "checkpoints" / "gat_encoder_last.pt"
        run_ctx = mlflow.start_run(run_name=tracker.run_dir.name) if mlflow is not None else None
        try:
            if mlflow is not None:
                mlflow.log_param("device", str(device))
                mlflow.log_param("run_dir", str(tracker.run_dir))
                for key, value in model_cfg.items():
                    mlflow.log_param(f"model.{key}", value)
                for key, value in train_cfg.items():
                    mlflow.log_param(f"training.{key}", value)

            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                n_batches = 0
                if same_size_batches or augmentation:
                    ordered = list(train_examples)
                else:
                    order = torch.randperm(
                        len(train_examples),
                        generator=torch.Generator().manual_seed(seed + epoch),
                    )
                    ordered = [train_examples[int(i)] for i in order]
                if online_augmentation:
                    aug_gen = torch.Generator().manual_seed(seed + 17_000 + epoch)
                    vs = torch.randint(0, AUGMENT_NUM, (len(ordered),), generator=aug_gen)
                    ordered = [
                        augment_example(ex, int(v))
                        for ex, v in zip(ordered, vs.tolist(), strict=True)
                    ]
                batch_gen = torch.Generator().manual_seed(seed + 3_000 + epoch)
                if same_size_batches or augmentation:
                    chunks = size_homogeneous_chunks(
                        ordered,
                        batch_size,
                        generator=batch_gen,
                        shuffle=True,
                        augmentation=augmentation,
                    )
                else:
                    chunks = (
                        ordered[start : start + batch_size]
                        for start in range(0, len(ordered), batch_size)
                    )
                for chunk in chunks:
                    batch = collate_batch(chunk)
                    coords, demands, capacity = customer_tensors_from_batch(batch)
                    coords = coords.to(device)
                    demands = demands.to(device)
                    capacity = capacity.to(device)
                    mask = batch.customer_mask.to(device)
                    m_true = batch.constraint_matrix.to(device)
                    optimizer.zero_grad()
                    logits = model(coords, demands, capacity, customer_mask=mask)
                    loss = diffusion_matrix_bce_loss(
                        logits,
                        m_true,
                        mask,
                        weighted=weighted_bce,
                        pos_weight_power=pos_weight_power,
                    )
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.item())
                    n_batches += 1
                train_loss /= max(n_batches, 1)

                model.eval()
                val_loss = 0.0
                v_batches = 0
                predictions: list[MatrixPrediction] = []
                with torch.no_grad():
                    if same_size_batches or augmentation:
                        val_chunks = size_homogeneous_chunks(
                            val_examples, batch_size, shuffle=False, augmentation=False
                        )
                    else:
                        val_chunks = (
                            val_examples[start : start + batch_size]
                            for start in range(0, len(val_examples), batch_size)
                        )
                    for chunk in val_chunks:
                        batch = collate_batch(chunk)
                        coords, demands, capacity = customer_tensors_from_batch(batch)
                        coords = coords.to(device)
                        demands = demands.to(device)
                        capacity = capacity.to(device)
                        mask = batch.customer_mask.to(device)
                        m_true = batch.constraint_matrix.to(device)
                        logits = model(coords, demands, capacity, customer_mask=mask)
                        val_loss += float(
                            diffusion_matrix_bce_loss(
                                logits,
                                m_true,
                                mask,
                                weighted=weighted_bce,
                                pos_weight_power=pos_weight_power,
                            ).item()
                        )
                        v_batches += 1
                        probs = torch.sigmoid(logits).detach().cpu().numpy()
                        for i, example in enumerate(chunk):
                            n = example.instance.n_customers
                            predictions.append(
                                MatrixPrediction.from_example(
                                    example, probs[i, :n, :n].astype(np.float64)
                                )
                            )
                val_loss /= max(v_batches, 1)
                val_metrics = compute_matrix_metrics(
                    predictions,
                    threshold=decision_threshold,
                    adaptive_threshold=adaptive_threshold,
                )
                val_auc = float(val_metrics.auc) if val_metrics.auc is not None else float("nan")

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_bce": val_metrics.bce,
                    "val_auc": val_metrics.auc,
                    "val_precision": val_metrics.precision,
                    "val_recall": val_metrics.recall,
                    "val_f1": val_metrics.f1,
                    "val_accuracy": val_metrics.accuracy,
                    "val_threshold": val_metrics.threshold,
                }
                tracker.log_metric_row(row)
                tracker.logger.info(
                    "epoch=%d train_loss=%.6f val_loss=%.6f val_auc=%s "
                    "val_f1=%.4f val_acc=%.4f val_prec=%.4f val_rec=%.4f thr=%.2f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics.auc,
                    val_metrics.f1,
                    val_metrics.accuracy,
                    val_metrics.precision,
                    val_metrics.recall,
                    val_metrics.threshold,
                )
                if mlflow is not None:
                    for key, value in row.items():
                        if key == "epoch":
                            continue
                        if isinstance(value, (int, float)) and value == value:
                            mlflow.log_metric(key, float(value), step=epoch)

                extra = {
                    "experiment_name": config["experiment_name"],
                    "seed": seed,
                    "model": model_cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_auc": val_metrics.auc,
                    "val_f1": val_metrics.f1,
                    "val_accuracy": val_metrics.accuracy,
                    "val_precision": val_metrics.precision,
                    "val_threshold": val_metrics.threshold,
                }
                save_gat_encoder_checkpoint(last_path, model.encoder, extra=extra)
                improved = val_metrics.auc is not None and float(val_metrics.auc) > best_auc
                if improved:
                    best_auc = float(val_auc)
                    epochs_without_improve = 0
                    save_gat_encoder_checkpoint(best_path, model.encoder, extra=extra)
                else:
                    epochs_without_improve += 1
                    if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
                        tracker.logger.info(
                            "early stop at epoch=%d (patience=%d, best_val_auc=%.4f)",
                            epoch,
                            early_stop_patience,
                            best_auc,
                        )
                        break

            if mlflow is not None:
                mlflow.log_metric("best_val_auc", best_auc)
                if best_path.is_file():
                    mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        finally:
            if run_ctx is not None and mlflow is not None:
                mlflow.end_run()

        print(f"GAT pretrain done. best_val_auc={best_auc:.4f}")
        print(f"encoder checkpoint: {best_path}")


if __name__ == "__main__":
    main()
