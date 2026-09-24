"""Leakage-safe helpers shared by matrix-predictor evaluation scripts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as ff

from vrp_diffusion_quantum.data.augment import AUGMENT_NUM, expand_examples
from vrp_diffusion_quantum.data.types import CVRPExample
from vrp_diffusion_quantum.metrics.matrix_metrics import MatrixPrediction, compute_matrix_metrics
from vrp_diffusion_quantum.models.matrix_predictor import MatrixPredictor

_LOSS_EPS = 1e-7


def report_instance_id_overlap(
    pool_a: Sequence[CVRPExample], pool_b: Sequence[CVRPExample]
) -> dict[str, int | float]:
    """Non-raising overlap diagnostic between two example pools, by stable instance id.

    Unlike :func:`validate_disjoint_examples`, this never raises — it's meant to surface a known
    overlap (e.g. an audited-reference subset drawn from the training pool) in provenance/reports
    rather than block a run.
    """
    ids_a = {example.instance.instance_id for example in pool_a}
    ids_b = {example.instance.instance_id for example in pool_b}
    overlap = ids_a & ids_b
    return {
        "overlap_count": len(overlap),
        "pool_b_size": len(ids_b),
        "overlap_fraction": (len(overlap) / len(ids_b)) if ids_b else 0.0,
    }


def _soft_wbce(
    m_prob: torch.Tensor,
    m_true: torch.Tensor,
    *,
    weighted: bool,
    pos_weight_power: float,
) -> torch.Tensor:
    """Off-diagonal BCE on probabilities; optional soft √ class weight (same as denoiser)."""
    n = m_prob.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=m_prob.device)
    prob = torch.clamp(m_prob[mask], _LOSS_EPS, 1.0 - _LOSS_EPS)
    target = m_true[mask].float()
    if not weighted:
        return ff.binary_cross_entropy(prob, target)
    pos = target.sum().clamp_min(1.0)
    neg = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = (neg / pos) ** float(pos_weight_power)
    loss = ff.binary_cross_entropy(prob, target, reduction="none")
    weights = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    return (loss * weights).mean()


def train_matrix_predictor(
    examples: Sequence[CVRPExample],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    augmentation: bool = False,
    weighted_bce: bool = True,
    pos_weight_power: float = 0.5,
) -> MatrixPredictor:
    """Train the P2.1 non-diffusion ``MatrixPredictor`` (fair recipe: x9 aug + soft sqrt-WBCE)."""
    torch.manual_seed(seed)
    model = MatrixPredictor(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_pool = expand_examples(examples) if augmentation else examples
    n_views = AUGMENT_NUM if augmentation else 1
    print(
        f"P2.1 fair train: n={len(examples)} epochs={epochs} "
        f"augmentation={augmentation} (x{n_views} -> {len(train_pool)}) "
        f"weighted_bce={weighted_bce} pos_weight_power={pos_weight_power} device={device}",
        flush=True,
    )
    for epoch in range(epochs):
        order = torch.randperm(
            len(train_pool), generator=torch.Generator().manual_seed(seed + epoch)
        )
        total = 0.0
        n_steps = 0
        for idx in order.tolist():
            view = train_pool[int(idx)]
            coords = torch.from_numpy(view.instance.customer_coords()).float().to(device)
            demands = torch.from_numpy(view.instance.customer_demands()).float().to(device)
            m_true = torch.from_numpy(view.constraint_matrix).float().to(device)
            optimizer.zero_grad()
            m_prob = model(coords, demands, float(view.instance.capacity))
            loss = _soft_wbce(
                m_prob,
                m_true,
                weighted=weighted_bce,
                pos_weight_power=pos_weight_power,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_steps += 1
        print(
            f"P2.1 epoch={epoch} train_loss={total / max(n_steps, 1):.4f} steps={n_steps}",
            flush=True,
        )
    model.eval()
    return model


@torch.no_grad()
def predict_matrix_predictor_probs(
    model: MatrixPredictor,
    examples: Sequence[CVRPExample],
    device: torch.device,
) -> list[npt.NDArray[np.float64]]:
    """Predict ``m_prob`` for each example with a trained P2.1 ``MatrixPredictor``."""
    model.eval()
    probabilities: list[npt.NDArray[np.float64]] = []
    for example in examples:
        coords = torch.from_numpy(example.instance.customer_coords()).float().to(device)
        demands = torch.from_numpy(example.instance.customer_demands()).float().to(device)
        m_prob = model(coords, demands, float(example.instance.capacity)).detach().cpu().numpy()
        probabilities.append(m_prob.astype("float64"))
    return probabilities


def validate_disjoint_examples(
    selection_examples: Sequence[CVRPExample], test_examples: Sequence[CVRPExample]
) -> None:
    """Reject validation/test leakage by stable instance id."""
    selection_ids = {example.instance.instance_id for example in selection_examples}
    test_ids = {example.instance.instance_id for example in test_examples}
    overlap = selection_ids & test_ids
    if overlap:
        raise ValueError(f"selection/test instance overlap detected: {sorted(overlap)[:5]}")


def score_matrix_probabilities(
    examples: Sequence[CVRPExample],
    m_probs: Sequence[npt.NDArray[np.float64]],
    *,
    m_hats: Sequence[npt.NDArray[np.float64]] | None = None,
    hard_from_hats: bool = False,
    threshold: float | None = None,
    adaptive_threshold: bool = True,
) -> dict[str, float]:
    """Score soft probabilities with an optional fixed hard matrix readout."""
    soft_predictions = [
        MatrixPrediction.from_example(example, probability)
        for example, probability in zip(examples, m_probs, strict=True)
    ]
    soft_metrics = compute_matrix_metrics(
        soft_predictions,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
    )

    if hard_from_hats:
        if m_hats is None:
            raise ValueError("hard_from_hats requires m_hats")
        hard_predictions = [
            MatrixPrediction.from_example(example, matrix)
            for example, matrix in zip(examples, m_hats, strict=True)
        ]
        hard_metrics = compute_matrix_metrics(
            hard_predictions,
            threshold=0.5,
            adaptive_threshold=False,
        )
        output: dict[str, float] = {
            "bce": float(soft_metrics.bce),
            "auc": (float(soft_metrics.auc) if soft_metrics.auc is not None else float("nan")),
            "precision": float(hard_metrics.precision),
            "recall": float(hard_metrics.recall),
            "f1": float(hard_metrics.f1),
            "threshold": 0.5,
            "calibration_error": float(soft_metrics.calibration_error),
            "capacity_consistency": float(soft_metrics.capacity_consistency),
            "num_examples": float(len(examples)),
        }
        predictions_for_size = hard_predictions
        threshold_for_size = 0.5
    else:
        output = {
            "bce": float(soft_metrics.bce),
            "auc": (float(soft_metrics.auc) if soft_metrics.auc is not None else float("nan")),
            "precision": float(soft_metrics.precision),
            "recall": float(soft_metrics.recall),
            "f1": float(soft_metrics.f1),
            "threshold": float(soft_metrics.threshold),
            "calibration_error": float(soft_metrics.calibration_error),
            "capacity_consistency": float(soft_metrics.capacity_consistency),
            "num_examples": float(len(examples)),
        }
        predictions_for_size = soft_predictions
        threshold_for_size = float(soft_metrics.threshold)

    by_size: dict[int, list[MatrixPrediction]] = {}
    for example, prediction in zip(examples, predictions_for_size, strict=True):
        by_size.setdefault(example.instance.n_customers, []).append(prediction)
    for size, size_predictions in sorted(by_size.items()):
        output[f"f1_n{size}"] = float(
            compute_matrix_metrics(
                size_predictions,
                threshold=threshold_for_size,
                adaptive_threshold=False,
            ).f1
        )
    return output


__all__ = [
    "predict_matrix_predictor_probs",
    "report_instance_id_overlap",
    "score_matrix_probabilities",
    "train_matrix_predictor",
    "validate_disjoint_examples",
]
