"""Spatial separation metrics for plain-CVRP stress distributions."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def spatial_metrics(coords: npt.ArrayLike, *, max_clusters: int = 8) -> dict[str, float]:
    """Compute dispersion and cluster metrics for one customer coordinate set."""
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("coords must have shape [n>=2, 2]")
    if not np.isfinite(points).all():
        raise ValueError("coords must contain only finite values")
    distances = squareform(pdist(points))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    center = points.mean(axis=0)
    radius = np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))
    pair_distances = pdist(points)

    bins = np.clip((points * 10).astype(int), 0, 9)
    counts = np.bincount(bins[:, 0] * 10 + bins[:, 1], minlength=100)
    probabilities = counts[counts > 0] / points.shape[0]
    grid_entropy = -float(np.sum(probabilities * np.log(probabilities))) / math.log(100)

    best_silhouette = -1.0
    upper_k = min(max_clusters, points.shape[0] - 1)
    for n_clusters in range(2, upper_k + 1):
        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=5).fit_predict(points)
        score = float(silhouette_score(points, labels))
        best_silhouette = max(best_silhouette, score)
    return {
        "mean_nearest_neighbor": float(nearest.mean()),
        "std_nearest_neighbor": float(nearest.std()),
        "mean_pair_distance": float(pair_distances.mean()),
        "radius_of_gyration": float(radius),
        "normalized_grid_entropy": grid_entropy,
        "best_kmeans_silhouette": best_silhouette,
    }


def summarize_spatial_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-instance spatial metrics with means and sample deviations."""
    if not rows:
        raise ValueError("cannot summarize empty spatial metrics")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("spatial metric rows have inconsistent keys")
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary
