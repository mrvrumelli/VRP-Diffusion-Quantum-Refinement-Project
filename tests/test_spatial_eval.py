import numpy as np
import pytest

from vrp_diffusion_quantum.eval.spatial import spatial_metrics, summarize_spatial_metrics


def test_clustered_points_have_smaller_nearest_neighbor_distance() -> None:
    rng = np.random.default_rng(0)
    uniform = rng.random((100, 2))
    clustered = np.vstack(
        [rng.normal((0.25, 0.25), 0.01, (50, 2)), rng.normal((0.75, 0.75), 0.01, (50, 2))]
    )
    uniform_metrics = spatial_metrics(uniform)
    clustered_metrics = spatial_metrics(clustered)
    assert clustered_metrics["mean_nearest_neighbor"] < uniform_metrics["mean_nearest_neighbor"]
    assert clustered_metrics["best_kmeans_silhouette"] > uniform_metrics["best_kmeans_silhouette"]
    assert clustered_metrics["normalized_grid_entropy"] < uniform_metrics["normalized_grid_entropy"]


def test_spatial_metric_summary_and_validation() -> None:
    rows = [spatial_metrics(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])) for _ in range(2)]
    summary = summarize_spatial_metrics(rows)
    assert summary["mean_nearest_neighbor_mean"] > 0
    assert summary["mean_nearest_neighbor_std"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="shape"):
        spatial_metrics(np.zeros((1, 2)))
