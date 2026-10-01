"""Is the diffusion prior's uncertainty informative where refinement needs it?

For every frozen baseline solution (``solve_with_baseline.py`` output: routes plus the 50-step
prior ``m_prob``), three questions against the instance's reference routes:

1. **Calibration.** For every customer pair, is ``m_prob`` a calibrated probability that the pair
   shares a route in the reference? Reliability table, expected calibration error, Brier score.
2. **Customer uncertainty.** Does the prior's route-affinity margin, the score the
   ``uncertain_m`` selector uses (own-route affinity minus best other-route affinity), flag the
   customers that the baseline put on the wrong route? Two labels: "misassigned" when its
   baseline route-mates overlap its reference route-mates by Jaccard < 0.5, and the stricter
   "misplaced" when most of its reference route-mates sit together on a different baseline route
   (so moving it there is the natural repair). Compared with a
   geometry-only score: distance to the nearest own-route customer minus distance to the nearest
   customer on another route.
3. **Edge confidence.** Do low ``m_prob`` edges of the baseline routes, the score of the
   ``low_confidence_edges`` selector, mark consecutive customers that the reference puts on
   different routes? Compared with edge length.

AUROC is computed on the pooled items; 95% intervals come from a bootstrap over graphs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.quantum.neighborhoods import route_affinities

ROOT = Path(__file__).resolve().parents[1]
FloatArray = npt.NDArray[np.float64]


def auroc(scores: FloatArray, labels: npt.NDArray[np.bool_]) -> float:
    """Probability that a positive scores above a negative (ties count half)."""
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _mates(routes: list[list[int]]) -> dict[int, set[int]]:
    return {c: set(route) - {c} for route in routes for c in route}


def _graph_items(record_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text())
    example = load_example(ROOT / record["file"])
    instance = example.instance
    m_prob = np.load(record_path.with_name(f"{record['instance_id']}.m_prob.npz"))["m_prob"]
    m_prob = np.clip((m_prob.astype(np.float64) + m_prob.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(m_prob, 0.0)
    routes = [list(r) for r in record["routes"] if r]
    reference = [list(r) for r in example.solution.routes if r]
    n = instance.n_customers
    ref_route = {c: k for k, route in enumerate(reference) for c in route}
    xy = np.asarray(instance.coords)[np.asarray(instance.customer_node_indices())]

    upper = np.triu_indices(n, k=1)
    pair_prob = m_prob[upper]
    pair_same = np.array(
        [ref_route[int(i)] == ref_route[int(j)] for i, j in zip(*upper, strict=True)]
    )

    baseline_mates, reference_mates = _mates(routes), _mates(reference)
    route_of = {c: k for k, route in enumerate(routes) for c in route}
    prior_score, geo_score, misassigned, misplaced = [], [], [], []
    for customer in range(n):
        own = route_of[customer]
        affinities = route_affinities(m_prob, routes, customer)
        others = [value for k, value in enumerate(affinities) if k != own]
        margin = affinities[own] - max(others) if others else 1.0
        prior_score.append(-margin)  # higher = more uncertain
        dist = np.linalg.norm(xy - xy[customer], axis=1)
        own_members = [c for c in routes[own] if c != customer]
        other_members = [c for k, r in enumerate(routes) if k != own for c in r]
        near_own = dist[own_members].min() if own_members else 0.0
        near_other = dist[other_members].min() if other_members else np.inf
        geo_score.append(float(near_own - near_other))
        a, b = baseline_mates[customer], reference_mates[customer]
        union = a | b
        jaccard = len(a & b) / len(union) if union else 1.0
        misassigned.append(jaccard < 0.5)
        holders = Counter(route_of[mate] for mate in b)
        misplaced.append(bool(holders) and holders.most_common(1)[0][0] != own)

    edge_conf, edge_len, edge_cross = [], [], []
    for route in routes:
        for a, b in pairwise(route):
            edge_conf.append(1.0 - m_prob[a, b])
            edge_len.append(float(np.linalg.norm(xy[a] - xy[b])))
            edge_cross.append(ref_route[a] != ref_route[b])
    return {
        "instance_id": record["instance_id"],
        "n": n,
        "pair_prob": pair_prob,
        "pair_same": pair_same,
        "prior_score": np.asarray(prior_score),
        "geo_score": np.asarray(geo_score),
        "misassigned": np.asarray(misassigned),
        "misplaced": np.asarray(misplaced),
        "edge_conf": np.asarray(edge_conf),
        "edge_len": np.asarray(edge_len),
        "edge_cross": np.asarray(edge_cross),
    }


def _pooled(graphs: list[dict[str, Any]], key: str) -> FloatArray:
    return np.concatenate([g[key] for g in graphs])


def _calibration(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    p, y = _pooled(graphs, "pair_prob"), _pooled(graphs, "pair_same").astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    index = np.clip(np.digitize(p, edges) - 1, 0, 9)
    table, ece = [], 0.0
    for b in range(10):
        mask = index == b
        if not mask.any():
            continue
        mean_p, freq = float(p[mask].mean()), float(y[mask].mean())
        table.append(
            {
                "bin": [float(edges[b]), float(edges[b + 1])],
                "pairs": int(mask.sum()),
                "mean_predicted": mean_p,
                "observed_same_route": freq,
            }
        )
        ece += mask.mean() * abs(mean_p - freq)
    return {
        "pairs": int(p.size),
        "base_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_constant_base_rate": float(np.mean((y.mean() - y) ** 2)),
        "ece": float(ece),
        "auroc": auroc(p, y.astype(bool)),
        "reliability": table,
    }


def _auroc_with_ci(
    graphs: list[dict[str, Any]], score: str, label: str, other: str | None, draws: int
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    point = auroc(_pooled(graphs, score), _pooled(graphs, label).astype(bool))
    out: dict[str, Any] = {"auroc": point}
    samples, diffs = [], []
    for _ in range(draws):
        pick = [graphs[i] for i in rng.integers(0, len(graphs), len(graphs))]
        labels = _pooled(pick, label).astype(bool)
        value = auroc(_pooled(pick, score), labels)
        samples.append(value)
        if other is not None:
            diffs.append(value - auroc(_pooled(pick, other), labels))
    out["ci95"] = [float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))]
    if other is not None:
        out["minus_" + other] = float(
            point - auroc(_pooled(graphs, other), _pooled(graphs, label).astype(bool))
        )
        out["minus_" + other + "_ci95"] = [
            float(np.nanpercentile(diffs, 2.5)),
            float(np.nanpercentile(diffs, 97.5)),
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    solutions = args.solutions if args.solutions.is_absolute() else ROOT / args.solutions
    graphs = [_graph_items(p) for p in sorted(solutions.glob("*.json")) if p.name != "summary.json"]
    report: dict[str, Any] = {"solutions": str(args.solutions), "by_size": {}}
    for size in [*sorted({g["n"] for g in graphs}), "all"]:
        sel = [g for g in graphs if size == "all" or g["n"] == size]
        report["by_size"][str(size)] = {
            "graphs": len(sel),
            "calibration": _calibration(sel),
            "customers": int(_pooled(sel, "misassigned").size),
            "misassigned_rate": float(_pooled(sel, "misassigned").mean()),
            "customer_prior_margin": _auroc_with_ci(
                sel, "prior_score", "misassigned", "geo_score", args.draws
            ),
            "customer_geometry": _auroc_with_ci(sel, "geo_score", "misassigned", None, args.draws),
            "misplaced_rate": float(_pooled(sel, "misplaced").mean()),
            "misplaced_prior_margin": _auroc_with_ci(
                sel, "prior_score", "misplaced", "geo_score", args.draws
            ),
            "misplaced_geometry": _auroc_with_ci(sel, "geo_score", "misplaced", None, args.draws),
            "edges": int(_pooled(sel, "edge_cross").size),
            "cross_route_edge_rate": float(_pooled(sel, "edge_cross").mean()),
            "edge_prior_confidence": _auroc_with_ci(
                sel, "edge_conf", "edge_cross", "edge_len", args.draws
            ),
            "edge_length": _auroc_with_ci(sel, "edge_len", "edge_cross", None, args.draws),
        }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    for size, block in report["by_size"].items():
        cal = block["calibration"]
        lines = [
            f"N{size}: ECE {cal['ece']:.3f}, Brier {cal['brier']:.3f} "
            f"(base-rate {cal['brier_constant_base_rate']:.3f}), pair AUROC {cal['auroc']:.3f}",
            f"  misassigned {block['misassigned_rate']:.3f}: prior "
            f"{block['customer_prior_margin']['auroc']:.3f}, geometry "
            f"{block['customer_geometry']['auroc']:.3f}",
            f"  misplaced {block['misplaced_rate']:.3f}: prior "
            f"{block['misplaced_prior_margin']['auroc']:.3f}, geometry "
            f"{block['misplaced_geometry']['auroc']:.3f}",
            f"  cross-route edges {block['cross_route_edge_rate']:.3f}: prior "
            f"{block['edge_prior_confidence']['auroc']:.3f}, length "
            f"{block['edge_length']['auroc']:.3f}",
        ]
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
