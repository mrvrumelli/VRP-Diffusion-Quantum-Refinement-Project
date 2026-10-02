"""Paired analysis of a matched refinement experiment (rows.jsonl from run_refinement_experiment)."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["classical_ls", "classical_ls_budget", "sa_qubo", "sa_qubo_bias", "sqa_qubo"]


def cluster_ci(diffs_by_instance: dict, seed: int = 0, draws: int = 5000) -> list[float]:
    """Bootstrap over instances (neighborhoods of one instance are correlated)."""
    keys = list(diffs_by_instance)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        values = np.concatenate([diffs_by_instance[keys[i]] for i in pick])
        means.append(values.mean())
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def main() -> None:
    folder = Path(sys.argv[1])
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    table = defaultdict(dict)
    for row in rows:
        table[row["id"]][row["method"]] = row
    report = {}
    for kind in ("reorder", "exchange"):
        ids = [i for i, m in table.items() if next(iter(m.values()))["kind"] == kind]
        block = {"neighborhoods": len(ids)}
        opt_improvable = [i for i in ids if table[i]["exact_reference"]["optimum_cost"] < table[i]["exact_reference"]["initial_cost"] - 1e-9]
        block["optimum_improves_on_start_rate"] = round(len(opt_improvable) / len(ids), 4)
        block["mean_optimal_improvement_percent"] = round(float(np.mean([
            100 * (table[i]["exact_reference"]["initial_cost"] - table[i]["exact_reference"]["optimum_cost"]) / table[i]["exact_reference"]["initial_cost"] for i in ids])), 4)
        for method in [m for m in METHODS if all(m in table[i] for i in ids)]:
            sel = [table[i][method] for i in ids]
            # accept-if-better: a refinement loop never keeps a worse candidate
            kept = [min(r["cost"], r["initial_cost"]) for r in sel]
            block[method] = {
                "hit_rate": round(float(np.mean([r["hit_optimum"] for r in sel])), 4),
                "failures": sum(r["excess_percent"] is None for r in sel),
                "mean_excess_percent": round(float(np.mean([r["excess_percent"] for r in sel if r["excess_percent"] is not None])), 4),
                "mean_kept_excess_percent": round(float(np.mean([100 * (k - r["optimum_cost"]) / r["optimum_cost"] for r, k in zip(sel, kept)])), 4),
                "worse_than_start_rate": round(float(np.mean([r["cost"] > r["initial_cost"] + 1e-9 for r in sel])), 4),
                "captured_improvement_share": round(float(
                    sum(r["initial_cost"] - k for r, k in zip(sel, kept)) /
                    max(1e-12, sum(r["initial_cost"] - r["optimum_cost"] for r in sel))), 4),
                "mean_seconds": round(float(np.mean([r["seconds"] for r in sel])), 4),
                "mean_samples": round(float(np.mean([r["samples"] for r in sel if r["samples"] is not None])), 1)
                if any(r["samples"] is not None for r in sel) else None,
            }
        if any("exact_qubo" in table[i] for i in ids):
            sel = [table[i]["exact_qubo"] for i in ids if "exact_qubo" in table[i]]
            block["exact_qubo"] = {
                "count": len(sel),
                "hit_rate": round(float(np.mean([r["hit_optimum"] for r in sel])), 4),
                "mean_excess_percent": round(float(np.mean([r["excess_percent"] for r in sel])), 4),
                "worse_than_start_rate": round(float(np.mean([r["cost"] > r["initial_cost"] + 1e-9 for r in sel])), 4),
            }
        # paired contrasts, clustered by instance
        for a, b in [c for c in (("sa_qubo_bias", "sa_qubo"), ("sa_qubo", "classical_ls_budget"), ("sa_qubo_bias", "classical_ls_budget"), ("sa_qubo", "classical_ls"), ("sqa_qubo", "sa_qubo"), ("sqa_qubo", "classical_ls_budget")) if all(c[0] in table[i] and c[1] in table[i] for i in ids)]:
            excess, hits = defaultdict(list), defaultdict(list)
            for i in ids:
                inst = table[i][a]["instance_id"]
                kept = [100 * (min(table[i][m]["cost"], table[i][m]["initial_cost"]) - table[i][m]["optimum_cost"]) / table[i][m]["optimum_cost"] for m in (a, b)]
                excess[inst].append(kept[0] - kept[1])
                hits[inst].append(float(table[i][a]["hit_optimum"]) - float(table[i][b]["hit_optimum"]))
            excess = {k: np.array(v) for k, v in excess.items()}
            hits = {k: np.array(v) for k, v in hits.items()}
            block[f"{a}_minus_{b}"] = {
                "kept_excess_pp": round(float(np.concatenate(list(excess.values())).mean()), 4),
                "kept_excess_pp_ci95": cluster_ci(excess),
                "hit_rate_diff": round(float(np.concatenate(list(hits.values())).mean()), 4),
                "hit_rate_diff_ci95": cluster_ci(hits),
            }
        # breakdown by QUBO size bucket
        buckets = defaultdict(list)
        for i in ids:
            v = table[i]["sa_qubo"]["qubo_num_variables"]
            bucket = "<=16" if v <= 16 else "17-49" if v <= 49 else "50-99" if v <= 99 else ">=100"
            buckets[bucket].append(i)
        block["by_qubo_size"] = {
            bucket: {"count": len(members), **{m: round(float(np.mean([table[i][m]["hit_optimum"] for i in members])), 3) for m in METHODS if all(m in table[i] for i in members)}}
            for bucket, members in sorted(buckets.items())
        }
        types = defaultdict(list)
        for i in ids:
            types[table[i]["sa_qubo"]["neighborhood_type"]].append(i)
        block["by_type_hit_rate"] = {
            t: {"count": len(members), **{m: round(float(np.mean([table[i][m]["hit_optimum"] for i in members])), 3) for m in METHODS if all(m in table[i] for i in members)}}
            for t, members in sorted(types.items())
        }
        report[kind] = block
    (folder / "analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
