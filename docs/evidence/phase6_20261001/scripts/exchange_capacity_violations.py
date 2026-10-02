"""Q2: capacity violations before repair on the frozen development exchange subproblems.

For each exchange neighborhood: 32 SA reads of 100 sweeps (seed 0) on the unbiased and on the
diffusion-biased QUBO, plus 32 uniform random assignments as a reference. Records the share of
samples whose decoded assignment violates capacity before repair, whether the lowest-energy SA
sample is capacity-feasible, and whether repair always yields a feasible assignment. Relocate/swap
only ever moves to feasible assignments, so it has no violations by construction.
"""

import json
from pathlib import Path

import numpy as np

from vrp_diffusion_quantum.data.dataset import load_example
from vrp_diffusion_quantum.quantum.neighborhoods import Neighborhood, extract_exchange_subproblem
from vrp_diffusion_quantum.quantum.qubo import solve_simulated_annealing
from vrp_diffusion_quantum.quantum.qubo_bias import DiffusionBiasConfig, build_biased_exchange_qubo
from vrp_diffusion_quantum.quantum.qubo_exchange import build_exchange_qubo, decode_exchange, repair_exchange

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/phase6_20261001"
neighborhood_set = json.loads((P6 / "neighborhoods_development.json").read_text())
records = {r["instance_id"]: r for r in neighborhood_set["instances"]}
cache: dict = {}
stats = {name: {"violating": [], "best_feasible": [], "repaired_feasible": []} for name in ("sa", "sa_bias", "random")}
for entry in neighborhood_set["neighborhoods"]:
    if entry["kind"] != "exchange":
        continue
    record = records[entry["instance_id"]]
    if record["instance_id"] not in cache:
        m = np.load(ROOT / record["m_prob"])["m_prob"].astype(float)
        m = np.clip((m + m.T) / 2, 0, 1)
        np.fill_diagonal(m, 0)
        cache = {record["instance_id"]: (load_example(ROOT / record["file"]).instance, m)}
    instance, m_prob = cache[record["instance_id"]]
    neighborhood = Neighborhood(entry["neighborhood_type"], "exchange", tuple(entry["route_indices"]), tuple(entry["customers"]), float(entry["score"]), None)
    sub = extract_exchange_subproblem(instance, record["routes"], neighborhood)
    try:
        plain = build_exchange_qubo(instance, sub)
    except ValueError:
        continue
    biased = build_biased_exchange_qubo(instance, sub, m_prob, DiffusionBiasConfig(enabled=True, alpha=0.5))
    rng = np.random.default_rng(len(stats["sa"]["violating"]))
    for name, wrapper in (("sa", plain), ("sa_bias", biased), ("random", plain)):
        if name == "random":
            states = [tuple(int(b) for b in rng.integers(0, 2, wrapper.qubo.num_variables)) for _ in range(32)]
        else:
            samples = solve_simulated_annealing(wrapper.qubo, num_reads=32, num_sweeps=100, seed=0)
            states = [s.x for s in samples]  # sorted by energy
        raws = [decode_exchange(wrapper, x) for x in states]
        ok = [sub.is_feasible(raw) for raw in raws]
        repaired = [repair_exchange(wrapper, raw) for raw in raws]
        stats[name]["violating"].append(1.0 - float(np.mean(ok)))
        stats[name]["best_feasible"].append(bool(ok[0]))
        stats[name]["repaired_feasible"].append(all(a is not None and sub.is_feasible(a) for a in repaired))
report = {
    name: {
        "subproblems": len(v["violating"]),
        "samples_violating_capacity_before_repair": round(float(np.mean(v["violating"])), 4),
        "first_sample_feasible": round(float(np.mean(v["best_feasible"])), 4),
        "all_samples_feasible_after_repair": round(float(np.mean(v["repaired_feasible"])), 4),
    }
    for name, v in stats.items()
}
(P6 / "exchange_capacity_violations_development.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report, indent=1))
