# Phase 6: refining the frozen baseline — 2026-10-01

Phase 6 asks how much local refinement improves the solutions of the frozen learned baseline, and
whether QUBO-based subproblem solvers, quantum-inspired or quantum, can deliver that improvement.
All experiments follow the protocol in [quantum_scope.md](quantum_scope.md) and start from the
frozen `baseline-v1.0` solutions (see the
[freeze record](classical_baseline_freeze_proposal_2026-09-30.md#freeze-record-2026-10-01)).
Everything here is development evidence. None of it is a claim of quantum advantage, speedup or
scalability.

<!-- SUMMARY -->

## 1. Setup

**Baseline solutions.** For every graph, the frozen pipeline regenerates the 50-step diffusion
prior and decodes the per-size policy greedily from 16 starts. Refinement starts from those routes
and also uses the prior matrix. Two graph sets are used: the 72-graph development panel, 24 graphs
per size, which also informed earlier model selection; and the 45 validation graphs, 16, 16 and 13
per size, used for confirmation. The reserved test set is not used.

**Quality measure.** The route gap is the cost above the instance's reference routes, in percent
of the reference cost, averaged over graphs. A gap of 0% means refinement matched the reference.
For scale, PyVRP reaches 0.00%, 0.09% and 1.11% on the development panel at N20, N50 and N100 in
1 second.

**The refinement loop** (`refine_solution` in `quantum/refinement.py`) repeats rounds of four
steps:

1. Select neighborhoods of the current solution. Four selectors exist. Two use the diffusion
   prior: customers whose prior is uncertain between two routes, and consecutive customers the
   prior considers unlikely to share a route. Two do not: the most expensive routes, and pairs of
   nearby routes ("geometric" selectors).
2. Extract each one as a self-contained subproblem: reorder one route segment between fixed
   endpoints, or reassign movable customers between two routes.
3. Solve it with a subproblem solver.
4. Accept the result only if the whole solution stays feasible and its true cost strictly falls.

Step 4 makes refinement monotone: it can never make a solution worse. The loop stops after a round
without improvement or at a round cap. The default limits are reorder segments of at most 5
customers, at most 8 movable customers per exchange, and at most 10 neighborhoods per selector per
round.

**Subproblem solvers.**

| Solver | What it does |
|---|---|
| Classical, one pass | One deterministic 2-opt run for reorders, one relocate/swap run for exchanges, from the current solution |
| Classical with restarts | The same local search from the current solution plus random starting points, keeping the best |
| SA on QUBO | Simulated annealing on the subproblem's QUBO, 32 reads of 100 sweeps; every read is decoded, repaired and scored on true cost |
| SA on QUBO + bias | The same with the diffusion prior added to the QUBO (α = 0.5) |
| QAOA | Exact statevector simulation of QAOA on QUBOs of at most 16 qubits; screening only (section 4) |

Every solver is deterministic for a given seed, and SA uses a fixed number of reads, so results do
not depend on machine load. Times are per graph on one worker process while other workers run
alongside, so they compare settings within a run rather than measure deployment speed.

## 2. Before and after refinement

Default limits, at most 3 rounds. Development panel, route gap %:

| Solutions | N20 | N50 | N100 | All 72 | Seconds per graph |
|---|---:|---:|---:|---:|---:|
| **Before refinement** | **9.19** | **21.51** | **24.73** | **18.48** | — |
| After classical, one pass | 2.41 | 11.02 | 14.37 | 9.26 | 1.1 |
| After classical, 10 restarts | 1.16 | 8.12 | 11.93 | 7.07 | 11.9 |
| After SA on QUBO | 2.39 | 8.23 | 11.92 | 7.51 | 12.6 |
| After SA on QUBO + bias, then classical polish | 1.60 | 6.98 | 11.22 | 6.60 | 13.7 |

Validation graphs, same settings:

| Solutions | N20 | N50 | N100 | All 45 | Seconds per graph |
|---|---:|---:|---:|---:|---:|
| **Before refinement** | **8.31** | **21.08** | **27.26** | **18.33** | — |
| After classical, one pass | 1.97 | 11.39 | 16.94 | 9.64 | 0.9 |
| After classical, 10 restarts | 1.59 | 8.22 | 13.48 | 7.38 | 8.4 |
| After SA on QUBO | 2.46 | 9.59 | 14.18 | 8.38 | 10.1 |
| After SA on QUBO + bias, then classical polish | 1.23 | 8.71 | 13.31 | 7.38 | 10.6 |

How solution quality changes, development panel, ranges across the solver settings above:

| Measure | N20 | N50 | N100 |
|---|---:|---:|---:|
| Route cost reduction | 6.1–7.3% | 8.6–11.9% | 8.3–10.8% |
| Share of the gap to the reference removed | 72–87% | 49–67% | 42–54% |
| Graphs at or below the reference after refinement | 21–54% | 0% | 0% |
| Routes per solution, before | 4.42 | 7.92 | 11.58 |
| Routes per solution, after | 3.92–4.21 | 6.88–7.46 | 11.00–11.50 |

1. **Every solution improves.** All 72 development and 45 validation solutions got cheaper under
   every setting, and every refined solution is feasible. Route cost falls by 6–12%.
2. **Even the cheapest setting halves the gap.** One classical pass takes about a second per graph
   and takes the panel from 18.5% to 9.3%. Spending about ten times longer brings it to 6.6–7.5%.
3. **Small instances nearly reach the reference; large ones do not.** At N20, up to half of the
   graphs end at or below the reference routes and the median gap reaches 0%. At N50 and N100 the
   gap roughly halves, but no graph reaches the reference.
4. **Part of the gain is fewer routes.** Exchanges empty and merge routes: N50 solutions drop from
   7.9 to about 6.9 routes. This repairs the route fragmentation the diffusion prior causes, a
   known source of the baseline's gap. Classical exchanges merge routes more often than SA.
5. **The validation graphs confirm the picture.** The gap falls from 18.3% to 9.6% after one
   classical pass and to 7.4% at best.
6. **The refined pipeline is still far from a strong classical solver.** PyVRP's 1-second gaps are
   below 1.2% at every size.

<!-- SECTION3 -->

## 4. Research questions

Each answer states the comparison, the evidence and the verdict. "Kept" costs apply the loop's
acceptance rule: a candidate worse than the starting solution is discarded.

### Q0. Can small local VRP repairs be expressed as QUBO?

**Yes for reorders. For exchanges, feasibility is exact but the objective is only a surrogate.**

- **Hand-crafted cases.** All 37 QUBO unit tests pass. For reorders, the energy of every
  permutation equals its path cost, and the ground state is a valid optimal permutation on random
  3- and 4-customer cases and on a line example with a known answer. For exchanges, the slack
  encoding represents every load exactly, feasible states have energy equal to the surrogate, and
  every capacity-violating state has higher energy than every feasible one.
- **Real subproblems.** The exact minimum of the reorder QUBO equals the true optimum on all 84
  development and validation segments small enough to enumerate. The exact minimum of the
  exchange QUBO is always capacity-feasible, on all 698 subproblems. It is the true optimum in only
  197 of 430 development and 117 of 268 validation subproblems, and worse than the starting
  solution in 20%. True route cost depends on visiting order, which a QUBO over route membership
  cannot express, so the exchange QUBO minimises insertion costs plus a dispersion term instead.

### Q1. Does quantum or annealing improve one-route reordering?

**No. 2-opt is as good or better, and thousands of times faster.**

Same segments, frozen neighborhood sets, 0.25 s budget per subproblem for the budgeted methods:

| Method | Optimum hit rate, dev / val | Kept excess over optimum, dev / val | Time per segment | Samples |
|---|---:|---:|---:|---:|
| 2-opt, one run | 92.9% / 95.9% | 0.14% / 0.13% | under 1 ms | 1 |
| 2-opt, restarts | 100% / 100% | 0.00% / 0.00% | 0.25 s | about 3000 |
| SA on QUBO | 50.0% / 53.9% | 0.91% / 0.91% | 0.25 s | 30–37 |

On average the optimal order is 2.5–2.7% cheaper than the starting segment, and 2-opt with
restarts always finds it. SA solves every segment of up to 4 customers but only
42–48% of 5-customer segments, which have 25 QUBO variables. The one-hot permutation encoding with
its large constraint penalty is a likely cause; 100 sweeps per read may also be too few.

QAOA fits only segments of up to 4 customers, 16 qubits, in simulation. Those 55 development
segments are too easy to separate methods: 2-opt finds the optimum in 98% of them, and SA, QAOA
and uniform random sampling scored the same way as QAOA find it in all of them. QAOA's own signal
is real but does not change the answer: at depth 2–3 the optimal 16-qubit permutation gets 212–541 times
its uniform probability, at 8–16 s of simulation per segment.

3-opt is not implemented. Held-Karp gives the exact optimum of these segments in under a
millisecond, which bounds what 3-opt could add.

### Q2. Does quantum or annealing improve two-route exchange?

**No. Relocate/swap with restarts is best; QAOA does not beat random sampling.**

Same subproblems, 0.25 s budget for the budgeted methods:

| Method | Optimum hit rate, dev / val | Kept excess, dev / val | Time | Capacity violations before repair |
|---|---:|---:|---:|---:|
| Relocate/swap, one run | 76.5% / 76.9% | 0.74% / 0.80% | 27–43 ms | none, by construction |
| Relocate/swap, restarts | 91.9% / 91.8% | 0.17% / 0.15% | 0.25 s | none, by construction |
| SA on QUBO | 67.7% / 69.0% | 1.93% / 2.43% | 0.25 s | 51% of reads |
| SA on QUBO + bias | 70.9% / 74.6% | 1.25% / 1.37% | 0.25 s | 51% of reads |

On the 192 development exchanges small enough for QAOA, every method is scored by the same rule:
decode and repair each stored sample, keep the best true cost, and discard it if it is worse than
the start. The random control draws 1024 uniform bitstrings and keeps the 50 with the lowest QUBO
energy, as the QAOA screening stored its 50 lowest-energy shots.

| Exchanges of at most 16 qubits | All 192: hit rate / kept excess | 1–8 qubits, 110 | 9–12 qubits, 28 | 13–16 qubits, 54 |
|---|---:|---:|---:|---:|
| Relocate/swap, one run | 89.1% / 0.28% | 97.3% | 96.4% | 68.5% |
| Relocate/swap, restarts | 97.4% / 0.00% | 100% | 100% | 90.7% |
| SA on QUBO | 72.4% / 3.35% | 70.0% | 82.1% | 72.2% |
| SA on QUBO + bias | 81.2% / 1.61% | 81.8% | 85.7% | 77.8% |
| QAOA, depth 1 | 87.0% / 1.87% | 84.5% | 92.9% | 88.9% |
| QAOA, depth 2 | 83.9% / 2.41% | 82.7% | 85.7% | 85.2% |
| QAOA, depth 3 | 83.9% / 2.38% | 82.7% | 89.3% | 83.3% |
| Uniform random bitstrings, same shots and rule | 90.6% / 0.60% | 95.5% | 82.1% | 85.2% |

The size columns show hit rates.

- **Capacity.** Relocate/swap only moves to feasible assignments. About half of all SA reads
  violate capacity before repair, against 65% of random assignments, and the lowest-energy read
  violates it in 27–30% of subproblems. Repair plus best-of-reads still ends feasible in every
  unbiased run and all but one biased run. 75–79% of QAOA's stored shots satisfy capacity before
  repair.
- **QAOA versus random.** Up to 8 qubits, 1024 random shots effectively enumerate the state space,
  and random sampling beats QAOA. From 9 to 16 qubits depth-1 QAOA is ahead of random sampling and
  of a single relocate/swap pass, but restarted relocate/swap stays best. These are small groups
  of 28 and 54 subproblems.
- **Deeper circuits do not help on exchanges.** Depth 2 and 3 are no better than depth 1. A deeper
  circuit can represent any shallower state, so this points to the angle optimiser: 4 random
  starts are too few for 4–6 angles.
- **The surrogate limits every QUBO method.** Even when QAOA or SA finds the QUBO optimum, the
  decoded assignment is the true optimum only about half the time.

### Q3. Does the diffusion bias help QUBO search?

**It helps SA on single exchange subproblems, not on reorders, and not in the full loop.**

| Comparison, bias on minus off | Development | Validation |
|---|---:|---:|
| Exchange, kept excess, points | −0.69 [−1.40, −0.11] | −1.06 [−2.12, −0.34] |
| Exchange, optimum hit rate, points | +3.3 [−1.4, +8.4] | +5.6 [0.0, +11.2] |
| Reorder, kept excess, points | +0.09 [−0.19, +0.35] | −0.11 [−0.49, +0.37] |
| Full loop, gap after refinement, points | +0.22 [−0.23, +0.71] | −0.12 [−0.58, +0.31] |

Feasibility does not change: 51% of reads violate capacity before repair with or without the
bias. Best energies are not compared, because the bias changes the objective, so biased and
unbiased energies measure different things; route cost is the common scale.

### Q4. Do diffusion-selected neighborhoods beat randomly selected ones?

**Yes, per neighborhood. For the final gap, they beat fully random selection, are not separable
from random selection among nearby routes, and help most as an addition to the geometric
selectors.**

The diffusion arm uses only the two prior-based selectors. Each round, the random arms draw
exactly as many reorder and exchange neighborhoods of exactly the same sizes, placed at random:
a random segment of a random route, and random customers from a random pair of routes. "Random
nearby" restricts the pairs to spatially adjacent routes. Each random arm runs with three seeds,
averaged per graph. The loop runs until a round brings no improvement.
(`scripts/run_selection_ablation.py`)

All graphs, development / validation:

| Measure | Diffusion | Random | Random nearby | Geometric |
|---|---:|---:|---:|---:|
| Share of attempts that improve, classical | 19.6% / 18.8% | 15.8% / 14.9% | 16.5% / 15.0% | 18.5% / 17.2% |
| Cost reduction per attempt, classical | 0.22% / 0.19% | 0.14% / 0.13% | 0.14% / 0.12% | 0.15% / 0.14% |
| First-round cost reduction, classical | 5.1% / 4.5% | 3.6% / 3.4% | 3.8% / 3.2% | 5.7% / 5.4% |
| Final gap, classical | 11.64% / 12.26% | 13.16% / 13.40% | 12.04% / 12.71% | 9.47% / 9.75% |
| Final gap, SA | 11.05% / 12.11% | 13.30% / 13.55% | 12.22% / 12.68% | 7.98% / 8.26% |

Paired differences, diffusion minus the other arm, in percentage points. A positive first-round
value and a negative final-gap value favour diffusion selection.

| Contrast | Classical, dev | Classical, val | SA, dev | SA, val |
|---|---:|---:|---:|---:|
| First-round cost reduction, minus random nearby | +1.31 [+0.91, +1.72] | +1.35 [+0.84, +1.94] | +1.92 [+1.46, +2.40] | +1.42 [+0.96, +1.91] |
| Final gap, minus random | −1.52 [−2.21, −0.92] | −1.14 [−1.80, −0.55] | −2.25 [−2.93, −1.62] | −1.44 [−2.21, −0.75] |
| Final gap, minus random nearby | −0.40 [−1.02, +0.15] | −0.45 [−1.10, +0.13] | −1.17 [−1.85, −0.52] | −0.56 [−1.43, +0.19] |

- **Per neighborhood, the prior picks better targets.** From the same starting solution with the
  same number and sizes of neighborhoods, diffusion selection removes 1.3–1.9 more points of cost
  in the first round, on both sets and with both solvers. Its attempts succeed more often and gain
  about 1.5 times as much each.
- **Over the whole loop, random nearby selection partly catches up.** It keeps finding
  improvements for more rounds, so its final gap is within noise of the diffusion arm's, except
  for SA on the development panel.
- **The geometric selectors alone do better than the diffusion selectors alone,** because they
  propose about twice as many neighborhoods; their counts are not matched.
- **Combined, the prior adds value.** On the development panel, all four selectors together end
  <!-- ALLFOUR --> below the geometric selectors alone.

### Q5. Does quantum add value after classical polishing?

<!-- Q5 -->

## 5. Limitations

- The development panel informed earlier model selection. The validation graphs confirm the main
  results. The reserved test set stays unused until a refinement method and budget are fixed.
- References are the instance labels, not proven optima.
- Exchange candidates are scored after re-ordering both routes with nearest neighbour plus 2-opt,
  so an exchange "optimum" is the best assignment under that ordering.
- "Quantum" in the loop means simulated annealing. QAOA was screened in exact simulation on
  subproblems of at most 16 qubits and could not be run inside the loop at useful neighborhood
  sizes.
- Times were measured with several worker processes sharing the machine.

## 6. Decisions for the team

<!-- DECISIONS -->

## Reproducing

| Step | Command or file |
|---|---|
| Baseline solutions | `outputs/baseline_freeze_20261001/frozen_panel`, `frozen_validation` (from `scripts/solve_with_baseline.py`) |
| Frozen neighborhood sets | `scripts/build_neighborhood_sets.py` |
| Single-subproblem comparison | `scripts/run_refinement_experiment.py` |
| Refinement loop | `scripts/run_refinement_loop.py`; outputs in `outputs/phase6_20261001/loop_*` |
| Rounds and neighborhood grid | `outputs/phase6_20261001/run_extension.sh`, `run_queue2.sh`; outputs in `outputs/phase6_20261001/extension/` |
| Selection ablation | `scripts/run_selection_ablation.py`; outputs in `outputs/phase6_20261001/selection/` |
| Classical polish after SA | `outputs/phase6_20261001/run_queue3.sh` |
| QAOA screening | `scripts/export_qubo_instances.py`, then `scripts/run_qaoa_screening.py` in `outputs/quantum_venv` |
| QAOA comparisons | `outputs/phase6_20261001/qaoa_vs_classical.py`, `qaoa_random_control.py` |
| Capacity violations | `outputs/phase6_20261001/exchange_capacity_violations.py` |
