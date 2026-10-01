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
per size, used here for confirmation. The reserved test set is not used.

**Quality measure.** The route gap is the cost above the instance's reference routes, in percent
of the reference cost, averaged over graphs. A gap of 0% means refinement matched the reference.
For scale, PyVRP reaches 0.00%, 0.09% and 1.11% on the development panel at N20, N50 and N100 in
1 second.

**The refinement loop** (`refine_solution` in `quantum/refinement.py`) repeats rounds of four
steps:

1. Select neighborhoods of the current solution with four selectors: the most expensive routes,
   customers whose prior is uncertain between two routes, consecutive customers the prior
   considers unlikely to share a route, and pairs of nearby routes.
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
   gap halves, but no graph comes close to the reference, and N50 and N100 stay at 7–14%.
4. **Part of the gain is fewer routes.** Exchanges empty and merge routes: N50 solutions drop from
   7.9 to about 6.9 routes. This repairs the route fragmentation the diffusion prior causes, a
   known source of the baseline's gap. Classical exchanges merge routes more often than SA.
5. **The validation graphs confirm the picture.** The gap falls from 18.3% to 9.6% after one
   classical pass and to 7.4% at best.
6. **The refined pipeline is still far from a strong classical solver.** PyVRP's 1-second gaps are
   below 1.2% at every size.

<!-- SECTION3 -->

## 4. Which subproblem solver

The QUBO path matters for the quantum stage, so classical and QUBO solvers were compared on
identical subproblems at equal budgets, first on single subproblems and then inside the loop.

**Single subproblems.** The three highest-scoring neighborhoods of each type per instance were
frozen as two sets before any solver ran (`scripts/build_neighborhood_sets.py`):

| Set | Instances | Neighborhoods | Reorder | Exchange | sha256 |
|---|---:|---:|---:|---:|---|
| Development | 72 | 822 | 392 | 430 | `8fc30f9b…8ef4444` |
| Validation | 45 | 511 | 243 | 268 | `c14038d9…5a1537` |

Each method solved every subproblem with 0.25 s of wall clock (`scripts/run_refinement_experiment.py`).
"Hit rate" is the share of subproblems where a method reaches the exact optimum, from Held-Karp or
exhaustive search. "Kept excess" is the cost above that optimum after accepting only improvements.

| Kind | Method | Hit rate, dev / val | Kept excess %, dev / val | Samples in budget, dev |
|---|---|---:|---:|---:|
| Reorder | Classical, one run | 92.9% / 95.9% | 0.14 / 0.13 | 1 |
| Reorder | Classical, restarts | 100% / 100% | 0.00 / 0.00 | 2752 |
| Reorder | SA on QUBO | 50.0% / 53.9% | 0.91 / 0.91 | 30 |
| Reorder | SA on QUBO + bias | 52.0% / 62.5% | 1.00 / 0.80 | 30 |
| Exchange | Classical, one run | 76.5% / 76.9% | 0.74 / 0.80 | 1 |
| Exchange | Classical, restarts | 91.9% / 91.8% | 0.17 / 0.15 | 47 |
| Exchange | SA on QUBO | 67.7% / 69.0% | 1.93 / 2.43 | 52 |
| Exchange | SA on QUBO + bias | 70.9% / 74.6% | 1.25 / 1.37 | 52 |
| Exchange | Exact QUBO minimum | 45.8% / 43.7% | — | — |

- **Classical search wins on single subproblems.** SA is worse on both sets and both kinds, with
  every paired interval excluding zero. Even one classical run beats SA's whole budget.
- **SA fails on larger reorders.** It solves every reorder of up to 4 customers, 16 variables, but
  only 42–48% of 5-customer ones, 25 variables. The one-hot permutation encoding with its large
  constraint penalty is a likely cause; 100 sweeps per read may also be too few.
- **The exchange QUBO is a surrogate and loses accuracy before any solver runs.** True route cost
  depends on visiting order, so the QUBO minimises insertion costs plus a dispersion term. Its exact
  minimum is the true optimum in only 44–46% of subproblems and is worse than the starting solution
  in 20%.
- **The diffusion bias helps SA on exchanges.** It lowers kept excess by 0.69 points [0.11, 1.40]
  on development and 1.06 [0.34, 2.12] on validation. On reorders its effect is not established.

**Inside the loop.** With the default 3-round loop, all 72 development graphs:

| Paired difference in gap after refinement | Points | 95% interval |
|---|---:|---:|
| SA minus classical, one pass | −1.75 | [−2.57, −1.00] |
| SA minus classical, 10 restarts, similar time | +0.44 | [+0.00, +0.90] |
| SA minus classical, 32 restarts, same sample count | +0.83 | [+0.44, +1.24] |
| SA with bias minus without | +0.22 | [−0.23, +0.71] |
| SA + bias + polish minus classical, 10 restarts | −0.47 | [−1.01, +0.07] |

On the 45 validation graphs, SA minus classical with 10 restarts is +1.00 [+0.37, +1.57], although
SA had about 20% more time, and the bias changes nothing (−0.12 [−0.58, +0.31]).

- **SA beats a single classical pass only because it does more work.** At similar time, restarted
  classical search ties SA on the development panel and beats it on validation. A comparison
  without classical restarts would wrongly favour SA by about 1.75 points.
- **The bias does not carry over to the loop.** Its gain on single exchanges does not change the
  final gap on either set.
- **SA followed by a classical polish ties classical search with restarts** on both sets.

## 5. QAOA screening in simulation

The 247 development-set QUBOs with at most 16 variables were exported
(`scripts/export_qubo_instances.py`): 55 reorder QUBOs of 4, 9 and 16 qubits (2–4 customers) and
192 exchange QUBOs of 1–16 qubits, slack bits included. `scripts/run_qaoa_screening.py` ran in the
isolated environment `outputs/quantum_venv` (Qiskit 2.5.2, qiskit-optimization 0.7.0). For each
QUBO and depth p = 1, 2, 3 it built the Ising Hamiltonian and `QAOAAnsatz`, optimised the angles
with COBYLA from 4 random starts on the exact expectation, and drew 1024 shots. A fast NumPy
statevector simulator was checked against Qiskit's `Statevector` for every QUBO and agreed to
within 2e-15.

The comparison is uniform random sampling with the same 1024 shots. "Amplification" is the
optimal state's probability under QAOA divided by its uniform probability.

| QUBOs | Qubits | Depth | Optimal-state probability | Median amplification | Best shot optimal, QAOA | Best shot optimal, random | Best shot matches true optimum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reorder | 9 | 1 | 0.035 | 9× | 100% | 98% | 100% |
| Reorder | 9 | 3 | 0.140 | 35× | 100% | 98% | 100% |
| Reorder | 16 | 1 | 0.0011 | 39× | 46% | 3% | 54% |
| Reorder | 16 | 2 | 0.0078 | 212× | 100% | 3% | 100% |
| Reorder | 16 | 3 | 0.0156 | 541× | 100% | 3% | 100% |
| Exchange | 1–4 | 3 | 0.70 | 2.8× | 100% | 100% | 50% |
| Exchange | 5–8 | 3 | 0.21 | 7.5× | 100% | 100% | 56% |
| Exchange | 9–12 | 2 | 0.0050 | 4× | 96% | 53% | 54% |
| Exchange | 13–16 | 1 | 0.0002 | 3× | 17% | 7% | 15% |
| Exchange | 13–16 | 2 | 0.0006 | 13× | 37% | 7% | 28% |
| Exchange | 13–16 | 3 | 0.0002 | 1.5× | 19% | 7% | 20% |

The full table is in `outputs/phase6_20261001/qaoa_report_development.json`.

- **Reorder QUBOs suit QAOA in simulation.** At depth 2 or more, the best of 1024 shots is the
  optimal permutation for every reorder QUBO, including all 26 at 16 qubits, where random
  sampling succeeds 3% of the time.
- **Exchange QUBOs do not.** At 13–16 qubits the best shot is QUBO-optimal in at most 37% of
  cases. Slack bits and capacity penalties are a likely cause; this screening does not isolate it.
- **Depth 3 is worse than depth 2 on exchanges.** A depth-3 circuit can represent any depth-2
  state, so this is an optimiser failure: 4 random starts are too few for 6 angles. Warm-starting
  each depth from the previous depth's angles is the standard fix.
- **The surrogate caps exchange quality again.** Even when QAOA finds the QUBO optimum, the
  decoded assignment is the true optimum only about half the time.
- **Absolute probabilities are tiny.** A 1.6% chance of the optimal state at 16 qubits is large
  relative to random but would need many shots on hardware. Runtimes are simulation times and say
  nothing about hardware.

## 6. Limitations

- The development panel informed earlier model selection, and the best setting per size is chosen
  on it. The validation graphs confirm the main results. The reserved test set stays unused until
  a refinement method and budget are fixed.
- References are the instance labels, not proven optima.
- Exchange candidates are scored after re-ordering both routes with nearest neighbour plus 2-opt,
  so an exchange "optimum" is the best assignment under that ordering.
- Times were measured with several worker processes sharing the machine.

## 7. Decisions for the team

<!-- DECISIONS -->

## Reproducing

| Step | Command or file |
|---|---|
| Baseline solutions | `outputs/baseline_freeze_20261001/frozen_panel`, `frozen_validation` (from `scripts/solve_with_baseline.py`) |
| Neighborhood sets | `scripts/build_neighborhood_sets.py` |
| Single-subproblem comparison | `scripts/run_refinement_experiment.py` |
| Refinement loop | `scripts/run_refinement_loop.py`; outputs in `outputs/phase6_20261001/loop_*` |
| Rounds and neighborhood grid | `outputs/phase6_20261001/run_extension.sh`; outputs in `outputs/phase6_20261001/extension/` |
| QAOA screening | `scripts/export_qubo_instances.py`, then `scripts/run_qaoa_screening.py` in `outputs/quantum_venv` |
