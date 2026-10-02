# Phase 6: refining the frozen baseline — 2026-10-01

Phase 6 asks how much local refinement improves the solutions of the frozen learned baseline, and
whether QUBO-based subproblem solvers, quantum-inspired or quantum, can deliver that improvement.
All experiments follow the protocol in [quantum_scope.md](quantum_scope.md) and start from the
frozen `baseline-v1.0` solutions (see the
[freeze record](classical_baseline_freeze_proposal_2026-09-30.md#freeze-record-2026-10-01)).
Everything here is development evidence. None of it is a claim of quantum advantage, speedup or
scalability.

## Summary

- **Refinement substantially improves every solution, and the gain holds on unseen graphs.** On
  the reserved test, scored once, one classical pass takes the gap from 19.2% to 9.7% in 0.2–3
  seconds per graph. The strongest refinement takes it to 4.1%, and the R/C/RC cells from 16.5%
  to 3.5%, in 3 seconds to 2.2 minutes per graph depending on size, timed in a single process. That refinement is restarted classical local search
  with larger neighbourhoods. The refined pipeline is still far from a strong classical solver.
- **More rounds saturate quickly; neighbourhood size and count matter more.** Every run converged
  within 15 rounds. Larger neighbourhoods help classical search by 1–2 points but not annealing,
  and more neighbourhoods per round help every solver.
- **The learned matrix picks better places to refine.** Neighbourhoods built from its uncertainty
  about route membership remove 1.4–2.0 more points of cost per round than matched random
  neighbourhoods, on the development and validation graphs and with both solvers. Its uncertainty ranks the
  customers that should move better than geometry does, although its probabilities are
  over-confident. The edge-confidence selector adds nothing in its current form, because it feeds
  reorder subproblems that cannot act on its signal.
- **Biasing the QUBO with the matrix helps on exchanges.** The gain grows with the bias strength
  and holds on both sets; in the full loop the strongest bias brings annealing level with
  classical search.
- **No QUBO operator beats classical search on identical subproblems.** Simulated annealing loses
  or ties, simulated quantum annealing does worse than annealing, and QAOA in the loop ties both an
  exact classical solver and random sampling, at about 40 times the exact solver's time. At the
  16 qubits that can be simulated, subproblems are small enough to be effectively enumerated.
- **A QUBO stage before classical polish gives at most a small gain.** About half a point on the
  final test sets, significant on one of them, at 26–39% more solver time; at about equal time on
  development and validation it was not separable from zero.


## 1. Setup

**Baseline solutions.** For every graph, the frozen pipeline regenerates the 50-step diffusion
prior and decodes the per-size policy greedily from 16 starts. Refinement starts from those routes
and also uses the prior matrix. Two graph sets are used: the 72-graph development panel, 24 graphs
per size, which also informed earlier model selection; and the 45 validation graphs, 16, 16 and 13
per size, used for confirmation. The reserved test set and the R/C/RC spatial OOD cells are
scored once, at the end, under a protocol declared before any refinement run on them (section 5).

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
| Classical exact | Held-Karp for reorders and exhaustive assignment search for exchanges; small subproblems only |
| SA on QUBO + bias | The same with the diffusion prior added to the QUBO (α = 0.5 unless stated) |
| SQA on QUBO | Simulated quantum annealing: path-integral Monte Carlo with 16 Trotter replicas (`quantum/annealing_solver.py`) |
| QAOA | QAOA in exact statevector simulation on QUBOs of at most 16 qubits (`quantum/qaoa_solver.py`); larger subproblems are skipped unchanged |
| Random QUBO sampling | Uniform random bitstrings with QAOA's shot count and decoding rule: the matched control for QAOA |

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

## 3. More rounds and larger neighbourhoods

Each configuration ran until a round brought no improvement, with a cap of 30 rounds, and the
cost was recorded after every round. Every solver is deterministic, so the cost after round r
equals what a run capped at r rounds would give. Three solvers were crossed with five limit
settings, written as reorder segment size / movable exchange customers / neighbourhoods per
selector per round.

**More rounds.** Default limits; gap % after each round:

| Solver | Before | Round 1 | Round 3 | Round 10 | Converged | Mean rounds, dev / val |
|---|---:|---:|---:|---:|---:|---:|
| Classical, one pass, dev / val | 18.48 / 18.33 | 11.20 / 11.64 | 9.26 / 9.64 | 8.70 / 9.34 | 8.70 / 9.34 | 3.8 / 3.6 |
| Classical, 10 restarts, dev / val | 18.48 / 18.33 | 10.11 / 10.52 | 7.07 / 7.38 | 5.83 / 6.21 | 5.81 / 6.20 | 5.1 / 4.8 |
| SA on QUBO, dev / val | 18.48 / 18.33 | 10.54 / 11.10 | 7.51 / 8.38 | 6.47 / 7.01 | 6.46 / 6.99 | 4.8 / 5.0 |

- **Most of the gain comes in the first round,** which removes 36–45% of the gap.
- **Running to convergence adds about 1–1.4 points over 3 rounds** for restarted search and SA,
  and 0.3–0.6 for a single classical pass.
- **More rounds saturate quickly.** Every run on both sets stopped on its own, after at most
  15 rounds, and after 4–5 on average for restarted search and SA. Beyond that, extra rounds change nothing, because each round
  revisits the same highest-ranked neighbourhoods.

**Larger and more neighbourhoods.** Final gap % over all graphs, development / validation:

| Limits | Classical, one pass | Classical, 10 restarts | SA on QUBO |
|---|---:|---:|---:|
| 5 / 8 / 10 (default) | 8.70 / 9.34 | 5.81 / 6.20 | 6.46 / 6.99 |
| 7 / 12 / 10 | 8.38 / 8.91 | 4.87 / 5.05 | 6.10 / 6.95 |
| 10 / 16 / 10 | 8.16 / 8.47 | 4.72 / 4.89 | 7.15 / 7.41 |
| 5 / 8 / 30 | 8.36 / 9.07 | 5.39 / 5.47 | 6.05 / 6.29 |
| 10 / 16 / 30 | 7.89 / 8.15 | **4.23 / 4.10** | 6.84 / 6.77 |

Paired change against default limits, points, development / validation:

| Limits | Classical, 10 restarts | SA on QUBO |
|---|---:|---:|
| 7 / 12 / 10 | −0.94 [−1.49, −0.39] / −1.14 [−2.05, −0.26] | −0.36 [−1.08, +0.33] / −0.04 [−0.71, +0.63] |
| 10 / 16 / 10 | −1.09 [−1.70, −0.48] / −1.31 [−2.36, −0.32] | +0.69 [−0.02, +1.41] / +0.41 [−0.39, +1.27] |
| 5 / 8 / 30 | −0.42 [−0.85, −0.06] / −0.73 [−1.42, −0.13] | −0.40 [−0.70, −0.15] / −0.70 [−1.32, −0.17] |
| 10 / 16 / 30 | −1.59 [−2.24, −0.94] / −2.10 [−3.30, −0.96] | +0.38 [−0.37, +1.13] / −0.23 [−1.00, +0.61] |

- **Larger neighbourhoods help classical search and not SA.** Restarted local search gains 1–2
  points, and its best setting, 10 / 16 / 30, reaches 4.2% and 4.1%: 1.5 / 3.8 / 7.7% at N20 /
  N50 / N100 on validation. SA does not gain, because reorder QUBOs of 49–100 variables are too
  hard for it (section 4, Q1).
- **More neighbourhoods per round help every solver** by about 0.4–0.7 points, because each round
  covers more of the solution.
- **The cost is time.** The best classical setting needs about 2 minutes of solver time per graph
  on average in these parallel runs, and 6 minutes at N100. Timed alone in a single process, it
  takes 3 s, 26 s and 2.2 minutes at N20, N50 and N100 (section 5).

The full tables per size are in `outputs/phase6_20261001/extension/tables_{development,validation}.md`.

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

**Simulated quantum annealing.** A second run of the same comparison added SQA
(`quantum/annealing_solver.py`: path-integral Monte Carlo with 16 Trotter replicas) at the same
0.25 s budget, on an otherwise idle machine. SQA solves 28% / 26% of reorders, against 60% / 59%
for SA in the same run and 100% for 2-opt with restarts; its kept excess is 2.0% on both sets.
Each SQA read simulates 16 coupled replicas, so only about 3 reads fit in the budget.

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

- **Simulated quantum annealing** does worse than SA on exchanges at the same 0.25 s budget: it
  solves 35% of them on both sets, against 74% / 72% for SA in the same run and 93% / 94% for
  restarted relocate/swap, with about 6–7 reads per subproblem.
- **Capacity.** Relocate/swap only moves to feasible assignments. About half of all SA reads
  violate capacity before repair, against 65% of random assignments, and the lowest-energy read
  violates it in 27–30% of subproblems. Repair plus best-of-reads still ends feasible in every
  unbiased run and all but one biased run. 75–79% of QAOA's stored shots satisfy capacity before
  repair.
- **QAOA versus random.** Up to 8 qubits, 1024 random shots effectively enumerate the state space,
  and random sampling beats QAOA. From 9 to 16 qubits depth-1 QAOA is ahead of random sampling and
  of a single relocate/swap pass, but restarted relocate/swap stays best. These are small groups
  of 28 and 54 subproblems.
- **Deeper circuits did not help on exchanges with cold starts.** Depth 2 and 3 were no better
  than depth 1. A deeper circuit can represent any shallower state, which pointed to the angle
  optimiser. Warm starts confirm it: rerunning the 82 development exchanges of 9–16 qubits with
  each depth started from the previous depth's interpolated angles, the best shot at depth 3 is
  QUBO-optimal in 70% of cases instead of 34%. The median amplification rises from 1.4× to 9×, and
  the expectation falls with depth in 68% of subproblems instead of 1%.
- **The surrogate limits every QUBO method.** Even when QAOA or SA finds the QUBO optimum, the
  decoded assignment is the true optimum only about half the time.
- **Validation repeats the screening.** Over 148 validation QUBOs, reorders again get hundreds of
  times their uniform probability on the optimum: a median of 265× at depth 2 and 642× at depth 3
  for 16 qubits. On exchanges, random sampling scored the same way again beats QAOA: 95.8% of
  optima against 82–87%, and 93.5% against 87.1% at 13–16 qubits. The small edge QAOA showed at
  9–16 qubits on development does not replicate.

### QAOA inside the refinement loop

`QAOASolver` (`quantum/qaoa_solver.py`) runs QAOA in exact simulation as a loop operator. To keep
every subproblem within 16 qubits, the limits are 4 customers per reorder and 6 movable customers
per exchange, with 10 neighbourhoods per selector per round, until convergence. Subproblems that
still exceed 16 qubits are skipped and left unchanged. That happened to under 1% of them. Depth
1, 4 random starts, 1024 shots, 50 lowest-energy shots decoded. All arms use the identical limits.

| Configuration | Dev gap % | Val gap % | Solver s per graph, dev |
|---|---:|---:|---:|
| Before refinement | 18.48 | 18.33 | — |
| Classical, one pass | 9.40 | 9.19 | 1.5 |
| Classical, 10 restarts | 6.66 | 7.12 | 23 |
| Classical exact: Held-Karp, exhaustive exchange | 6.30 | 6.70 | 7 |
| SA on QUBO | 7.01 | 7.71 | 27 |
| QAOA | 6.43 | 7.08 | 211 |
| Uniform random QUBO sampling, same shots and decoding | 6.19 | 6.88 | 9 |
| QAOA, then classical one pass | 6.30 | 6.96 | 211 |

| Paired difference, points | Development | Validation |
|---|---:|---:|
| QAOA minus random sampling | +0.23 [−0.01, +0.53] | +0.19 [−0.13, +0.57] |
| QAOA minus classical exact | +0.12 [−0.21, +0.39] | +0.37 [−0.26, +1.01] |
| Classical exact minus classical with restarts | −0.36 [−0.71, −0.08] | −0.42 [−0.80, −0.10] |

- **At the sizes QAOA can simulate, every subproblem is effectively enumerated.** Random sampling
  with the same decoding rule matches the exact classical solver, and QAOA is within noise of
  both. Exact classical search is about 30 times faster than simulated QAOA.
- **The right classical baseline at these sizes is exact, not local search.** Exact search beats
  restarted local search on both sets.

### Q3. Does the diffusion bias help QUBO search?

**It helps SA on exchange subproblems, increasingly with its strength. It does not help reorders.**

First comparison, α = 0.5 in probability mode, budgeted SA on the frozen sets:

| Comparison, bias on minus off | Development | Validation |
|---|---:|---:|
| Exchange, kept excess, points | −0.69 [−1.40, −0.11] | −1.06 [−2.12, −0.34] |
| Exchange, optimum hit rate, points | +3.3 [−1.4, +8.4] | +5.6 [0.0, +11.2] |
| Reorder, kept excess, points | +0.09 [−0.19, +0.35] | −0.11 [−0.49, +0.37] |
| Full loop, gap after refinement, points | +0.22 [−0.23, +0.71] | −0.12 [−0.58, +0.31] |

**Sweep** (`scripts/run_bias_sweep.py`): SA with 32 reads, α from 0.1 to 2, both bias modes, three
seeds, every subproblem of both frozen sets; each biased run is paired with the unbiased run of
the same seed. Kept excess, biased minus unbiased, in points:

| Setting | Exchange, dev | Exchange, val | Reorder, dev | Reorder, val |
|---|---:|---:|---:|---:|
| α = 0.1, probability | +0.02 [−0.08, +0.14] | −0.09 [−0.20, +0.02] | +0.00 [−0.16, +0.16] | +0.05 [−0.16, +0.26] |
| α = 0.5, probability | −0.58 [−1.08, −0.18] | −0.68 [−1.43, −0.12] | −0.09 [−0.23, +0.04] | +0.01 [−0.16, +0.20] |
| α = 1, probability | −0.71 [−1.28, −0.22] | −1.09 [−2.12, −0.29] | +0.05 [−0.09, +0.20] | +0.12 [−0.10, +0.33] |
| α = 2, probability | **−0.84 [−1.42, −0.35]** | **−1.27 [−2.27, −0.47]** | +0.09 [−0.09, +0.25] | +0.04 [−0.21, +0.30] |
| α = 1, confidence | −0.65 [−1.22, −0.19] | −0.75 [−1.68, −0.09] | +0.11 [−0.04, +0.25] | +0.20 [−0.06, +0.47] |
| α = 2, confidence | −0.75 [−1.33, −0.25] | −0.94 [−1.96, −0.13] | +0.17 [+0.02, +0.31] | +0.05 [−0.16, +0.26] |

- **Exchanges benefit, and more with a stronger bias.** From α = 0.5 upward the improvement is
  significant on both sets, in both modes. At α = 2 in probability mode the hit rate rises from
  74% to 78% on development and from 76% to 82% on validation.
- **Reorders do not benefit at any setting,** and the strongest confidence-mode bias slightly
  hurts on development.
- **Feasibility does not change:** 51% of reads violate capacity before repair with or without
  the bias.
- Best energies are not compared, because the bias changes the objective; route cost is the
  common scale.

**In the full loop.** The first comparison used α = 0.5, where the bias changed nothing. A follow-up
with α = 2 in probability mode, chosen from the sweep and therefore post hoc, runs the SA loop at
default limits until convergence:

| Set | SA unbiased, gap % | SA with α = 2, gap % | Difference, points | Against classical with 10 restarts |
|---|---:|---:|---:|---:|
| Development | 6.46 | 5.72 | −0.74 [−1.34, −0.15] | −0.09 [−0.64, +0.45] |
| Validation | 6.99 | 6.51 | −0.48 [−1.18, +0.27] | +0.31 [−0.39, +1.03] |

The strong bias lowers SA's final gap on both sets, significantly on development. It brings SA
level with restarted classical search, at about 40% more solver time. It does not make SA better
than classical search.

### Q4. Do diffusion-selected neighbourhoods beat randomly selected ones?

**Yes per neighbourhood, through the uncertainty selector. Over the whole loop, the prior's
selectors add value on top of the geometric ones. The edge-confidence selector adds nothing in
its current form.**

The diffusion arm uses only the prior-based selectors. Each round, the random arms draw exactly
as many reorder and exchange neighbourhoods of exactly the same sizes, placed at random: a
random segment of a random route, and random customers from a random pair of routes. "Random
nearby" restricts the pairs to spatially adjacent routes. Each random arm runs with three seeds,
averaged per graph. The loop runs until a round brings no improvement.
(`scripts/run_selection_ablation.py`)

**Both prior-based selectors together.** All graphs, development / validation:

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

**Each selector on its own**, against random neighbourhoods matched to it:

| Contrast | Classical, dev | Classical, val | SA, dev | SA, val |
|---|---:|---:|---:|---:|
| Uncertainty selector: first-round reduction, minus random nearby | +1.42 [+1.02, +1.84] | +1.46 [+0.95, +2.06] | +2.01 [+1.54, +2.50] | +1.47 [+1.02, +1.97] |
| Uncertainty selector: final gap, minus random nearby | −0.34 [−0.92, +0.18] | −0.50 [−1.19, +0.09] | −1.37 [−2.03, −0.76] | −0.56 [−1.35, +0.18] |
| Edge selector: first-round reduction, minus random segments | +0.13 [−0.04, +0.32] | −0.06 [−0.24, +0.12] | +0.11 [−0.03, +0.28] | −0.03 [−0.17, +0.12] |
| Edge selector: final gap, minus random segments | +0.41 [+0.15, +0.69] | +0.52 [+0.22, +0.82] | +0.40 [+0.14, +0.66] | +0.52 [+0.24, +0.80] |

**Adding the prior's selectors to the geometric ones,** all four together minus geometric only,
final gap at default limits:

| Solver | Development | Validation |
|---|---:|---:|
| Classical, one pass | −0.77 [−1.21, −0.38] | −0.41 [−0.93, +0.10] |
| SA on QUBO | −1.52 [−2.33, −0.76] | −1.27 [−2.34, −0.25] |

**Is the prior's uncertainty informative?** (`scripts/analyze_prior_calibration.py`, against the
reference routes of all 117 graphs.)

| Measure | Development | Validation |
|---|---:|---:|
| Pair-level discrimination of same-route pairs, AUROC | 0.896 | 0.897 |
| Mean predicted same-route probability / observed rate | 0.26 / 0.11 | 0.26 / 0.11 |
| Expected calibration error | 0.156 | 0.154 |
| Customers that should move: prior margin AUROC / geometry AUROC | 0.766 / 0.720 | 0.780 / 0.735 |
| Prior minus geometry | +0.045 [+0.025, +0.068] | +0.045 [+0.020, +0.072] |
| Edges the reference splits: prior AUROC / edge-length AUROC | 0.756 / 0.707 | 0.766 / 0.710 |
| Prior minus edge length | +0.049 [+0.030, +0.073] | +0.056 [+0.030, +0.086] |

A customer "should move" when most of its reference route-mates sit together on a different
baseline route.

- **The prior picks better exchange targets.** The uncertainty selector's neighbourhoods remove
  1.4–2.0 more points of cost in their first round than random neighbourhoods of the same number
  and size, on both sets and with both solvers. Its attempts succeed more often and gain about
  1.5 times as much each.
- **The uncertainty is informative but not calibrated.** The prior ranks well. It flags the
  customers that should move, and the edges the reference splits, better than geometry does.
  But it predicts same-route pairs about 2.4 times too often.
- **The edge selector is wired to the wrong move.** Low-confidence edges do flag customers that
  belong on different routes, but this selector only creates reorder subproblems, which cannot
  move a customer to another route. Matched against random segments, it adds nothing.
- **Over the whole loop, random nearby selection partly catches up,** because it keeps finding
  small improvements for more rounds. Added to the geometric selectors, the prior's selectors
  lower the final gap significantly in three of four comparisons.
- **The geometric selectors alone beat the prior's alone** because they propose about twice as
  many neighbourhoods; their counts are not matched.

### Q5. Does quantum add value after classical polishing?

**No measurable value once the classical polish is strong; a clear gain over a weak polish, at
much higher cost.**

All runs at default limits until convergence, seed 0. "SA, then classical" starts the classical
loop from SA's converged solutions (`outputs/phase6_20261001/run_queue4.sh`).

| Configuration | Dev gap % | Val gap % | Solver s per graph, dev / val |
|---|---:|---:|---:|
| Classical, one pass | 8.70 | 9.34 | 2 / 2 |
| SA, then classical one pass | 5.86 | 5.99 | 36 / 42 |
| Classical, 10 restarts | 5.81 | 6.20 | 34 / 35 |
| Classical, 20 restarts | 5.71 | 6.04 | 61 / 62 |
| SA, then classical 10 restarts | 5.64 | 5.45 | 51 / 61 |

| Paired difference, points | Development | Validation |
|---|---:|---:|
| SA + one pass minus one pass | −2.84 [−3.73, −2.06] | −3.35 [−4.38, −2.38] |
| SA + 10 restarts minus 10 restarts | −0.17 [−0.77, +0.43] | −0.75 [−1.55, 0.00] |
| SA + 10 restarts minus 20 restarts, about equal time | −0.07 [−0.59, +0.45] | −0.59 [−1.32, +0.02] |

- **Over a single classical pass, the SA stage gains about 3 points,** but it costs about 15 times
  the solver time.
- **Against restarted classical search at about equal time, the SA stage is not separable from
  zero.** Development shows no difference. Validation leans toward a small benefit, but its
  interval touches zero. Doubling the classical restarts on its own gains only 0.1–0.2 points, so
  if the SA stage helps, it is by handing the polish a different starting point, not by more
  search.
- **Robustness across seeds.** Over seeds 0, 1 and 2, restarted classical search averages 5.70% on
  development and 6.35% on validation, and SA averages 6.49% and 7.02%. The standard deviation
  across seeds is 0.11–0.23 points. SA is worse in every seed, by +0.79 [+0.40, +1.20] points
  averaged over seeds on development and +0.67 [−0.06, +1.37] on validation.

The pre-declared final test scores this comparison once on the reserved test and the R/C/RC
cells.

## 5. Final test

The reserved test (96 graphs, strengthened references) and the R/C/RC spatial OOD cells (72
graphs, instance labels) were each refined once, under the
[protocol](phase6_final_test_protocol_2026-10-02.md) declared before any run on them. It fixed ten
configurations, four primary comparisons and their decision rules. One amendment, adding the exact
classical baseline T10, was made and committed before any run. Every configuration ran until a
round brought no improvement, with seed 0, and was scored once.

Route gap %, N20 / N50 / N100 / all:

| Configuration | Reserved test | R/C/RC cells | Solver s per graph, reserved |
|---|---:|---:|---:|
| **Before refinement** | 8.56 / 21.88 / 27.20 / 19.22 | 8.29 / 19.20 / 22.09 / 16.53 | — |
| T1 Classical, one pass, 5/8/10 | 3.85 / 9.59 / 15.63 / 9.69 | 2.03 / 8.75 / 12.75 / 7.84 | 2 |
| T2 Classical, 10 restarts, 5/8/30 | 1.92 / 5.81 / 8.89 / 5.54 | 1.09 / 5.73 / 8.18 / 5.00 | 48 |
| **T3 Classical, 10 restarts, 10/16/30** | **1.16 / 4.48 / 6.77 / 4.14** | **0.95 / 3.40 / 6.28 / 3.54** | 121 |
| T4 SA on QUBO, 5/8/30 | 3.05 / 6.36 / 9.36 / 6.26 | 1.81 / 6.40 / 7.48 / 5.23 | 43 |
| T5 SA, then classical 10 restarts, 5/8/30 | 1.89 / 5.18 / 8.19 / 5.09 | 1.02 / 5.65 / 6.81 / 4.49 | 66 |
| T8 Classical, 10 restarts, 4/6/10 | 2.52 / 6.82 / 12.24 / 7.19 | 1.23 / 6.51 / 11.29 / 6.34 | 28 |
| T10 Classical exact, 4/6/10 | 2.21 / 6.74 / 12.13 / 7.03 | 1.17 / 6.43 / 10.72 / 6.10 | 6 |
| T6 QAOA depth 1, 4/6/10 | 2.36 / 6.77 / 12.24 / 7.12 | 1.29 / 6.46 / 10.84 / 6.20 | 245 |
| T7 Random QUBO sampling, 4/6/10 | 2.43 / 7.57 / 12.36 / 7.46 | 1.29 / 6.49 / 10.67 / 6.15 | 10 |
| T9 QAOA, then classical 10 restarts, 4/6/10 | 2.24 / 6.67 / 12.05 / 6.99 | 1.17 / 6.36 / 10.64 / 6.06 | 252 |

The pre-declared comparisons, paired over all graphs of each set, in points:

| Comparison | Reserved test | R/C/RC cells | Verdict under the declared rule |
|---|---:|---:|---|
| 1. T1 minus baseline | −9.53 [−10.54, −8.49] | −8.68 [−9.90, −7.57] | Refinement is useful on both sets |
| 1. T3 minus baseline | −15.08 [−16.47, −13.65] | −12.98 [−14.54, −11.43] | Refinement is useful on both sets |
| 2. T4 minus T2: SA against classical, equal limits | +0.72 [+0.16, +1.27] | +0.23 [−0.33, +0.79] | SA worse on the test, tied on the cells |
| 3. T5 minus T2: QUBO stage before classical polish | −0.45 [−0.96, +0.04] | −0.51 [−1.02, −0.01] | Rule not met on the test; met narrowly on the cells |
| 4. T6 minus T7: QAOA against random sampling | −0.34 [−0.76, +0.02] | +0.05 [−0.10, +0.20] | No difference |
| 4. T6 minus T10: QAOA against exact classical | +0.09 [−0.18, +0.35] | +0.09 [−0.06, +0.29] | No difference |
| 4. T6 minus T8: QAOA against classical restarts | −0.07 [−0.56, +0.38] | −0.15 [−0.53, +0.23] | No difference |
| 4. T9 minus T8: QAOA before classical polish | −0.21 [−0.68, +0.24] | −0.29 [−0.63, +0.03] | No difference |

- **Refinement transfers to unseen graphs.** The strongest classical refinement takes the reserved
  test from 19.2% to 4.1% and the R/C/RC cells from 16.5% to 3.5%, close to its development and
  validation results. Every graph with a positive starting gap improved under every
  configuration; the one reserved graph left unchanged already matched its reference.
- **The quantum-inspired operator does not beat classical search.** At equal limits, SA is worse
  on the reserved test and tied on the cells.
- **A QUBO stage before classical polish gives a small, inconsistent gain.** About half a point on
  both sets, significant on the cells only, at 26–39% more solver time. With the time-matched
  development and validation comparisons, where it was not separable from zero, the evidence that
  a QUBO stage adds value after polishing is weak.
- **QAOA in the loop ties both random sampling and the exact classical solver,** on both sets, at
  about 40 times the exact solver's time in simulation. It skipped 0.6–0.8% of subproblems for
  exceeding 16 qubits.
- **The strongest classical refinement remains the best configuration on every set,** and it uses
  neighbourhoods the QUBO operators cannot handle.

**Clean timing.** All other times in this report were measured with ten worker processes sharing
six physical cores. A separate pass timed each configuration in a single process, on 4
development graphs per size, with numeric libraries limited to one thread and at most four such
processes at once (`outputs/phase6_20261001/run_queue7.sh`). Refinement wall-clock seconds per
graph; end to end, add the baseline's 0.8–1.1 s for the prior and the policy on the GPU:

| Configuration | N20 | N50 | N100 |
|---|---:|---:|---:|
| T1 Classical, one pass, 5/8/10 | 0.2 | 1.0 | 2.9 |
| T2 Classical, 10 restarts, 5/8/30 | 1.5 | 15.6 | 49.3 |
| T3 Classical, 10 restarts, 10/16/30 | 3.3 | 26.2 | 134.6 |
| T4 SA on QUBO, 5/8/30 | 3.8 | 17.9 | 47.2 |
| T5 SA, then classical 10 restarts, 5/8/30 | 5.1 | 22.2 | 79.1 |
| T10 Classical exact, 4/6/10 | 0.5 | 3.0 | 5.0 |
| T6 QAOA depth 1, simulated, 4/6/10 | 40.1 | 159.2 | 145.7 |
| T7 Random QUBO sampling, 4/6/10 | 0.9 | 5.1 | 8.2 |

The parallel runs overstate single-process times by about 1.7–2.7 times. QAOA times are those of
exact classical simulation and say nothing about quantum hardware.

## 6. Limitations

- The development panel informed earlier model selection, and refinement settings were chosen on
  it and checked on the validation graphs. The reserved test and the R/C/RC cells were scored once,
  under the protocol declared before any run on them.
- References are the instance labels, except on the reserved test, which uses strengthened
  PyVRP references. None is a proven optimum.
- Exchange candidates are scored after re-ordering both routes with nearest neighbour plus 2-opt,
  so an exchange "optimum" is the best assignment under that ordering.
- Every quantum operator was simulated on a classical computer: QAOA as an exact, noise-free
  statevector of at most 16 qubits, and quantum annealing as path-integral Monte Carlo. There are no
  hardware results, and none of the timings say anything about quantum hardware.
- Inside the loop, QAOA ran at depth 1 with 4 random starts; warm starts were tested only in the
  screening.
- The strong-bias loop test (α = 2) used a strength chosen from the sweep, so it is post hoc.
- Three seeds were run for SA and restarted classical search at default limits; every other
  setting used seed 0.
- Times were measured with several worker processes sharing the machine, except in the clean
  timing pass of section 5, which used 4 graphs per size.
- The low-confidence-edge selector feeds reorder subproblems, which cannot act on the signal it
  detects (Q4). Its results describe that design, not the value of the signal.

## 7. Decisions for the team

1. **Adopt classical refinement in the pipeline.** One classical pass halves the gap for 0.2–3
   seconds per graph; restarted local search with larger neighbourhoods quarters it for 3 seconds
   at N20 up to about 2 minutes at N100. The team should choose the time budget.
2. **Decide whether the quantum stage continues, and at what subproblem size.** At sizes that can
   be simulated, no quantum or quantum-inspired operator adds value over exact classical search. A
   benefit is only conceivable on subproblems that are expensive to solve classically, which need
   hardware or approximate simulation.
3. **If it continues, fix the formulations first.** The exchange QUBO's surrogate objective loses
   about half of the optimal assignments before any solver runs, and the one-hot reorder encoding
   defeats annealing beyond 4 customers.
4. **Rewire and recalibrate the prior's guidance.** Feed the edge-confidence signal into exchange
   neighbourhoods instead of reorders, and recalibrate the matrix so the selection thresholds and
   the bias act on meaningful probabilities.
5. **Fix the budget unit.** The conclusions hold under both wall clock and sample count, but
   further claims need one agreed unit.

## Reproducing

All outputs are under `outputs/phase6_20261001/` unless stated.

| Step | Command or file |
|---|---|
| Baseline solutions | `outputs/baseline_freeze_20261001/frozen_panel`, `frozen_validation`, `frozen_ood`, `reserved_test_scored` (from `scripts/solve_with_baseline.py`) |
| Frozen neighbourhood sets | `scripts/build_neighborhood_sets.py` |
| Single-subproblem comparison, with SQA | `scripts/run_refinement_experiment.py`; `experiment_*`, `experiment_*_sqa` |
| Refinement loop | `scripts/run_refinement_loop.py`; `loop_*` (3-round runs) |
| Rounds and neighbourhood grid, Q5 polish | `run_extension.sh`, `run_queue2.sh`, `run_queue4.sh`; `extension/` |
| QAOA in the loop, extra seeds, time-matched Q5 | `run_queue5.sh`; `extension/*_qaoa_small*`, `*_seed*`, `*_restarts20` |
| Selection ablation, per selector | `scripts/run_selection_ablation.py`; `selection/` |
| Bias sweep | `scripts/run_bias_sweep.py`; `bias_sweep/` |
| Prior calibration | `scripts/analyze_prior_calibration.py`; `calibration/` |
| QAOA screening and warm start | `scripts/export_qubo_instances.py`, then `scripts/run_qaoa_screening.py` in `outputs/quantum_venv`; `qaoa_*` |
| QAOA comparisons | `qaoa_vs_classical.py`, `qaoa_random_control.py`, `analyze_qaoa_loop.py` |
| Capacity violations | `exchange_capacity_violations.py` |
| Final test, scored once | `docs/phase6_final_test_protocol_2026-10-02.md`, `run_queue6.sh`; `final/` |
| Clean timing pass | `run_queue7.sh`; `timing/` |
| Evidence copies | `docs/evidence/phase6_20261001/` |
