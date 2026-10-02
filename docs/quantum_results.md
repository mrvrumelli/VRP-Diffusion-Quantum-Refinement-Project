# Quantum results: interpretation — 2026-10-02

Task Q2.6. This document interprets the Phase 6 experiments: when quantum or quantum-inspired
operators help the refinement of the diffusion-based solver, when they fail, and whether any
benefit survives a fair classical comparison. The numbers, methods and intervals are in the
[Phase 6 write-up](phase6_refinement_experiments_2026-10-01.md); the final test followed a
[protocol declared in advance](phase6_final_test_protocol_2026-10-02.md). The rules for what may be
claimed are in [quantum_scope.md](quantum_scope.md); nothing here claims quantum advantage, speedup
or scalability.

## The study in one paragraph

A diffusion model predicts a soft same-route matrix for each CVRP instance, and a learned policy
decodes it into routes. Refinement then repeatedly selects small neighbourhoods of the solution,
turns each into a QUBO, solves it with a quantum or annealing operator or with classical search on
the identical subproblem, and accepts only strict improvements. The learned matrix enters twice:
it chooses which neighbourhoods to try, and it can bias the QUBO. Classical local search is the
fair baseline for every operator and the final polish in hybrid configurations.

**Verdict.** Refinement guided by the learned matrix substantially improves the diffusion-based
solver, and the improvement holds on a reserved test scored once. The matrix's contribution is
real: it picks better neighbourhoods and, as a QUBO bias, improves annealing on exchanges. The
quantum and quantum-inspired operators, however, never beat classical search on identical
subproblems. The best refinement found is entirely classical.

## What the learned matrix contributes

**Choosing neighbourhoods: yes, through its uncertainty about route membership.**

- Neighbourhoods built around customers whose prior is uncertain between two routes remove 1.4–2.0
  more points of cost in a round than random neighbourhoods of the same number and size. This
  holds on both graph sets and with both a classical and an annealing solver.
- The prior's uncertainty is informative: it flags customers that sit on the wrong route, and
  edges the reference solution splits, better than geometry does. Its probabilities are not
  calibrated, though: it predicts same-route pairs about 2.4 times too often. It is a ranking
  signal, not a probability.
- Over a full refinement run, random selection partly catches up by trying for more rounds. Added
  to the geometric selectors, the prior's selectors still lower the final gap in three of four
  comparisons.
- The second prior-based selector, low-confidence edges, adds nothing in its current form. It
  turns its signal into route reorderings, which cannot move a customer to another route. Its
  signal belongs in exchange neighbourhoods.

**Biasing the QUBO: yes for exchanges, more with a stronger bias, and not for reorders.** The bias
lowers annealing's excess over the optimum on two-route exchanges at every strength from α = 0.5
to 2, on both sets and in both bias modes. In the full loop, the strongest bias lowers annealing's
final gap by 0.5–0.7 points, significantly on one of two sets, which brings annealing level with
classical search but not past it.

## When quantum and annealing operators help

- **QAOA concentrates on optimal route orders.** In exact simulation, depth-2 and depth-3 QAOA
  puts hundreds of times the uniform probability on the optimal 16-qubit permutation, on both
  graph sets. This is a genuine property of the reorder formulation under QAOA.
- **Warm-started QAOA fixes its own optimiser problem.** Starting each depth from the previous
  depth's angles doubles how often the best shot is optimal at depth 3.
- **Annealing followed by classical polish beats a single classical pass** by about 3 points of
  gap, on both sets, at about 15 times the computation.

## When they fail

- **Against classical search on identical subproblems at equal budgets, every QUBO operator loses
  or ties.** Simulated annealing loses to restarted local search on single subproblems and in the
  full loop, in every seed tried. Simulated quantum annealing loses to simulated annealing, because
  each read simulates 16 coupled replicas. QAOA ties an exact classical solver at 30–80 times its
  time in simulation.
- **The subproblems that fit in simulation are nearly trivial.** At up to 16 qubits, uniform
  random sampling with QAOA's decoding rule finds every reorder optimum and 91–96% of exchange
  optima, and inside the loop it matches an exact classical solver.
- **QAOA on exchanges is no better than random sampling.** With the same shots and decoding rule,
  random bitstrings match or beat QAOA on both sets.
- **The exchange QUBO is a surrogate.** Its exact minimum is the true optimum only about 45% of
  the time, because route cost depends on visiting order, which a membership QUBO cannot express.
  No solver of that QUBO can recover what the formulation loses.
- **Larger neighbourhoods help classical search but hurt annealing.** The one-hot permutation
  encoding becomes hard for single-flip annealing at 25 or more variables, so annealing cannot
  follow classical search to the larger neighbourhoods that give the best results.

## Does any benefit survive the classical comparison?

**Not on the development and validation graphs.**

- Annealing followed by restarted classical polish against restarted classical search alone, at
  about equal time: −0.07 points [−0.59, +0.45] on development and −0.59 [−1.32, +0.02] on
  validation. Not separable from zero on either set.
- QAOA inside the loop against an exact classical solver on the same small neighbourhoods: +0.12
  [−0.21, +0.39] and +0.37 [−0.26, +1.01]. No benefit, at about 30 times the cost.
- The best refinement overall is classical: restarted local search with larger neighbourhoods,
  4.2% and 4.1% gap on the two sets, a setting the QUBO operators cannot use.

**On the final test, scored once under the declared protocol:**

- Annealing followed by restarted classical polish against restarted classical search alone:
  −0.45 [−0.96, +0.04] points on the reserved test and −0.51 [−1.02, −0.01] on the R/C/RC cells.
  The declared rule required an interval entirely below zero: not met on the test, met narrowly on
  the cells. The QUBO stage also costs 26–39% more solver time.
- QAOA against the exact classical solver: +0.09 on both sets, both intervals spanning zero, at
  about 40 times the exact solver's time. QAOA against random sampling: no difference.
- The strongest classical refinement reaches 4.1% and 3.5% on the two sets; no configuration with
  a QUBO operator comes close.

So a small benefit of a QUBO stage before classical polish may exist, about half a point, but it
is not consistently significant and it costs more time. No quantum operator shows any benefit.

## The research questions

| Question | Answer |
|---|---|
| Q0. Can small local repairs be written as QUBOs? | Yes. Reorders exactly; exchanges with exact feasibility but a surrogate objective. |
| Q1. Does quantum or annealing improve one-route reordering? | No. 2-opt is as good or better and far faster; QAOA's optimum concentration does not translate into better routes. |
| Q2. Does quantum or annealing improve two-route exchange? | No. Restarted relocate/swap is best; QAOA does not beat random sampling. |
| Q3. Does the diffusion bias help QUBO search? | Yes for exchange subproblems, increasingly with strength; not for reorders; in the loop it brings annealing level with classical search. |
| Q4. Do diffusion-selected neighbourhoods beat random ones? | Yes per neighbourhood, through the uncertainty selector; over a full run they add value on top of geometric selection. |
| Q5. Does quantum add value after classical polishing? | At most about half a point, significant on one of two final-test sets, at 26–39% more time; not separable from zero at equal time on development and validation. |

## What would have to change for a quantum benefit to be plausible

1. **Subproblems that are hard classically.** Every subproblem studied here is solved exactly by a
   classical method in milliseconds. A benefit can only appear where exact classical solution is
   expensive, which today also means beyond statevector simulation, so the next step needs
   hardware or approximate simulation, and a size chosen explicitly for it.
2. **A better exchange formulation.** An objective that reflects route order would remove the
   surrogate loss that caps every QUBO solver on exchanges.
3. **A better reorder encoding for annealing.** The one-hot permutation encoding defeats
   single-flip annealing beyond 4 customers.
4. **Calibrated priors and rewired selectors.** Recalibrating the matrix would let the bias and the
   selection thresholds work on meaningful probabilities, and the edge-confidence signal should
   drive exchange neighbourhoods.
