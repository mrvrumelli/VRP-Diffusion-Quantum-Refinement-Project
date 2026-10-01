# Quantum results: interpretation — 2026-10-02

Task Q2.6. This document interprets the Phase 6 experiments: when quantum or quantum-inspired
operators help the refinement of the diffusion-based solver, when they fail, and whether any
benefit survives a fair classical comparison. The numbers, methods and intervals are in the
[Phase 6 write-up](phase6_refinement_experiments_2026-10-01.md). The rules for what may be claimed
are in [quantum_scope.md](quantum_scope.md); nothing here claims quantum advantage, speedup or
scalability.

## The study in one paragraph

A diffusion model predicts a soft same-route matrix for each CVRP instance, and a learned policy
decodes it into routes. Refinement then repeatedly selects small neighbourhoods of the solution,
turns each into a QUBO, solves it with a quantum or annealing operator or with classical local
search on the identical subproblem, and accepts only strict improvements. The learned matrix
enters twice: it chooses which neighbourhoods to try, and it can bias the QUBO. Classical local
search is the fair baseline for every operator and the final polish in hybrid configurations.

<!-- PENDING: summary verdict once the in-loop QAOA and final test are in -->

## What the learned matrix contributes

**Selecting neighbourhoods: yes.** Neighbourhoods chosen by the prior's uncertainty produce more
improvement per attempt than the same number of equally sized random neighbourhoods. The effect
appears on both graph sets and with both classical and annealing solvers. The prior's ranking is
informative: it flags customers that sit on the wrong route, and edges the reference solution
breaks, better than geometry alone. Its probabilities are over-confident, though, so the signal is
a ranking, not a calibrated probability.

**Biasing the QUBO: only on single exchange subproblems.** The bias helps simulated annealing on
two-route exchanges, does nothing on route reordering, and leaves the final gap of the full loop
unchanged.

<!-- PENDING: bias sweep and per-selector ablation -->

## When quantum and annealing operators help

- **QAOA concentrates on optimal route orders.** In exact simulation, depth-2 and depth-3 QAOA
  puts hundreds of times the uniform probability on the optimal 16-qubit permutation, on both
  graph sets. This is a genuine property of the reorder formulation under QAOA.
- **Warm-started QAOA fixes its own optimiser problem.** Starting each depth from the previous
  depth's angles roughly doubles how often the best shot is optimal at depth 3.
- **Annealing plus classical polish beats a single classical pass,** by about 3 points of gap,
  when given about 15 times the computation.

## When they fail

- **Against classical local search with restarts, on identical subproblems and equal budgets,
  every QUBO operator tested so far loses or ties.** This holds for simulated annealing on single
  subproblems and in the full loop, under both budget units, and for QAOA on the subproblems that
  fit in simulation. <!-- PENDING: SQA and in-loop QAOA -->
- **The subproblems that fit in simulation are trivial.** Reorders of up to 4 customers are solved
  exactly by every method, including uniform random sampling with the same decoding rule.
- **The exchange QUBO is a surrogate.** Its exact minimum is the true optimum only about 45% of
  the time, because route cost depends on visiting order, which a membership QUBO cannot express.
  No solver of that QUBO can recover what the formulation loses.
- **QAOA on exchanges is no better than random sampling.** With the same shot count and the same
  decode-and-keep-best rule, uniform random bitstrings match or beat QAOA on both sets.
- **Larger neighbourhoods hurt annealing but help classical search.** The one-hot permutation
  encoding becomes hard for single-flip annealing at 25 or more variables.

## Does any benefit survive the classical comparison?

<!-- PENDING: Q5 time-matched control, in-loop QAOA and final test -->

## What would have to change for a quantum benefit to be plausible

1. **Subproblems that are hard classically.** Every subproblem studied here is solved exactly by
   a classical method in under a millisecond. A benefit can only appear where exact classical
   solution is expensive, which today also means beyond statevector simulation.
2. **A better exchange formulation.** An objective that reflects route order would remove the
   surrogate loss that currently caps every QUBO solver on exchanges.
3. **Calibrated priors.** Recalibrating the matrix could make the bias and the selection
   thresholds operate on meaningful probabilities.
