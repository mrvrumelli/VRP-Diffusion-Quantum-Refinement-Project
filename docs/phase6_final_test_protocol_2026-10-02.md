# Phase 6 final test protocol — declared 2026-10-02

Written before any refinement run on the sets below. Every configuration and comparison here was
chosen on the development panel and checked on the validation graphs; none is changed after the
test runs. Each configuration is scored once (`--once`), and its results are reported whatever
they are.

## Sets and starting solutions

| Set | Graphs | Starting routes and prior | Reference |
|---|---:|---|---|
| Reserved test | 96 (32 per size) | `outputs/baseline_freeze_20261001/reserved_test_scored` | Strengthened references (best of label and two seeded PyVRP runs), as in the baseline's one scoring |
| R/C/RC spatial OOD cells | 72 (8 per regime and size) | `outputs/baseline_freeze_20261001/frozen_ood` | Instance labels |

The baseline (`baseline-v1.0`) was scored on both sets on 2026-10-01; those are the "before"
numbers.

## Configurations

All loops run until a round brings no improvement, capped at 30 rounds, seed 0. Limits are
reorder segment size / movable exchange customers / neighbourhoods per selector per round. All
four selectors are used.

| ID | Refinement | Limits | Why it is included |
|---|---|---|---|
| T1 | Classical, one pass | 5 / 8 / 10 | Cheapest refinement |
| T2 | Classical, 10 restarts | 5 / 8 / 30 | Classical baseline at the limits SA uses |
| T3 | Classical, 10 restarts | 10 / 16 / 30 | Strongest classical refinement found on development and validation |
| T4 | SA on QUBO, 32 reads of 100 sweeps | 5 / 8 / 30 | Best limits for SA on development and validation |
| T5 | T4, then classical 10 restarts | 5 / 8 / 30 | Hybrid: QUBO operator plus classical polish |
| T6 | QAOA depth 1, exact simulation, at most 16 qubits | 4 / 6 / 10 | Quantum operator in the loop |
| T7 | Uniform random QUBO sampling, same shots and decoding | 4 / 6 / 10 | Matched control for T6 |
| T8 | Classical, 10 restarts | 4 / 6 / 10 | Classical baseline at QAOA's limits |
| T9 | T6, then classical 10 restarts | 4 / 6 / 10 | Hybrid with the quantum operator |
| T10 | Classical exact: Held-Karp reorders, exhaustive exchanges | 4 / 6 / 10 | The strongest classical baseline at QAOA's limits (amendment below) |

## Primary comparisons and decision rules

Paired per graph, 95% bootstrap intervals over graphs, each set reported separately.

1. **Refinement value.** T1 and T3 against the baseline. Refinement is useful if the gap change has
   an interval entirely below zero.
2. **Quantum-inspired operator against classical search, equal limits.** T4 minus T2.
3. **Does a QUBO stage add value after classical polishing (Q5)?** T5 minus T2. It adds value only
   if the interval lies entirely below zero.
4. **Quantum operator.** T6 minus T7 (does QAOA beat random sampling?), T6 minus T10 (does it
   match an exact classical solver on the same subproblems?), T6 minus T8 (does it match classical
   local search?), and T9 minus T8 (does it add value before classical polish?).

Secondary: gap by size, share of graphs improved, routes per solution, solver time per graph, and
the share of subproblems skipped by T6 and T7 for exceeding 16 qubits.

## Amendment, 2026-10-02, before any run on these sets

T10 was added after the development panel showed that, at QAOA's small limits, uniform random
QUBO sampling beats restarted local search: 6.19% against 6.66%. At 4 customers per reorder and
6 per exchange, the sampling-and-decoding rule nearly enumerates each subproblem. The fair
classical baseline there is an exact solver, not local search. No configuration above was changed.

## Commands

`outputs/phase6_20261001/run_queue6.sh` runs every configuration above with
`scripts/run_refinement_loop.py --once`.

## Execution

Run on 2026-10-02, after the declaration (commit `eec546d6b`) and the amendment (commit
`ee7f45c74`), each configuration once with `--once`. Results and verdicts are in section 5 of the
[Phase 6 write-up](phase6_refinement_experiments_2026-10-01.md#5-final-test); the raw outputs are in
`outputs/phase6_20261001/final/`, with summaries copied to `docs/evidence/phase6_20261001/final/`.
