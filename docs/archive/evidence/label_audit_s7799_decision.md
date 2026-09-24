# s7799 strong-label audit decision

## Audit integrity

- Input: 1,500 deterministic training examples, 500 each for CVRP20/50/100.
- Input SHA-256: `42e8cd106e434e66eb57a7e1521edb90259355e55c17456ff1298fd2bff17e7f`.
- PyVRP: 6,000/6,000 runs completed with four deterministic seeds.
- OR-Tools: 446/446 selected challenges completed.
- Solver errors: zero.
- Per-instance results: 1,500; reference examples: 1,500; accepted matrix examples: 1,204.

The audit is complete and internally consistent. OR-Tools did not beat the best PyVRP candidate in
any challenged case, so the unresolved issue is convergence and route-membership multiplicity,
not evidence that OR-Tools supplies better references.

## Results by size

| Size | Cost-converged | Matrix accepted | Needs review | Mean cost improvement | Mean matrix change from original |
|---|---:|---:|---:|---:|---:|
| CVRP20 | 100.0% | 98.6% | 1.4% | 0.00027% | 0.22% |
| CVRP50 | 99.2% | 94.4% | 5.6% | 0.208% | 3.95% |
| CVRP100 | 64.4% | 47.8% | 52.2% | 1.676% | 7.33% |

CVRP100 rejection reasons overlap: 113 examples fail both cost convergence and matrix stability,
65 fail only cost convergence, and 83 fail only matrix stability. Only 239/500 pass both gates.

## Initial policy decision (superseded by bounded training evidence below)

- **CVRP20:** existing one-second binary matrices are adequate for pilot and learning-curve
  training. Stronger labels do not materially improve cost and 98.6% pass matrix stability.
- **CVRP50:** use stronger canonical labels for versioned training data. The cost reference is
  accepted for 99.2%, but the original matrix changes materially in the upper tail. Exclude or
  represent the 5.6% ambiguous cases with multiple near-best references rather than pretending a
  single partition is certain.
- **CVRP100:** do not train a final binary-target model on the original labels. Four 40-second seeds
  leave 35.6% cost-unconverged and 39.2% matrix-unstable; 52.2% need review. A bounded 80-second
  follow-up is required before choosing between longer canonical solving and multi-reference
  training.

## Bounded follow-up gate

Select 50 CVRP100 examples deterministically:

- 20 failing both cost and matrix gates;
- 10 failing only cost convergence;
- 10 failing only matrix stability;
- 10 accepted stable controls.

Rerun the same four derived seeds at 80 seconds. Proceed as follows:

1. If cost convergence improves substantially but route ambiguity persists, use multiple unique
   near-best route partitions as repeated labels for each instance.
2. If both convergence and matrix stability pass at a high rate, use 80-second best-of-four
   canonical references for a bounded training subset.
3. If cost convergence remains poor, test 120 seconds only on the still-unconverged follow-up
   cases; do not immediately relabel the full corpus.
4. Preserve all original examples and write every follow-up candidate and derived dataset under a
   new versioned directory with its manifest and hashes.

At this point in the workflow, full training was not yet authorized.

## 80-second follow-up result

The stratified 50-instance follow-up completed 200/200 PyVRP runs and 38/38 OR-Tools
challenges with zero solver errors. Stable controls remained stable. Doubling CVRP100 runtime from
40 to 80 seconds changed the selected outcomes as follows:

- 12/30 previously cost-unconverged cases became cost-converged;
- 12/40 previously rejected matrix targets became accepted;
- only 1/10 matrix-only ambiguous cases became accepted;
- all 10 stable controls remained accepted;
- 18/50 still fail cost convergence and 23/50 still fail matrix stability.

Longer runtime helps underconvergence but does not remove route-partition ambiguity. Run a final
120-second comparison only on the 18 cases still failing cost convergence. Do not extend the
matrix-only cases: their near-best disagreement persisted despite the extra runtime, so they require
multi-reference handling rather than more canonical-solver time.

## 120-second follow-up result

The 18 CVRP100 cases that were still cost-unconverged at 80 seconds completed 72/72 PyVRP runs
and 17/17 OR-Tools challenges without errors. Only 3/18 passed reference acceptance and 1/18
passed matrix acceptance. Increasing the budget again is therefore not a reliable way to obtain a
unique binary target; no broader 120-second relabel was performed.

## Final training-label policy (v2)

The first derived policy retained CVRP20, used canonical-or-multiple references for CVRP50, and
multiple competitive references for CVRP100. It contained 2,304 label files for 1,500 source
instances, so instances with more alternatives received more optimizer weight. A controlled CUDA
comparison exposed this confound:

| Policy | Labels | Epochs | Approx. diffusion steps | Validation AUC | Validation F1 |
|---|---:|---:|---:|---:|---:|
| Original one-second | 1,500 | 5 | 470 | 0.9117 | 0.5831 |
| Strongest audited, all sizes | 1,500 | 5 | 470 | 0.9134 | 0.5877 |
| Mixed/multi-reference | 2,304 | 5 | 720 | 0.9163 | 0.5945 |
| Mixed/multi-reference, compute-matched | 2,304 | 3 | 432 | 0.9107 | 0.5830 |

The apparent multi-reference gain disappears at comparable optimizer steps. Policy v2 therefore
uses exactly one label per source: retain the original CVRP20 matrix and use the lowest-cost
feasible audited PyVRP reference for CVRP50 and CVRP100. The versioned dataset is
`data/processed/s7799_audit_policy_v2`, contains 500 examples per size, and has dataset hash
`842b4bbfee69f1802bc559daf51c7f655d5fe1c70cccbc0dba10930bc8289487`.

This is a pragmatic binary-target policy, not a claim that CVRP100 has a unique optimum. The audit
still proves substantial route-partition ambiguity; a future soft-target or one-reference-per-
instance stochastic objective should be evaluated before scaling beyond this pool.
