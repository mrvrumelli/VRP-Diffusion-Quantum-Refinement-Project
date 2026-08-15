# CVRP route-partition ambiguity and full-chain improvement plan

## Scope and current evidence

The model predicts the customer route-membership matrix: `M[i,j] = 1` exactly when customers
`i` and `j` occur in the same route. Current training uses one hard binary matrix per instance and
BCE-with-logits in [`train_diffusion.py`](../src/vrp_diffusion_quantum/train/train_diffusion.py).

The audit in [`label_audit_s7799_decision.md`](label_audit_s7799_decision.md) found several
feasible, near-best solver references that disagree on route membership. Only 239/500 CVRP100
sources passed the strict matrix-stability gate. Increasing solver budgets from 40 to 80 and 120
seconds did not reliably remove the disagreement.

This shows that the **available supervision is empirically multi-valued**. It does not prove that
every competitive candidate is globally optimal, that candidates are equally valid, or that an
accepted case has a mathematically unique answer. Audit acceptance means only that a case passed
the declared cost and matrix-stability gates.

Ambiguity is not the complete explanation for current end-to-end quality:

| Size | Exact untouched-test F1 | Strict test matrix acceptance |
|---|---:|---:|
| CVRP20 | 0.4206 | 20/20 |
| CVRP50 | 0.4522 | 18/20 |
| CVRP100 | 0.4305 | 9/20 |

CVRP20 has stable supervision but still has low full-chain F1. Reverse diffusion and route
decoding are therefore system-wide primary bottlenecks. Ambiguity-aware supervision is a targeted
second workstream, especially for CVRP100.

## Workstream A: full-chain sampling and routing utility (primary)

### A1. Establish decoded route-quality evaluation

Convert predicted matrices into routes and report on fixed frozen examples:

- capacity-feasible rate and violation counts;
- decoded cost and gap to the strongest audited reference;
- vehicle count and route-size plausibility;
- decoding, repair, classical-refinement, and quantum-refinement runtime;
- matrix metrics before and after repair/refinement.

Matrix recovery alone cannot establish whether predictions are useful for routing. This evaluator
is required before generating a larger label corpus.

### A2. Diagnose reverse-process degradation on stable CVRP20

With the same checkpoint and examples, compare:

- exact 700-step ancestral sampling;
- deterministic/probability-based transitions;
- fewer steps using explicitly defined timestep subsequences;
- realistic initial positive prevalence versus unconstrained noise;
- alternative Bernoulli schedules and 100/300/700 training timesteps;
- constraint-guided versus unguided sampling.

Keep inference seed, examples, timestep definition, and hard threshold fixed. Never compare runs
with different `step_stride` values as if they used the same sampler.

### A3. Select checkpoints using full-chain evidence

Noisy-time AUC remains a useful diagnostic but should not be the only checkpoint criterion.
Future bounded runs should evaluate a fixed full-chain validation panel every one or two epochs,
keep its sampler fixed, and select primarily by full-chain F1 or decoded route quality.

## Workstream B: ambiguity-aware supervision and evaluation

### B1. Partition-invariant evaluation

For audited ambiguous cases, report:

1. canonical-reference matrix metrics;
2. best-of-K metrics, labeled **oracle best-known-reference agreement**;
3. decoded feasibility and route-cost gap.

Freeze the candidate set and competitive-cost tolerance before evaluating models. Use one declared
matching criterion to choose the oracle reference; do not choose a different reference for every
metric. Oracle agreement does not prove that a generated partition is feasible or optimal.

### B2. Separate audit categories

| Audit category | First experimental treatment |
|---|---|
| Cost-stable, matrix-stable | Hard strongest target |
| Cost-stable, matrix-unstable | Stochastic or consensus target |
| Cost-unstable, matrix-stable | Reduced-confidence hard target plus exclusion control |
| Cost-unstable, matrix-unstable | Masked/exclusion control before consensus training |

The 18-instance 120-second follow-up was deliberately enriched for difficult cases and cannot
estimate a population percentage. The corpus-wide result remains 261/500 CVRP100 sources needing
review under the declared audit.

### B3. Compute-matched stochastic-reference probe

For each ambiguous source, select one precomputed competitive hard reference per epoch using a
deterministic `(seed, epoch, source_id)` rule. Every source contributes at most one example per
epoch, matching policy-v2 instance weight. Load candidate matrices once; do not read solver JSON
inside every `Dataset.__getitem__`. Use the selected hard matrix for both `m_t` and the loss target.

This is a probe, not assumed equivalent to consensus BCE. Dynamic class weights and
reference-dependent forward noising make the objectives different.

### B4. Explicit consensus targets and confidence masks

For cost-stable, matrix-unstable sources, construct:

```text
target_probability[i,j] = mean_k(M_k[i,j])
target_confidence[i,j]  = agreement_weight(M_1[i,j], ..., M_K[i,j])
```

Do **not** weaken `CVRPExample.constraint_matrix`; it should remain binary, symmetric,
zero-diagonal, and consistent with `solution.routes`. Add training-only target/confidence fields or
a versioned sidecar referenced by the training-label manifest.

The first consensus experiment should preserve valid hard diffusion states:

1. deterministically sample one competitive hard reference;
2. construct `m_t` from that hard reference;
3. predict clean membership probabilities;
4. score against the consensus target;
5. mask or down-weight disputed pairs.

Directly noising fractional `m_0` changes the Bernoulli forward process and belongs in a separate,
explicitly derived experiment. Freeze positive-class weights from the corpus, or account for them
explicitly, so experimental arms remain comparable.

### B5. Exclusion control

Exclude the least trustworthy category as a control, not as the assumed solution. Report unique
sources and optimizer steps, and do not silently give retained instances extra updates.

## Controlled experiment

After Workstream A supplies a fixed evaluator, compare:

| Arm | Stable sources | Ambiguous sources | Weight per source |
|---|---|---|---:|
| Policy-v2 baseline | Hard | Hard strongest | 1 |
| Exclusion control | Hard | Excluded by declared category | 0 or 1 |
| Stochastic reference | Hard | One hard reference per epoch | 1 |
| Consensus target | Hard | Mean competitive matrix | 1 |
| Consensus plus mask | Hard | Mean matrix and pair confidence | 1 |

Keep fixed across arms:

- source membership and declared exclusions;
- optimizer-step budget and effective batch size;
- seeds, GAT checkpoint, diffusion schedule, and timestep sampler;
- positive-class weighting definition;
- validation membership and candidate sets;
- full-chain seed, timestep sequence, and threshold policy;
- decoder and refinement budgets.

Report overall and per-size noisy-time metrics, exact full-chain metrics, oracle agreement, decoded
feasibility/cost, calibration, runtime, and memory. Require a CVRP100-specific improvement without
material CVRP20/CVRP50 regression before adopting ambiguity-aware supervision.

## Recommended execution order

```text
1. Implement matrix-to-route feasibility and cost evaluation
2. Diagnose exact full-chain degradation using stable CVRP20
3. Freeze canonical plus oracle best-of-K evaluation
4. Run the compute-matched stochastic-reference probe
5. Add explicit consensus-target and confidence-mask schema
6. Compare baseline, exclusion, stochastic, consensus, and masked arms
7. Repeat exact validation and decoded route evaluation
8. Generate more strong labels only after a verified model-side gain
```

## Decision gates

- Do not generate a broad label corpus merely to improve noisy-time metrics.
- Do not use untouched test data for sampler, threshold, checkpoint, or ambiguity-policy selection.
- Adopt a new target representation only if it improves fixed full-chain or decoded-route
  validation under matched compute.
- Use the untouched test once after the complete sampler, target, checkpoint rule, and decoder are
  frozen.
- Preserve originals and version every target sidecar, candidate set, config, manifest, and result
  with hashes.
