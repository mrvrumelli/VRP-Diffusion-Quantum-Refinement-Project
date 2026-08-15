# Resolving CVRP100 route-partition ambiguity

## The problem, precisely

The model target is the customer-customer route-membership matrix built by
[`build_constraint_matrix`](../src/vrp_diffusion_quantum/utils/constraint_matrix.py):
`M[i,j] = 1` iff customers `i` and `j` are in the same route. Training uses a single hard
binary `M` per instance and plain BCE-with-logits against it
(`diffusion_matrix_bce_loss` in
[`train_diffusion.py`](../src/vrp_diffusion_quantum/train/train_diffusion.py)).

The audit (`label_audit.py`, see [`label_audit_s7799_decision.md`](label_audit_s7799_decision.md))
showed that for CVRP100, multiple route partitions routinely tie on cost
(`near_best_relative_tolerance`) but disagree on which customers share a route
(`max_near_best_matrix_disagreement`, `max_customer_membership_instability`). Only 47.8% of
CVRP100 sources are `matrix_target_accepted`. Longer solving (40s → 80s → 120s) fixed
under-convergence but barely touched this disagreement (80s: only 1/10 matrix-only-ambiguous
cases became accepted; 120s: 1/18).

**This means the ground truth itself is multi-valued for roughly half of CVRP100.** No amount of
extra solving or extra same-shape training data fixes that; a single hard binary target is the
wrong representation for those instances. The two probing rounds already ruled out one
lever (solver time) and one framing (append the alternates as duplicate examples — the
`mixed/multi-reference` policy's apparent AUC gain vanished once compute-matched, per the
policy-v2 table in the decision doc). What's left is changing *what* the model is asked to fit
on ambiguous instances, not how much data or compute it gets.

## Candidate approaches

### 1. Soft/averaged target over near-best matrices (recommended first experiment)

For each accepted-cost-but-matrix-unstable instance, `label_audit.py` already computes the set
of `near_best` PyVRP candidates. Instead of picking one and discarding the rest, build
`M_soft[i,j] = mean_k(M_k[i,j])` across the near-best candidate matrices and train with BCE
against `M_soft` (soft targets, not one-hot) rather than replicating each candidate as a
separate hard-labeled example.

- Why this is different from the multi-reference policy already tried: that policy duplicated
  *examples* (more optimizer steps per instance), which is exactly the confound the decision doc
  identified. A soft target changes the *loss target* for the same one-example-per-instance,
  compute-matched setup — it never leaves the audit's step-matched protocol.
- Implementation surface: add a `soft_multi_reference` `LabelMode` to
  `TrainingLabelPolicy`/`materialize_training_labels` in
  [`training_labels.py`](../src/vrp_diffusion_quantum/data/training_labels.py) that writes a
  float-valued matrix (or stores the `near_best` route list and averages at load time) instead of
  a single `LabeledSolution`. `diffusion_matrix_bce_loss` already takes arbitrary float targets
  via `m_true` — soft labels work unmodified as long as `dataset.py`'s loader accepts non-binary
  targets.
- Where to apply it: only on `needs_review` / `matrix_target_accepted=False` instances. Leave
  stable instances (CVRP20, most of CVRP50, the accepted ~48% of CVRP100) as-is with hard
  targets, since the audit already confirms those have a real single answer — softening them
  would only add noise.

### 2. Mask ambiguous pairs instead of guessing them

For each instance, `label_audit.py`'s customer-instability computation (`np.mean(first_m !=
second_m, axis=1)`) already identifies *which specific `(i,j)` pairs* disagree across near-best
solutions, not just that the instance as a whole is unstable. Most pairs in a CVRP100 instance
are *not* ambiguous — only the boundary customers near a route split are.

- Build a per-instance confidence/consensus mask: `agree[i,j] = 1` if all near-best candidates
  agree on whether `i,j` share a route, `0` otherwise. Train BCE only on `agree==1` pairs (set
  `pair_ok` in `diffusion_matrix_bce_loss` to `pair_ok & agree`), or down-weight disagreeing pairs
  instead of dropping them.
- This is likely higher-value than approach 1 for CVRP100 specifically, because the audit's own
  numbers say the ambiguity is localized (mean matrix disagreement is much smaller than "half the
  matrix is wrong" — it's boundary customers), so most of a CVRP100 instance's signal is still
  trustworthy and shouldn't be diluted by averaging.
- Combine with #1: use the mask to decide *where* to blend probabilities, and use the hard/near-1/near-0
  target elsewhere.

### 3. Stochastic one-reference-per-epoch resampling

Instead of writing multiple materialized files or a fixed soft target, resample which near-best
candidate is used as the hard label each epoch (fixed seed per epoch, e.g.
`candidate = near_best[epoch_seed % len(near_best)]`). Over many epochs the expected loss
approximates the soft target from #1, without a new label format — implemented entirely in the
`Dataset.__getitem__`/collate path in [`dataset.py`](../src/vrp_diffusion_quantum/data/dataset.py).

- Cheaper to prototype than #1 since it needs no new materialized dataset, only a dataset-side
  hook that reads the already-cached `near_best` candidates from the audit directory.
- Downside: noisier per-epoch loss, and still forces one full hard answer per step rather than
  reflecting genuine uncertainty in a single forward pass — worse than #1 for eval-time
  calibration, better as a quick sanity check that soft-target-style training helps before
  investing in #1's data materialization.

### 4. Exclude irreducibly ambiguous instances from binary-target training

For the ~9% of CVRP100 (per the 120s follow-up: only 1/18 resolved) that stay matrix-unstable
even at 120s, drop them from binary-target training entirely and route them to a held-out
"structurally ambiguous" evaluation bucket instead of forcing a label. This is the cheapest
option and a reasonable floor/control to compare 1–3 against — it directly tests whether the
remaining ambiguous-but-included instances are net helpful or net harmful to include at all.

### 5. Reframe evaluation to be partition-invariant

Independent of what training does, `predict_matrix.py` / full-chain eval currently scores against
one frozen reference matrix per validation/test instance. For instances with multiple accepted
near-best partitions, evaluate against the *best-matching* near-best reference (max AUC/F1 over
the accepted candidate set) rather than a single arbitrary one. This doesn't change training but
stops the eval from penalizing the model for recovering a *different, equally valid* partition
than the one frozen as ground truth — which is likely inflating the apparent full-chain error on
CVRP100 independent of any modeling fix above.

## Recommended sequence

1. Implement #5 (partition-invariant eval) first — it's cheap, changes nothing about training,
   and gives a cleaner baseline to measure #1–#3 against. Some of the current CVRP100 full-chain
   gap may already shrink from this alone.
2. Run #3 (stochastic resampling) as a fast, low-engineering probe on the existing
   `needs_review` CVRP100 subset, compute-matched against the current policy-v2 baseline.
3. If #3 shows a real, compute-matched improvement, invest in #1 (materialized soft targets) plus
   #2 (pair masking) as the production label policy for CVRP100 — these are more principled and
   give better-calibrated probabilities than epoch-level resampling.
4. Use #4 as a control arm in the same comparison: does keeping the worst ~9% help or hurt once
   1–3 are in place?

## Guardrails (carried over from the policy-v2 comparison)

- Match optimizer steps/epochs exactly across arms — the original multi-reference confound was
  purely a step-count artifact, not a real modeling gain.
- Keep CVRP20/CVRP50 and the already-accepted CVRP100 instances on hard targets; only change the
  representation for instances the audit already flagged as `needs_review`.
- Never select among these options using the untouched test split — use the same frozen
  validation set and fixed seeds as the existing comparisons in
  [`3060ti_training_report.md`](3060ti_training_report.md).
- Record dataset hashes and manifests for any new materialized label set, same as
  `training_label_manifest.json` does today.
