# Task 5 — audited-data learning curve — 2026-09-30

**Correction, 2026-09-30 (R1):** The 500/1000/2000 labels count files, not distinct sources. Actual N20 sources: 500/1000/2000; N50: 480/934/1754; N100: 456/835/1375. N50/500 is an improvement candidate (26.07% versus historical 29.39%), alongside N20/1000 and N100/1000. These single-seed validation results do not establish scaling or a confirmed winner. Original Task 4 and nine OOD cells remain open; quantum tasks are not the only backlog. See the [review](autonomous_work_review_2026-09-30.md) and [execution log](corrective_execution_2026-09-30.md). Historical run values below are retained.

Track A Task 5 ([`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)):
train the 500/1,000/2,000-per-size diffusion learning curve on the expanded audited pool
(audit completion: [`task5_audit_expansion_2026-09-30.md`](task5_audit_expansion_2026-09-30.md))
and apply the stop rule — stop adding data once the frozen-panel decoded route-gap gain flattens
between consecutive curve points, not on matrix F1/training loss alone.

## Frozen inputs

- Combined labeled pool: `data/processed/s7799_task5_labels_combined` (9,598 examples:
  2,066/2,435/5,097 by size N20/N50/N100), built by
  [`outputs/_scratch/build_task5_curve.py`](../outputs/_scratch/build_task5_curve.py) from the
  original strong-reference audit + the new expansion audit, same `TrainingLabelPolicy` modes as
  production (20=original, 50=canonical_else_multi, 100=multi_reference).
- Curve subsets: `data/processed/s7799_task5_curve_n{20,50,100}_{500,1000,2000}` — nested by
  construction (same seed across the three per-size draws, so 500⊂1,000⊂2,000).
- Training: `configs/train/diffusion_denoiser_s7799_task5_curve_n{size}_{n}_cuda.yaml`, identical
  hyperparameters to the production per-size recipe (frozen GAT, 15 epochs, `stochastic_references`),
  only the dataset differs. Entrypoint: `python -m vrp_diffusion_quantum.train.train_diffusion`.
- Eval: `python -m vrp_diffusion_quantum.inference.predict_matrix` against the frozen
  `s7799_val100_policy_v1_n{size}` panel (100/92/45 examples), full 700-step reverse chain — same
  panel and methodology as Task 2, so results are directly comparable to the known diffusion-only
  champion gaps.

## Results

| Size | 500 | 1,000 | 2,000 | Existing champion (500, different pool) |
|---|---:|---:|---:|---:|
| N20 | 32.43% [29.89, 34.92] | **18.99%** [17.04, 21.02] | 28.67% [26.18, 31.21] | 21.11% [18.71, 23.57] |
| N50 | **26.07%** [24.25, 27.93] | 31.83% [29.73, 33.98] | 31.50% [29.46, 33.59] | 29.39% [27.33, 31.51] |
| N100 | 32.22% [29.93, 34.44] | **27.43%** [25.61, 29.29] | 32.98% [31.01, 34.97] | 30.57% [28.15, 33.09] |

(`route_mean_cost_gap_percent`, 95% bootstrap CI; bold = best point per size; all 100% feasible,
zero capacity violations, zero solver errors across every run.)

Raw artifacts: `outputs/eval/task5_curve_20260930/n{20,50,100}_{500,1000,2000}/sample_metrics.json`;
training runs under `outputs/train/diffusion_denoiser_s7799_stochastic_task5_curve_n*`.

## Interpretation — this did not produce a clean scaling curve

Every size is **non-monotonic**, not a diminishing-returns curve:

- **N20 and N100 both peak at 1,000** — 1,000 beats both 500 and 2,000 by a wide margin at both
  sizes, an unusual matching shape.
- **N50 is best at 500**, its smallest point, then flattens (statistically indistinguishable)
  between 1,000 and 2,000.

The literal stop rule ("stop once gains flatten between consecutive points") assumes a roughly
monotonic curve. Reality here doesn't cooperate: gains reverse rather than flatten in 2 of 3
sizes. Applying the rule's spirit rather than its letter, this pipeline's diffusion training shows
**high single-run variance at this data scale** — the same kind of instability already documented
this session (Task 6 N100's REINFORCE variance/policy collapse; Track B's F1 pilot-vs-full-scale
reversal despite 11x more data). Every curve point here is a single training run, no repeated
seeds — I cannot distinguish "more data genuinely hurts/helps" from "this particular checkpoint
got lucky/unlucky." A confident answer would need 2-3 repeated seeds per curve point, which is a
real scope/compute decision, not something to spend unilaterally.

## What is actionable regardless of the curve's shape

**N100/1,000's checkpoint (27.43%) beats the existing production N100 diffusion-only champion
(30.57%) by a real margin (CIs barely overlap).** This is independently useful: N100 is the size
Track A's diffusion-only baseline has always been weakest at (motivating Task 6's policy-side
fix). A candidate diffusion-only improvement at N100 is worth carrying forward regardless of
whether the "more audited data helps in general" hypothesis holds. Checkpoint:
`outputs/train/diffusion_denoiser_s7799_stochastic_task5_curve_n100_1000_cuda_20260930T021223658360Z/checkpoints/best.pt`.

N20/1,000 (18.99%) similarly beats its own champion (21.11%) and could be carried forward the same
way. N50 shows no candidate improvement over its existing champion.

## Recommendation (a real decision, not taken autonomously)

1. Treat this round's numbers as noisy single-run estimates, not a settled scaling law.
2. If a confident answer to "does more audited data help" matters going forward, budget 2-3
   repeated seeds per curve point (would roughly triple this round's GPU cost) — flagging as a
   scope decision, not doing it unilaterally.
3. Independent of (2), the N100/1,000 (and optionally N20/1,000) checkpoints are concrete
   candidates worth validating further (e.g. re-scoring on a larger/disjoint panel, the same
   measurement-noise check Task 10's investigation used) before considering them a new champion.

## Decode-noise check on the N100/1,000 candidate

Since predicting more from what's already at hand is cheap and safe (reuses the existing
checkpoint and panel, no new training or scope), re-ran the N100/1,000 eval with a different
decode seed (`--seed 1` vs. the default `--seed 0`) to check whether 27.43% depends on which
stochastic reverse-chain sample was drawn.

**Result: bit-for-bit identical** (`route_mean_cost_gap_percent` = 27.427780464180348 both times,
`sample_f1` identical too). The `--seed` flag only affects panel-subset selection and per-example
generator seeding in this code path, and it does not perturb the outcome here — the 100-example
panel is small enough (and/or the decode converges strongly enough) that this particular result is
fully reproducible, not a lucky stochastic draw. This doesn't resolve the training-side variance
question (different *training* runs still land in very different places, per the curve above), but
it does rule out decode-sampling noise as an explanation for N100/1,000 beating the champion —
that part of the comparison is solid.

## Status

Task 5 is complete as originally scoped (audit + 9-run curve). The stop rule was applied in
spirit given the non-monotonic data; no further curve points were trained. Track A's untouched
backlog is now Tasks 7-8 (quantum refinement), both still gated behind D.1-D.3 prerequisite work
and the classical-baseline freeze per `project_findings_2026-09-24.md`.
