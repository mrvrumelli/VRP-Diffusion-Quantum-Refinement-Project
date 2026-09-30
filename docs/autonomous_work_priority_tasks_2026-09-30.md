# Prioritized tasks following the autonomous-work review — 2026-09-30

Evidence and rationale: [findings report](autonomous_work_review_2026-09-30.md).
Original task numbers refer to the
[long-compute guide](autonomous_long_compute_guide_2026-09-24.md). The `R` identifiers below
distinguish corrective work from those historical tasks.

The first priority is trustworthy inference and evaluation. New labels and larger training runs
should follow evidence that they address the remaining limitation. The sampler mismatch and
missing depot input are confirmed; their combined contribution to the F1 deficit is not known.
Calibration and BatchNorm remain hypotheses. The small sampler correction probe improved F1
from 0.4677 to 0.4780, so it is not a demonstrated complete solution.

**Status:** R1-R12 have reached their bounded completion/stop gates under the user's execution
authorization. See the [execution log](corrective_execution_2026-09-30.md),
[experiment report](corrective_results_2026-09-30.md) and
[decision manifest](evidence/corrective_20260930/candidate_baseline.json).
Checked items include resolved conditional gates, not a claim that every possible follow-up
was launched. The source-count expansion failed its rule, so no finalist was sent to the
untouched test. Stronger final references remain conditional on a future qualified finalist.
The paper F1 target is unresolved; the classical/paper freeze is deferred and downstream
quantum work remains gated. Historical checkpoints and scores are preserved.

**Priority order**

P0 establishes correctness and comparable measurements. P1 diagnoses the remaining model and
data problems. P2 confirms a selected recipe and closes benchmark gaps. Within a priority,
follow the numbered order; a task may proceed once its stated dependencies are met.

| Order | Priority | Task | Depends on | Main resource |
|---|---|---|---|---|
| R1 | P0 | Correct the evidence summary and backlog | Findings report | Documentation / artifact inspection |
| R2 | P0 | Establish one evaluation contract and safe panels | R1 | Code / dataset inspection |
| R3 | P0 | Fix soft-posterior composition with legacy compatibility | R1 | Code / mathematical checks |
| R4 | P0 | Re-score existing checkpoints under the corrected protocol | R2, R3 | Bounded GPU inference |
| R5 | P1 | Repair source-level datasets and label provenance | R2 | CPU / existing artifacts |
| R6 | P1 | Run small learning and inference sanity checks | R3, R5 | Bounded CPU/GPU diagnostics |
| R7 | P1 | Test depot-aware conditioning | R4, R6 | Bounded paired training |
| R8 | P1 | Test loss and probability calibration | R4, R6; freeze the input recipe within this comparison | Bounded paired training/inference |
| R9 | P1 | Test BatchNorm and size-batch ordering | R4, R6; freeze other choices within this comparison | Bounded diagnostics/training |
| R10 | P2 | Rebuild the empirical scaling result with repeated seeds | R5–R9 and a selected recipe | GPU training |
| R11 | P2 | Complete Task 4 and report all R/C/RC cells | R2–R4; R10 if evaluating its new finalists | CPU baselines / GPU inference |
| R12 | P2 | Reassess the classical freeze and paper-track gates | Relevant R4–R11 evidence | Analysis / manifests |

R11 can evaluate existing checkpoints without waiting for a new learning curve. It must remain
visible in the backlog while the model investigations proceed. The original CPU/GPU resource
rules continue to apply: one heavy job per resource pool, with any allowed CPU/GPU overlap
following the guide's limits.

**R1 — Correct the evidence summary and backlog**

- [x] Add dated corrections to the affected reports and guide, retaining the original run values.
- [x] Qualify Task 6's N100 victory until both methods use the same gap definition and inference
  conditions. Distinguish validation-based candidates from independently confirmed improvements.
- [x] Correct Task 5's distinct-source counts and restore N50/500 as an improvement candidate:
  26.07% versus the historical 29.39% comparator.
- [x] Correct the OOD acceptance claim: only N100's acceptance rate improved in that comparison.
- [x] Remove the unsupported claims of F1 confidence intervals, identical filtered/unfiltered
  panels, and five eliminated explanations from the current interpretation.
- [x] Restore original Task 4 and the nine R/C/RC-by-size results as outstanding work. Record
  launch-failure and cache-resume explanations as unresolved where causality was not established.
- [x] Archive the review's small evidence JSONs and reproducers in a durable, tracked location
  if `outputs/` artifacts are excluded from version control.

**Completion gate:** one consistent status table distinguishes confirmed observations,
unconfirmed candidates, hypotheses and pending tasks. No remaining summary presents Tasks 7–8
as the only unfinished work or marks a methodological rejection as a completed experiment.

**R2 — Establish one evaluation contract and safe panels**

- [x] Inventory training sources for every compared GAT, denoiser and policy, including alternative
  references and augmented descendants. Build comparison panels outside the union of those
  training sources. Keep development/selection panels separate from final untouched tests.
- [x] Record instance IDs, source identities, hashes, labels, selection seeds and checkpoint hashes.
  Reuse existing suitable data first; flag any verified reference shortage separately.
- [x] Report generated positive-class F1 at fixed threshold 0.5, per size and pooled. Report the
  equal-size average separately. Label noisy-time and validation-tuned-threshold results explicitly.
- [x] Export per-instance TP/FP/FN, AUC inputs or sufficient prediction artifacts, decoded cost,
  reference cost, feasibility and runtime. Bootstrap instances rather than treating dependent
  matrix entries as independent observations. Recompute pooled F1 from resampled TP/FP/FN.
- [x] Calculate both mean instance-relative gap and ratio-of-total-cost gap for every method.
  Predeclare one primary metric for each comparison and use it consistently on both sides.
- [x] Freeze sampler version, inference steps, batch size, precision, start count and seeds.
  Distinguish a matched-prior decoder comparison from an end-to-end matched-time comparison.
- [x] Add meaningful checks for metric aggregation, source overlap detection, and paired bootstrap
  calculations on hand-computable examples.

**Completion gate:** the same per-instance results yield identical metric definitions across
evaluators; every comparison panel passes the source-overlap audit; F1 and gap uncertainty are
separately identified. The previous filtered panel must not be reused for the unfiltered model:
1,352 of its 1,500 instances occur in that model's training set.

**R3 — Fix soft-posterior composition with legacy compatibility**

- [x] Keep the conditional posterior for hard clean states distinct from the learned reverse
  transition. For a predicted clean probability `p`, mix the two normalized hard-state posteriors
  with weights `1-p` and `p`.
- [x] Add independent numerical checks for fractional probabilities, including the `t=20 -> s=0`,
  `x_t=1`, `p=0.2` counterexample. Cover adjacent/skipped transitions, the clean endpoint,
  probability normalization and binary/symmetric matrix sampling.
- [x] Preserve the old inference behavior under an explicit legacy identifier. Record the sampler
  version in new evaluation artifacts and avoid silently reinterpreting old checkpoint scores.
- [x] Route standalone evaluation and policy-prior generation through the same corrected path.
  Update the alignment evidence ledger where the implementation and prior claims disagree.

**Completion gate:** independent soft-state enumeration agrees with the corrected reverse
transition, the counterexample returns approximately 0.212728, and explicit legacy evaluation
remains reproducible. Relevant diffusion, inference and checkpoint checks pass.

**R4 — Re-score existing checkpoints before retraining**

- [x] First compare legacy and corrected samplers on a small fixed panel with identical random
  seeds. Expand to a predeclared validation panel only after the numerical checks pass.
- [x] Compare pilot, full filtered, full unfiltered and N20-only paper-mode checkpoints on appropriate
  common panels. A comparison of old checkpoints is diagnostic; it does not isolate label
  filtering causally when other training choices or learned GATs also differ.
- [x] Recheck a bounded inference-step curve, including 50 steps, using the same instances and
  declared sampling seeds. Do not infer sampler correctness from the shape of that curve.
- [x] Reassess the Task 5 candidates N20/1,000, N50/500 and N100/1,000 against their champions.
  Reassess the three Task 6 N100 policy seeds with the corrected metric and matched prior settings.
- [x] Save per-instance paired differences and uncertainty. Check seed sensitivity by instance ID;
  equal aggregate F1 or mean cost alone does not establish identical generated matrices/routes.

**Completion gate:** a reproducible comparison table reports effect sizes, uncertainty, inference
settings and qualification of every retained claim. A sampler defect can be fixed without an F1
gain; a negative result is valid and must not trigger an automatic large training run.

**R5 — Repair source-level datasets and label provenance**

- [x] Select nested sets of 500/1,000/2,000 independent sources per size, then attach all eligible
  references for each selected source. Verify nesting, source counts and held-out separation.
- [x] Preserve existing Task 5 subsets as historical experiments. Record their actual N100 counts
  of 456/835/1,375 and N50 counts of 480/934/1,754 in their new interpretation.
- [x] Materialize a declared single-reference HGS dataset from existing candidates. A true
  single-solve policy must choose a predetermined seed rather than the best of several runs.
  If best-of-seeds is used, label that policy accurately.
- [x] Freeze source-level train/validation/test assignments before applying label filtering.
  For a filtering ablation, use the same target-reference selection and evaluation panel in both
  arms; only the training-source inclusion rule should differ.
- [x] Validate actual dataset manifests against the alignment claim instead of accepting only
  the `label_policy` string. Report sources, references and training examples as separate counts.
- [x] Record any shortfall against the declared training-set target after holdouts. Do not assume
  50,580 solved sources automatically provide 50,000 training examples plus all required holdouts.

**Completion gate:** manifests enforce source-level counts, nesting, split separation and actual
label provenance. No new HGS campaign is needed just to repair materialization or metadata.

**R6 — Run small learning and inference sanity checks**

- [x] Freeze a tiny set with one consistent hard target per source. Check that the loss falls,
  gradients and intended parameter updates are present, and predicted matrices fit the targets.
- [x] Measure denoising at fixed low, middle and high noise levels, separately from generation
  starting at the random prior. Record both train-set fit and held-out behavior.
- [x] Use an oracle clean-state predictor to check reverse-sampling endpoints independently of
  learned-model quality. Verify that frozen components remain frozen.
- [x] Set a small runtime/step budget and a numerical fit criterion before running; record failures
  as diagnostics rather than increasing the dataset until the check appears to pass.

**Completion gate:** explain any tiny-set fit failure and separate learning, probability prediction
and reverse-generation errors before scaling. Feasible repaired routes alone do not satisfy this gate.

**R7 — Test depot-aware conditioning**

- [x] Choose and document one explicit representation, such as depot-relative coordinates plus
  depot distance, or a depot node in the encoder. Version the input/checkpoint schema.
- [x] Carry the information consistently through GAT pretraining, diffusion training, augmentation
  and inference. Preserve loading and behavior of legacy checkpoints.
- [x] Verify that changing only the depot changes the intended features and that geometric
  transforms preserve the feature convention. Use the exact six-customer counterexample.
- [x] Train a bounded matched pair: identical source splits, seeds, budgets, losses and sampler,
  with depot information as the changed factor. Retrain comparable GAT controls where needed.
- [x] Record this as an explicit reconstruction assumption or diagnostic variant, pending evidence
  about the authors' input representation. Do not silently redefine the paper baseline.

**Completion gate:** quantify generated-F1 and route-cost effects on the frozen validation panel.
Confirm promising effects with another training seed before selecting the new input recipe.
The information-loss proof alone does not establish the size of the practical improvement.

**R8 — Test the objective and probability calibration**

- [x] Document the relationship between the current weighted clean-target BCE and the intended
  diffusion objective. Distinguish a deliberate surrogate from a claimed equivalent objective.
- [x] With inputs and sampler fixed, compare weighted and unweighted BCE under matched training
  budgets. Test an equation-consistent objective only after its definition and numerics are checked.
- [x] Examine probability calibration across noise levels. Fit any calibration or threshold on
  development data and freeze it before evaluating held-out performance.
- [x] Separate changing probabilities inside reverse transitions from changing only the final
  classification threshold. Keep fixed-threshold generated F1 as a visible primary measure.
- [x] Do not assume subtracting one global weight correction exactly calibrates a model trained
  with changing per-batch class weights; validate any such approximation empirically.

**Completion gate:** a controlled comparison identifies whether objective/calibration changes
improve generation and route utility, with validation uncertainty and seed confirmation for a
selected change. An apparent gain from tuning on the scored test labels does not count.

**R9 — Test BatchNorm and size-batch ordering**

- [x] Inspect running statistics and model outputs by size and noise level. Compare training and
  inference behavior without altering the saved checkpoint's persistent buffers.
- [x] Shuffle the order of homogeneous-size batches and compare against the current fixed
  N20-then-N50-then-N100 traversal. Keep the same examples and update budget.
- [x] If needed, test normalization alternatives as separately labeled variants, changing one
  factor per comparison. Test per-size models separately from batch-order effects.
- [x] Enforce the calibration boundary: no normalization recalibration or test-buffer fitting
  was performed. Train/eval probes used disposable clones, preserving original buffers.
  Batch sizes and the separate N100-only control protocol are recorded.

**Completion gate:** report whether batch order or normalization materially changes generated F1
and route cost. If the hypothesis is unsupported, record that result and stop this branch.

**R10 — Run the corrected learning curve with replication**

- [x] Freeze the selected architecture, objective, normalization, sampler and checkpoint rule.
  Use R5's source-based subsets and a sufficiently sized development panel for selection.
- [x] Start with adjacent source-count points on the priority size, using repeated training seeds.
  Expand the remaining points/sizes only when the result and declared compute budget justify it.
- [x] Keep epochs and stopping policy comparable; report optimizer steps and elapsed time because
  fixed epochs naturally give larger datasets more updates. If a fixed-compute curve is also run,
  label it as a separate comparison.
- [x] Before launch, declare the practically meaningful route-gap improvement, uncertainty rule,
  seed budget and plateau stop condition. Do not equate overlapping separate CIs with equality.
- [x] Apply the paired-validation finalist gate. Both seeds failed the declared expansion rule;
  no finalist qualifies, so the 96 reserved test sources remain unscored. A future qualifying
  finalist would require stronger reference labels and one frozen test evaluation.

**Completion gate:** a replicated source-count curve distinguishes data-size effects from training
variation. A plateau, reversal or inconclusive result is an acceptable outcome; no indefinite tuning.

**R11 — Complete matched baselines and the OOD breakdown**

- [x] Execute the outstanding original Task 4 comparison: policy, diffusion-only, PyVRP and
  OR-Tools on identical held-out instances with declared budgets and end-to-end timing.
- [x] Replace the guide's training-pool example command with the correct common held-out panel.
- [x] Expand CVRPLIB beyond the smoke case, recording reference costs and distance conventions.
- [x] Report all nine R/C/RC-by-size cells for champion and merged checkpoints, including counts,
  route gaps, generated F1, feasibility and uncertainty. Identify the stable-label selection and
  its coverage; do not generalize that panel to every OOD instance without further evidence.
- [x] Keep matched-prior decoder effects separate from comparisons of complete methods under
  matched runtime budgets. Include reference costs and runtime accounting in saved outputs.

**Completion gate:** one matched-method report plus a CVRPLIB table and nine-cell OOD table exists.
Conclusions describe the evaluated checkpoints and panels, not every possible training strategy.

**R12 — Reassess baseline and downstream gates**

- [x] Update the candidate baseline manifest with selected checkpoint/data hashes, sampler/input
  versions, evaluation settings and independently confirmed evidence.
- [x] Classify each finding as reproduced, supported, inconclusive or rejected. Preserve the
  distinction between a locally improved recipe and a paper reproduction.
- [x] Reassess the external 0.823-at-50-steps target with its unresolved metric/author assumptions
  explicitly listed. Do not treat the highest unrelated F1 value as meeting that target.
- [x] Keep original paper Tasks 11–14 subject to their stated prerequisites and the current pause
  status. Improving F1 in a diagnostic does not automatically complete those stages.
- [x] Keep quantum Tasks 7–8 behind the classical freeze, D.1–D.3, and the guide's relevant
  paper-track dependencies. Record any future deliberate scope change explicitly.

**Completion gate:** the freeze decision is backed by reproducible artifacts and actual completed
prerequisites. If reproduction remains unresolved, the report states the remaining gap and the
bounded next hypothesis rather than declaring completion.

**Required record for each experiment**

Before launch, record the hypothesis, baseline, changed factor, fixed splits, seeds, step/runtime
cap, selection metric and stopping rule. After completion, attach the resolved config, code and
checkpoint versions, dataset/source manifest, per-instance outputs, aggregate metrics and
uncertainty. Mark a task complete when its stated gate is met, including a well-supported negative
answer. A failed or rejected launch is a status to explain, not a successful experiment.
