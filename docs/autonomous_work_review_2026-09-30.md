# Review of autonomous work, September 24–30, 2026

The solver audits and saved experiment results represent useful work, but several reports
overstate what the experiments establish. There are also concrete implementation and comparison
problems. The present evidence does **not** justify the conclusion that implementation errors
have been eliminated and that the remaining F1 gap must come from the authors' labeling budget.

I would prioritize correcting the inference/evaluation contract and investigating missing model
inputs before another labeling campaign, broad seed sweep, or quantum-refinement experiment.

Reviewed: all recent `task1_`, `task2_`, `task3_`, `task5_`, `task6_`, `task9_`, and `task10_`
reports; the complete [autonomous long-compute guide](autonomous_long_compute_guide_2026-09-24.md);
the alignment contract, evidence ledger and plans; source code; saved configurations, split
manifests, training CSVs, and evaluation JSONs. Repository HEAD was
`26deeb74c7ab4b4081a3f48c61ab7c9741f573fd`, with pre-existing local changes.

This review added documentation and bounded diagnostic artifacts only. It did not modify
production code, checkpoints, datasets, or the existing reports, and did not launch training or
solver-labeling jobs. New numerical checks are reproducible with
[verify_review.py](../outputs/autonomous_review_20260930/verify_review.py); results are in
[evidence.json](../outputs/autonomous_review_20260930/evidence.json).

**1. Confirmed: the reverse sampler's soft-prediction formula is not Equation 9's mixture. High priority.**

The paper's Equation 9 mixes conditional posteriors over possible clean binary states.
The implementation instead substitutes a soft prediction into a prior and then normalizes.
Those operations do not commute. [Paper, Equation 9](https://arxiv.org/html/2603.07568v1#S4.SS2.SSS2)

With `p = P_model(x_clean=1 | x_t, graph)`, the required mixture is:

```text
(1-p) * q(x_s=1 | x_t, x_clean=0)
  + p * q(x_s=1 | x_t, x_clean=1).
```

The current implementation computes:

```text
normalize(likelihood(x_t | x_s) * marginal(x_s | soft_clean=p)).
```

See [diffusion.py](../src/vrp_diffusion_quantum/models/diffusion.py), especially
`q_posterior_between_prob`, and its soft-input call in
[policy_support.py](../src/vrp_diffusion_quantum/inference/policy_support.py).

A numerical counterexample using the actual 1,000-step schedule: at `t=20 -> s=0`, with
`x_t=1` and predicted clean probability `p=0.2`, the implementation returns **0.975865**;
the mixture returns **0.212728**. This interval occurs near the end of the 50-step schedule.
The current formula can therefore retain a noisy edge far more strongly than the model's
clean-state prediction warrants.

The existing brute-force tests use hard clean states, where the formula is correct. The soft-input
test checks that probabilities are valid, not that they equal the mixture. Passing these tests
cannot establish correctness of the learned reverse transition.

I tested a corrected mixture in memory on the saved full-scale checkpoint, keeping the same 24
validation instances, seeds, and 50-step budget. F1 changed **0.467675 -> 0.477966**. This is a
small diagnostic panel, without replication or a significance test. It establishes a mathematical
defect and a measurable change in behavior; it does **not** establish that this correction alone
recovers the missing 0.35 F1. No production sampler was changed.

**2. Confirmed: the diffusion model cannot see the depot. High priority; F1 impact unquantified.**

[example_to_model_inputs](../src/vrp_diffusion_quantum/inference/predict_matrix.py) passes customer
coordinates, customer demands and capacity. The GAT features are `[x, y, demand, demand/capacity]`;
there is no depot node, depot coordinate, customer-to-depot distance, or depot-relative coordinate
in this denoiser path. Training uses the same customer-only representation.

I moved a validation instance's depot and verified that **all five returned model-input tensors
remain exactly equal**. The generated matrix distribution must therefore remain unchanged.
Yet optimal route membership can depend on depot position. An independent exact six-customer
enumeration demonstrates two different optimal partitions for identical customers/demands/capacity
under different depots; see [counterexample and coordinates](../outputs/autonomous_review_20260930/depot_counterexample.json)
and [reproducer](../outputs/autonomous_review_20260930/depot_counterexample.py).

This is a concrete information bottleneck, not solver-label noise. Its contribution to the observed
F1 deficit requires a depot-aware training ablation; the counterexample does not quantify that
contribution on N20/N50/N100. It affects both existing diffusion tracks.

**3. Confirmed training-recipe differences remain; their individual effects are hypotheses.**

The paper presents a variational diffusion objective. The implemented training loss is weighted
clean-matrix BCE. Equivalence has not been established or recorded in the evidence ledger.
[Paper, Equations 6–7](https://arxiv.org/html/2603.07568v1#S4.SS2.SSS2)

In [the full training config](../configs/train/diffusion_denoiser_paper_cmd_full.yaml),
`weighted_bce: true` and `pos_weight_power: 1.0` make the positive weight `negative_count / positive_count`.
[diffusion_matrix_bce_loss](../src/vrp_diffusion_quantum/train/train_diffusion.py) implements this.
For a fixed positive weight `w`, weighted BCE's optimal sigmoid output is
`w*p / (1-p+w*p)`, rather than the unweighted class probability `p`. The sampler nevertheless
uses this output directly as its clean-state probability, and generated F1 uses threshold 0.5.
This is an unresolved probability-calibration issue, not merely a choice of how to print F1.

The full-scale 1,500-instance evaluation has precision **0.3858**, recall **0.6037**, and AUC
**0.8280**. These are consistent with a model/probability problem, but do not identify a single
cause. In the new 24-example probe, post-hoc threshold selection barely changed the original
sampler's F1 (**0.467675 -> 0.468031**); a final-threshold change alone is not a demonstrated fix.

Another untested concern: [size_homogeneous_chunks](../src/vrp_diffusion_quantum/data/dataset.py)
traverses sizes in fixed ascending order, shuffling only within a size. Mixed training therefore
finishes each epoch with N100. The denoiser uses BatchNorm running statistics and unnormalized
sums of neighbor messages. Size-order effects and training-versus-inference BatchNorm behavior
deserve controlled checks. Their impact is not established by this review.

**4. Confirmed: Tasks 2 and 6 subtract different definitions of route gap. High priority.**

[Policy evaluation](../src/vrp_diffusion_quantum/train/train_policy.py) reports:

```text
100 * (sum(predicted_cost) - sum(reference_cost)) / sum(reference_cost).
```

[Diffusion routing evaluation](../src/vrp_diffusion_quantum/eval/routing.py) reports:

```text
mean(100 * (predicted_cost_i - reference_cost_i) / reference_cost_i).
```

The first weights individual relative gaps by reference cost; the second weights instances
equally. Consequently, the reported policy-versus-diffusion deltas are not strict like-for-like
comparisons. This particularly weakens Task 6's **0.10pp** seed-4333 win and **0.77pp** mean win.
It does not prove the ranking reverses; it means the stated gate has not been checked under one
consistent metric.

There are additional differences: the Task 2 policy command defaults to batch size 16 and uses a
50-step prior from its checkpoint config; the diffusion comparator uses batch size 1 and 700
steps. The candidate baseline manifest freezes policy batch size 1, and earlier profiling already
showed that batching can change greedy routes. K=1 matches the number of route starts, not the
whole inference budget or numerical protocol.

These runs remain useful comparisons of particular pipelines. To establish the claimed decoder
gain, use one gap definition, matched prior generation and batch settings, per-instance outputs,
and a paired interval. An untouched test confirmation remains outstanding; the 45-instance N100
validation panel was also used for checkpoint selection and tuning.

**5. Confirmed: Task 5 did not implement the requested source-count learning curve.**

[build_task5_curve.py](../outputs/_scratch/build_task5_curve.py) first materializes alternative
route labels, then [select_example_subset](../src/vrp_diffusion_quantum/data/subsets.py) samples
individual JSON files. During training,
[select_stochastic_references](../src/vrp_diffusion_quantum/data/training_labels.py) groups those
files by source and selects one reference per source per epoch.

Both the source IDs and training CSVs confirm these effective dataset sizes:

| Size | Nominal 500 | Nominal 1,000 | Nominal 2,000 |
|---|---:|---:|---:|
| N20 distinct sources | 500 | 1,000 | 2,000 |
| N50 distinct sources | 480 | 934 | 1,754 |
| N100 distinct sources | 456 | 835 | 1,375 |

The combined N100 pool contains 5,097 label files, not 5,097 independent CVRP instances.
Sampling files also favors sources with more alternative labels and changes the available label
set for existing sources as the curve grows. This changes more than the intended data-size axis.
The correct construction is to select nested **source IDs**, then retain all eligible references
for each selected source. I found zero source-ID overlap between these nine training subsets and
their corresponding frozen validation panels.

All nine runs completed 15 epochs, so the one-hour runtime cap did not explain their different
results. However, they use one training seed and choose checkpoints on only five route-decoded
validation examples per size. Non-monotonic results across different datasets do not estimate
run-to-run variance. “High single-run variance” is plausible, not a measured diagnosis.

The seed-0/seed-1 N100 check genuinely has identical reported F1 and mean gap. The explanation
that `--seed` has little effect because the panel is small is unsupported: it seeds actual reverse
sampling and changes instance ordering too. The defensible conclusion is that this checkpoint's
aggregate scores were stable for those two runs. It does not prove all matrices/routes are
bitwise identical or eliminate other sources of sampling variability.

There is also a direct reporting error: the report says **N50 has no candidate improvement**,
but its own table and JSON give **26.07% for N50/500 versus 29.39% for the old champion**.
That candidate merits the same qualification and further validation as N20/1,000 and N100/1,000.

**6. Confirmed: the F1 investigation's supposedly controlled comparisons were not all controlled.**

The filtered and unfiltered experiments were reported as evaluated on the same large panel.
Their saved `eval_config.yaml` files instead point to separate validation splits:

| Run | Validation directory | Reported F1 |
|---|---|---:|
| Filtered | `paper_cmd_full_50k_splits/val` | 0.470731 |
| Unfiltered | `paper_cmd_full_50k_unfiltered_splits/val` | 0.473142 |

Reconstructing the actual selection procedure gives **47 shared instances out of 1,500**.
The filtered panel also has **1,352 instances present in the unfiltered model's training set**.
This does not establish leakage in either originally reported run, each of which uses its own
validation split. It means simply reusing the filtered panel for both models would now leak
training data. A common panel must be disjoint from both training sets.

The “tight 95% F1 CI” claim is unsupported by the archived output: those JSON files contain
bootstrap intervals for **route-cost gap**, not F1. Similar F1 point estimates on different
panels do not establish statistical indistinguishability or rule out filtering effects.

Likewise, rescoring only the full model on a larger panel did not, by itself, establish a
noise-free pilot-versus-full difference. My new comparison puts both checkpoints on the same
24 full-validation instances, disjoint from both training sets: pilot **0.501772**, full
**0.467675**. This supports a real quality difference as a working hypothesis, but remains a
small diagnostic comparison. Full is actually better at N20 on this panel, so “worse at every
size” is not a stable general conclusion.

The other elimination claims need narrowing:

- A sweep of inference-step counts tests step count under the implemented sampler. It cannot
  rule out a sampler defect, as finding 1 demonstrates.
- The N20-only run argues against mixed-size interference being the sole explanation. It does
  not establish that every architecture/training issue is eliminated.
- A configuration rejected before training is not a negative experimental result. Four tests
  and one blocked attempt do not eliminate five explanations.
- Falling training loss plus declining generated F1 is compatible with overfitting, but also
  objective/sampling mismatch and normalization problems. Test-suite success does not select
  between these explanations. Epoch 1 was also the first scheduled generated-F1 measurement,
  not evidence of an observed earlier rise followed by a peak. The pilot report's confident
  data-shortage diagnosis was premature.

**7. Confirmed: the augmentation rejection was misinterpreted as a research dead end.**

Paper-mode training already uses `online_augmentation: true` and
`augmentation_recipe: paper_cmd_labeled`. The rejected `augmentation: true` flag requests the
different offline x9 path. In the trainer it disables online augmentation; with the paper recipe
it is also explicitly unsupported. The attempted config therefore would not add x9 augmentation
“on top of” the existing recipe as its comment claims.

[alignment.py](../src/vrp_diffusion_quantum/utils/alignment.py) locks a specific list of settings,
not every training hyperparameter. For example, diffusion BCE weighting and early-stop patience
are not in that required-value list. Several locked choices are reconstruction assumptions,
rather than confirmed author instructions. The appropriate response is to distinguish baseline
settings from explicitly labeled diagnostic variants and version any revised assumptions.
The rejection does not establish that augmentation is irrelevant or all useful tuning is impossible.

**8. Confirmed: the labeling workflow drifted from the declared reconstruction protocol.**

The comparison contract calls for a single HGS route partition without stability filtering.
Task 9 instead used two seeds, challengers, and matrix-stability acceptance, then passed
`label_policy: single_hgs_route_partition` as configuration metadata. The validation checks the
string, not whether the dataset actually follows that provenance.

The full audit solved 50,580 sources but trained on **33,449 accepted examples**:
**15,080 N20 / 12,200 N50 / 6,169 N100**. The later unfiltered run trained on 45,522 examples,
but its changed split prevents the originally claimed controlled filtering comparison.

The two-seed floor is a constraint of the audit tool, not proof that a single-solve labeling
path cannot be implemented. A feasible HGS route already defines a binary target even if other
near-optimal partitions exist; stability is an experimental filtering choice. Spending roughly
72 hours on the stricter audit generated useful reusable data, but did not produce the declared
50,000-training-example baseline. No additional labeling is needed merely to fix this mismatch:
the existing candidate solutions can be materialized under an explicit single-reference policy
with shared source-level train/validation/test assignments.

**9. Confirmed reporting and queue errors remain in the guide and task summaries.**

| Location | Problem | Defensible correction |
|---|---|---|
| Task 1, repeated in Task 3 and guide | Acceptance allegedly higher on OOD at every size | N20 98.4% < 98.6%; N50 86.8% < 94.4%; only N100 is higher, 56.8% > 47.8% |
| Task 3 | “No distribution” benefits from the merge | Saved tables pool R/C/RC by size; the required nine regime-by-size cells are not reported. The specific checkpoints lose on the aggregate panels |
| Task 3 | Better OOD route gaps explained by easier matrix prediction | N50/N100 champion OOD matrix F1 actually falls versus IID, despite lower route gaps. Matrix F1 and decoded cost must be analyzed separately |
| Task 6 | Base always selects epoch 0 | Its own table selects epoch 5; epoch-5 best-of-8 gap 27.71% is better than epoch-0 ~28.10% |
| Task 6 | Larger gradients after halving LR disprove step-size problems | Gradient size is not step size; this observation does not establish the proposed cause |
| Task 6 | “All 15 training runs” | The described base + three variants + two replications identify six runs, five newly trained variants/replications |
| Task 5/guide final status | Only quantum Tasks 7–8 remain | Task 4's matched comparison/CVRPLIB expansion has no located completion report; the available OR table remains two examples per size |
| Task 5 audit resume | 239 recomputed candidates described as simultaneously in flight | This is not explained by 11 active workers and conflicts with the earlier claim that all PyVRP candidates had completed. Cache invalidation/incomplete files need an actual explanation |
| Guide incident diagnosis | Exit 127 attributed definitively to CPU contention | Retry success and temporal coincidence do not establish the cause; preserve this as an unresolved launch incident |

Task 4's proposed command also points at the original strong-audit **training** pool, while the
proposed learned-model numbers come from validation. Before executing it, choose a common held-out
instance panel for all methods. Otherwise the proposed command itself cannot produce the stated
matched comparison.

The original guide correctly keeps quantum work behind the classical freeze and the relevant
paper-track prerequisites. The later “only Tasks 7–8 remain” summary must not override those gates.

**Why the documented F1 is so far below the paper's result**

The external target is real: Figure 8 reports **0.823 at 50 steps** and **0.875 at 1,000**.
However, its exact size mix, averaging and threshold protocol are not sufficiently specified
for an automatic equality gate. The text also leaves per-step sampling behavior unclear.
[Paper, page 13, Figure 8 and sensitivity analysis](https://arxiv.org/pdf/2603.07568v1#page=13)

The repository's F1 values need their evaluation context attached:

| Value | What it measures |
|---:|---|
| 0.505324 | Best mixed-size paper-mode pilot generated F1, selected on 24 validation instances, 50 steps |
| 0.470731 | Full-scale paper-mode generated F1 on 1,500 validation instances, 50 steps |
| 0.538906 | Later N20-only generated F1 on 1,341 instances; not a mixed-size result |
| 0.622194 | Existing robust N20 champion on 100 IID validation instances, 700 steps |
| 0.6695 | Older documented noisy-time denoising validation F1, not generation from a random prior |

Thus “0.505 is the highest overall” only makes sense with the mixed-size paper-mode generated
metric restriction. The old 0.6695 result is not directly comparable to generated F1: its input
contains a corrupted version of the target, and its threshold is selected adaptively.

The current generated metric is positive-class F1 pooled over off-diagonal pairs at threshold
0.5. With equal numbers of N20/N50/N100 graphs, N100 supplies about **77.8% of evaluated entries**;
the aggregate is not an equal-weight average of per-size F1 scores. The pooling can depress the
headline score relative to smaller-size results, but does not explain why even N20 remains near
0.54–0.56.

My best-supported explanation is a combination of **an incompletely matched inference/training
recipe and missing conditioning information**, compounded by weak experimental comparisons.
The wrong soft-posterior composition is confirmed; depot information loss is confirmed; weighted
BCE probability calibration and BatchNorm behavior remain concrete, testable suspects. Label
ambiguity and unspecified author details remain possible contributors, but the evidence does not
justify ranking HGS solve time as the dominant cause.

More data cannot restore an input that is absent or correct a mismatched transition formula.
Conversely, the small sampler improvement demonstrates that fixing one identified defect is
unlikely to be the entire answer. A causal allocation of the F1 deficit requires controlled
ablations; claiming a single proven root cause now would repeat the reports' central mistake.

**Recommended next work, in order**

1. Define a common evaluation contract: fixed source IDs disjoint from every compared training
   set, explicit matrix F1 aggregation and threshold, one route-gap definition, matched inference
   settings, per-instance outputs, and instance-bootstrap paired differences.
2. Correct and independently test the soft-posterior mixture, retaining a named legacy sampler
   for old checkpoints/results. Re-evaluate saved checkpoints before considering retraining.
3. Check depot-aware features, unweighted/probability-calibrated objectives, and BatchNorm/batch
   ordering with bounded, explicitly labeled diagnostics. Include a tiny-set fit sanity check and
   fixed-noise-level evaluations. Keep the paper ambiguity ledger honest about these assumptions.
4. Reuse the existing HGS candidates with shared source-level splits. Rebuild the Task 5 subsets
   by source, preserving all references. Only then use repeated seeds to study a scaling curve.
5. Reassess all three Task 5 candidates, including N50/500, and the N100 policy gate on a disjoint
   panel with consistent metrics. Complete Task 4 and the missing R/C/RC cell breakdown before
   treating the classical baseline as frozen.

The completed audits' source counts, solver-run counts and zero reported solver errors agree with
their saved `metrics.json` files. The Task 3 aggregate checkpoint losses and Task 5 route-gap table
also agree with the raw evaluations. Those artifacts remain useful. The main repair needed is to
the interpretation, comparison design, and parts of the model/inference specification—not a
blanket dismissal of the work already done.
