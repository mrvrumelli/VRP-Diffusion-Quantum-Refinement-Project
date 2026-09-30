# rcfull-merged vs. champion: IID and OOD verdict — 2026-09-28

**Correction, 2026-09-30 (R1):** The evidence supports aggregate checkpoint losses, not a verdict for each distribution: the nine R/C/RC-by-size cells remain outstanding. N50/N100 OOD matrix F1 falls despite lower route gaps. Only N100 acceptance improves over IID. See the [review](autonomous_work_review_2026-09-30.md) and [execution log](corrective_execution_2026-09-30.md). Historical run values below are retained.

Track A Task 3 ([`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)):
the real arbiter [[project_status_2026-08]] flagged as unresolved back in August — is the
`rcfull`-merged training experiment (R/C/RC spatial-stress data folded into the per-size
diffusion training pools) a genuine win on that harder distribution despite its known regression
on the original s7799 IID panel, or just a strictly worse model overall? Using Task 1's freshly
completed `rc_full_eval` panel (never seen in training, unlike the original `rc_full` pool that
went into the merged training set) alongside the existing IID panel settles it.

## Frozen inputs

- IID panel: `data/processed/s7799_val100_policy_v1_n{20,50,100}` (100/92/45 examples).
- OOD panel: `outputs/label_audit/rc_full_eval/accepted_matrix_examples`, filtered by size
  (2,952/2,604/1,705 examples) — the panel from
  [`task1_rc_full_eval_audit_2026-09-25.md`](task1_rc_full_eval_audit_2026-09-25.md).
- Champions: `diffusion_denoiser_s7799_stochastic_persize_n{20,50,100}_cuda` (frozen per-size
  stochastic-reference recipe).
- rcfull-merged: `diffusion_denoiser_s7799_rcfull_persize_n{20,50,100}_cuda` (same recipe, trained
  on IID + `rc_full`-audited labels merged together).

## Results

| Size | Champion IID | Champion OOD | rcfull-merged IID | rcfull-merged OOD |
|---|---:|---:|---:|---:|
| N20 | 21.11% | **17.89%** | 36.38% | 35.81% |
| N50 | 29.39% | **25.77%** | 37.35% | 34.32% |
| N100 | 30.57% | **29.56%** | 32.99% | **41.64%** |

All 12 runs 100% capacity-feasible. Raw artifacts: `outputs/eval/task3_rcfull_ood_20260925/{champion,rcfull}_{iid,ood}_n{20,50,100}/`.

## Verdict: rcfull-merged is strictly worse, not a real tradeoff

At every size, on both panels, the rcfull-merged checkpoint scores worse than the champion.
There is no size or distribution where merging the R/C/RC data helped — the hypothesis that it
might be trading IID quality for genuine OOD robustness is **not supported**. If anything, N100 —
the size where the merge added the most R/C/RC data relative to the original IID pool (7,759
`rc_full` labels vs. 1,268 IID labels, per
[`rc_full_artifact_reconstruction_2026-09-24.md`](rc_full_artifact_reconstruction_2026-09-24.md))
— is the *worst* result of the whole table: rcfull-merged's OOD gap (41.64%) is dramatically worse
than its own IID gap (32.99%), the only case in the entire comparison where a model scores worse
on OOD than on its own IID panel.

## A second, unexplained finding: champions generalize *better* to OOD than IID

At every size, the champion's OOD gap is lower than its IID gap (N20: 17.89% vs. 21.11%; N50:
25.77% vs. 29.39%; N100: 29.56% vs. 30.57%) — consistent and not small. This was not something
this task set out to test and is not yet explained. Plausible candidates, none confirmed:

- The R/C/RC spatial-stress instances may have structurally less ambiguous optimal route
  partitions than uniform-random IID instances at the same size (the labeling audits already show
  higher matrix-acceptance rates on R/C/RC data at every size — see
  [`task1_rc_full_eval_audit_2026-09-25.md`](task1_rc_full_eval_audit_2026-09-25.md) — so the
  *reference labels themselves* may simply be easier targets, not that the model handles R/C/RC
  inputs better).
- Panel-size effects (the OOD panels are much larger — 2,952/2,604/1,705 vs. 100/92/45 — so the
  IID numbers carry more sampling noise; the direction is consistent enough across sizes that this
  alone seems unlikely to fully explain a ~3-4pp gap, but hasn't been ruled out with confidence
  intervals compared directly).

Flagged for whoever picks up R/C/RC-related work next — worth a dedicated look, not resolved here.

## Consequence

- **Do not adopt the rcfull-merged checkpoints.** The frozen per-size stochastic-reference
  champions remain the correct active recipe, confirmed now against genuine OOD data, not just
  the original IID panel.
- Updates the open item in
  [`3060ti_training_todo.md`](archive/plans/3060ti_training_todo.md)'s 2026-09-24 status-correction
  note and [[project_status_2026-08]] from "unresolved" to a definitive no.
- The `rc_full`/R/C/RC labeled data itself is not wasted — it's real, audited, hashed evidence; it
  just doesn't belong merged into the per-size training pools as currently formulated. Whether a
  *separate* R/C/RC-specialized model, or a different merging strategy (e.g. down-weighting rather
  than pooling equally), would fare differently is a distinct, unstarted question.
