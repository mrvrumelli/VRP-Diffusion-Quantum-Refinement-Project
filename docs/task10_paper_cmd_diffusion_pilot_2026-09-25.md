# paper_cmd diffusion training — pilot-scale result — 2026-09-25

Track B Task 10 ([`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)):
train the paper-config GAT + diffusion denoiser on Task 9's pilot labels and check the paper's
qualitative gate — approximately F1 0.823 at 50 inference steps — before deciding whether the full
50,000-instance labeling campaign is needed.

## Frozen inputs

- Data: `data/processed/paper_cmd_pilot_5k_splits/{train,val}` — 2,886 train (1,310/1,066/510),
  540 val (245/200/95). Source: Task 9's pilot audit, see
  [`task9_paper_cmd_pilot_audit_2026-09-24.md`](task9_paper_cmd_pilot_audit_2026-09-24.md).
- GAT: `configs/train/gat_pretrain_paper_cmd_pilot.yaml`, 50 epochs, paper contract (5-layer,
  8-head). Final val F1 0.561, AUC 0.899.
  Checkpoint: `outputs/paper_cmd/gat_pretrain_paper_cmd_pilot_20260925T001632213104Z/checkpoints/gat_encoder_best.pt`.
- Diffusion: `configs/train/diffusion_denoiser_paper_cmd_pilot.yaml` — full paper contract
  (T=1000, BatchNorm, `noisy_matrix` edge input, batch 32, frozen GAT), 50-epoch budget,
  checkpoint-selected on `sample_f1` (full reverse-chain, 50-step `skipped_posterior` sample
  eval every 2 epochs — this is the actual gate mechanism, not a separate check).

## Result

Training **early-stopped at epoch 22/50** (`early_stop_patience: 10`, no `sample_f1` improvement
after epoch 1). Best checkpoint is epoch 1:

| Metric | Value |
|---|---:|
| **Overall sample F1 (target: ≈0.823)** | **0.505** |
| N20 sample F1 | 0.564 |
| N50 sample F1 | 0.574 |
| N100 sample F1 | 0.484 |
| Route feasibility | 100% |
| Route mean cost gap | 44.71% |
| Noisy-time val F1 (epoch 22, final — not the gate metric) | 0.637 |

Sample F1 across evaluated epochs (every 2 epochs): 0.505 (ep1) → 0.453 → 0.448 → 0.444 → 0.470 →
0.466 → 0.467 → 0.466 → 0.451 → 0.459 → 0.463 (ep21) — peaks immediately at epoch 1 and never
recovers, while train_loss keeps falling (0.974 → 0.573) — the signature of overfitting a
severely undersized training pool, not an implementation problem.

## Gate decision: NOT met

**0.505 vs. the ≈0.823 target is not close** — this isn't a borderline miss, it's roughly 60% of
the target. Per the guide's own stated decision rule (Task 9, step 2): this is the signal that
the pilot's 2,886-example pool is genuinely too small, not evidence against the paper_cmd
implementation itself. Two independent pieces of engineering evidence support that reading rather
than a code-correctness concern:

1. [`phase3_faithful_diffusion_status_2026-09-24.md`](phase3_faithful_diffusion_status_2026-09-24.md)
   already verified the skipped-posterior math, BatchNorm/edge-input wiring, and checkpoint
   round-trips with 81 focused + 507 full-suite tests passing.
2. The overfitting signature (train loss keeps improving, sample F1 peaks at epoch 1 and
   degrades) is exactly what undersized-data overfitting looks like, not what a broken sampler or
   wrong hyperparameter looks like (which would typically show a flat or noisy F1 from the start,
   not an early peak followed by decay).

N100 scoring lowest (0.484) and having the smallest training slice (510 examples, vs. 1,310/1,066
for N20/N50) is also consistent with a pure data-scale explanation, not a size-specific bug.

## Consequence

This is real evidence the full 50,000-instance labeling campaign
(autonomous_long_compute_guide_2026-09-24.md Task 9, step 3, ~29-30h CPU) is actually needed to
have a chance at the paper's reported F1 — the pilot did its job of testing this cheaply before
committing blind. **Not launched automatically** given the scale of that commitment (a fresh
~30-hour CPU job, competing with Task 1's still-running audit for the same CPU slot) — flagging
this as the next real decision point for the research owner rather than launching it unilaterally.

## Caveat

50 epochs was the paper's stated diffusion epoch count for its full 50,000-instance protocol; it
is not necessarily the right epoch count for a 2,886-example pilot, where overfitting this fast
suggests fewer epochs (or stronger regularization/augmentation) might have been more informative
here. This was not re-tuned for the pilot specifically, since the pilot's purpose was to test
"does the current data scale get close to the gate," not to find the best pilot-scale recipe.
