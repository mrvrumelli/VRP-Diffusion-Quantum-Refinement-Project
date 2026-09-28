# paper_cmd diffusion training — full-scale result — 2026-09-28

Track B Task 10, rerun at full scale after the pilot's clean negative result
([`task10_paper_cmd_diffusion_pilot_2026-09-25.md`](task10_paper_cmd_diffusion_pilot_2026-09-25.md))
was read as "not enough data" and used to justify the full ~59-hour, 50,580-instance labeling
campaign ([`task9_paper_cmd_full50k_audit_2026-09-28.md`](task9_paper_cmd_full50k_audit_2026-09-28.md)).
**This run's result contradicts that reading.**

## Frozen inputs

- Data: `data/processed/paper_cmd_full_50k_splits/{train,val}` — 33,449 train (11.6x the pilot),
  2,973 val, split from the full-scale audit's 37,165 accepted examples.
- GAT: `configs/train/gat_pretrain_paper_cmd_full.yaml`, early-stopped epoch 27, val F1 0.563 /
  AUC 0.9015 (both slightly better than the pilot's GAT: 0.561 / 0.899).
- Diffusion: `configs/train/diffusion_denoiser_paper_cmd_full.yaml` — identical paper contract to
  the pilot run (T=1000, BatchNorm, `noisy_matrix` edge input, frozen GAT, `sample_f1`
  checkpoint selection), only the data changed.

## Result: gate still not met, and worse than the pilot at every size

Training early-stopped at **epoch 22/50** (same `early_stop_patience: 10` as the pilot). Best
checkpoint is epoch 1:

| Metric | Pilot (2,886 train) | Full scale (33,449 train) | Target |
|---|---:|---:|---:|
| **Overall sample F1** | 0.505 | **0.474** | ≈0.823 |
| N20 sample F1 | 0.564 | 0.547 | — |
| N50 sample F1 | 0.574 | 0.500 | — |
| N100 sample F1 | 0.484 | 0.459 | — |
| Route feasibility | 100% | 100% | — |
| Route mean cost gap | 44.71% | 40.88% | — |

F1 across evaluated epochs (every 2 epochs, `sample_num_examples: 24`, 8 per size): 0.474 (ep1) →
0.459 → 0.456 → 0.451 → 0.455 → 0.453 → 0.456 → 0.460 → 0.450 → 0.466 → 0.455 (ep21) — flat,
noisy, no upward trend, never approaching either the pilot's own level or the target.

## This overturns the pilot's "not enough data" reading

The pilot's report reasoned from an overfitting signature (train loss kept falling while F1 peaked
early and decayed) to conclude the 2,886-example pool was too small. **Eleven times more training
data, and roughly the same multiple more gradient steps by epoch 1, produced a worse result at
every size, not a better one.** If data scale were the binding constraint, this run should have
beaten the pilot by a wide margin. It did not beat it at all.

**One real caveat, stated plainly rather than used to explain the finding away**: the periodic
`sample_f1` metric evaluates only 8 examples per size (24 total) per checkpoint — a small,
stochastic (50-step `skipped_posterior`) reverse-chain sample, so any single epoch's number
carries real sampling noise. But the direction is consistent across all three sizes and across
both training loss curves, which a pure noise explanation doesn't comfortably account for. This
should be checked against a larger, fixed evaluation panel before being fully trusted, not
dismissed as noise by default.

## What this actually points at

Since more data did not help, the bottleneck is more likely one of:

1. **An unresolved paper ambiguity** — [`cmd_paper_comparison_contract.md`](cmd_paper_comparison_contract.md)
   already lists exact HGS budget/seeds, augmentation enumeration, and the skipped-transition
   schedule as open author questions. This project's 2-seed HGS labeling (the tooling floor, not
   the paper's exact protocol) and 10/20/40s time budgets (a documented assumption, not a
   confirmed one) could plausibly produce systematically different — not just noisier — training
   targets than whatever the paper's authors actually used.
2. **A genuine hyperparameter or training-recipe gap** unrelated to raw example count — e.g. the
   `paper_cmd_labeled` augmentation recipe's actual effect not yet isolated, or the GAT/diffusion
   learning rate and epoch budget tuned for the paper's stated 50,000-instance/50-epoch protocol
   not transferring as cleanly as assumed to this project's specific label-generation choices.
3. Something in the **evaluation itself** — worth a sanity check with a larger, fixed sample panel
   (say 100+ per size instead of 8) before concluding anything further about the model.

## Recommendation

**Do not immediately commit to more data collection** — this result is direct evidence that data
scale alone is not the lever that closes this gap, reversing the pilot's own conclusion. The next
useful step is diagnostic, not another scale-up: re-run the sample-F1 evaluation on a larger fixed
panel to rule out (1) above, and/or resolve the open HGS-budget/seed-count ambiguity explicitly
(even provisionally) before spending further GPU time on this exact recipe. This is a genuine
open problem for the research owner's judgment, not something to route around autonomously.
