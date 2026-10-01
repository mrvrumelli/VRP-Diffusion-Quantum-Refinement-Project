# Classical baseline freeze proposal — 2026-09-30

Task F9 of the [F1-gap follow-up list](autonomous_f1gap_followup_tasks_2026-09-30.md). This is a
proposal, not a freeze: `baseline-v1.0` still has open criteria, listed at the end. Quantum and
quantum-inspired refinement stays gated on that freeze, as the
[decision manifest](evidence/corrective_20260930/candidate_baseline.json) requires.

All numbers are on the frozen 72-graph development panel (24 per size), which has been used for
model selection. They are development evidence, not test claims. Gaps are mean instance-relative
gaps to the panel reference routes.

## Proposed baseline

| Size | Method | Prior | Checkpoint (sha256 prefix) |
|---|---|---|---|
| 20 | Policy, 16 starts | Per-size champion diffusion (corrected 50-step sampler) | `policy_reinforce_s7799_n20_heldout_cuda_20260828T081136097007Z` (6b2a840c) |
| 50 | Policy, 16 starts | Per-size champion diffusion | `policy_reinforce_s7799_n50_heldout_cuda_20260828T082044076031Z` (fd254a13) |
| 100 | Policy trained with 16 starts, 16 starts | Per-size champion diffusion | `policy_reinforce_s7799_n100_starts16_cuda_20260924T165104398879Z` (03dc96e6) |

The N100 entry is the primary training seed (4331) of the 16-start policy, not the best of its
three seeds on this panel. Choosing the panel-best seed would select on development data.

## Policy versus diffusion decoding

Same graphs, same cached champion prior. Intervals are 95% bootstrap intervals over graphs.

| Size | Policy gap % | Diffusion-only gap % | Paired difference (pp) |
|---|---:|---:|---:|
| 20 | 9.19 [7.75, 10.63] | 21.33 | −12.1 [−17.6, −6.6] |
| 50 | 21.51 [19.66, 23.40] | 29.61 | −8.1 [−12.9, −3.4] |
| 100, original policy | 26.95 [25.40, 28.57] | 28.51 | −1.6 [−5.1, 2.0] |
| 100, 16-start policy, seed 4331 | 24.73 | 28.51 | −3.8 [−7.2, −0.5] |
| 100, 16-start policy, seed 4332 | 23.20 | 28.51 | −5.3 [−8.3, −2.2] |
| 100, 16-start policy, seed 4333 | 25.08 | 28.51 | −3.4 [−6.6, −0.3] |

The policy is the better method at every size. At N100 only the 16-start policies separate from
diffusion decoding.

## Classical references on the same panel

| Solver and budget | N20 gap % | N50 gap % | N100 gap % |
|---|---:|---:|---:|
| PyVRP, 1 s | 0.00 | 0.09 | 1.11 |
| PyVRP, 3 s | 0.00 | 0.04 | 0.41 |
| OR-Tools, 1 s | 0.96 | 5.07 | 9.83 |
| OR-Tools, 3 s | 0.65 | 4.45 | 8.75 |

The learned baseline is far from PyVRP at every size. That leaves substantial headroom for local
refinement, which is what the quantum stage targets.

## Why the prior is not changed

The F1-gap probes produced depot-aware joint priors that improve on the original `paper_cmd`
checkpoint, but none beats the per-size champions. Per size, at 50 steps with graph-ID seeding:

| Prior | N20 F1 / recall / gap % | N50 F1 / recall / gap % | N100 F1 / recall / gap % |
|---|---:|---:|---:|
| Original `paper_cmd`, no depot | 0.542 / 0.53 / 45.7 | 0.529 / 0.60 / 35.3 | 0.442 / 0.58 / 48.1 |
| Per-size champions (current prior) | 0.623 / 0.69 / 21.3 | 0.584 / 0.69 / 29.6 | 0.518 / 0.76 / 28.5 |
| Per-size Task 5 candidates | 0.622 / — / 17.6 | 0.576 / — / 25.1 | 0.545 / — / 26.7 |
| Depot-aware, distance edges (F2) | 0.547 / 0.49 / 42.4 | 0.566 / 0.57 / 37.7 | 0.507 / 0.62 / 39.6 |
| Depot-aware, trainable GAT | 0.598 / 0.60 / 24.9 | 0.575 / 0.59 / 38.5 | 0.504 / 0.63 / 45.1 |
| F5, corrected paper reconstruction at full scale | 0.456 / 0.34 / 79.9 | 0.555 / 0.53 / 50.6 | 0.530 / 0.61 / 42.8 |

Two points follow. First, per size the champions match or beat the depot-aware priors on F1 at N20
and N50 as well as on routes; pooled F1 hid this because N100 pairs dominate it. Second, the route
gap tracks recall more closely than F1. Missed same-route pairs fragment routes into extra depot
round trips, while extra pairs are partly repaired by the decoder's capacity splitting. F5's N20
prior misses two thirds of same-route pairs and decodes to an 80% gap.

Adding the depot is not what worsened F5's routes: on the same recipe and data, the depot-aware
probes improve pooled route gap over the original checkpoint (36–40% versus 43%). The Task 5
candidates decode better than the champions but failed their declared promotion rule, so the
champions remain the prior. Changing the prior would also require retraining the policies.

**Depot-aware champion recipe (task F10, 2026-10-01).** Retraining the champion recipe with
depot-relative input, matched against a control rerun that reproduces the champions exactly, lowered
the route gap in all six size–seed pairs but met the declared two-seed rule at no size. Seed-averaged
(post hoc) changes are −5.6 pp at N20, −4.1 pp at N50 and −2.0 pp at N100 (the last not separable
from zero). The proposal therefore keeps the existing champions, and lists a pre-declared
three-seed confirmation at N20 and N50 as the step that could promote a depot-aware prior.

**Confirmation (task F11, 2026-10-01).** On three new seeds the depot-aware recipe met the declared
rule at N50 (paired change −2.3 pp [−4.0, −0.5], better in 3 of 3 seeds) and failed it at N20
(−0.8 pp [−4.8, +2.8]). The proposed N50 prior is therefore the primary-seed depot-aware model
(`champ_n50_dep_s4331_20261001T082938994407Z`, sha256 prefix 5dc6a986), pending a retrain of the
N50 policy against it; N20 and N100 keep the existing champions.

## Open `baseline-v1.0` criteria

- **Commit and hashes.** The code used here is uncommitted in the working tree. A freeze needs a
  commit and a manifest pairing that commit with the checkpoint hashes above.
- **Independent evaluation.** The panel informed selection. The reserved test manifest
  (`outputs/corrective_20260930/reserved_test.json`) is still unscored and should be scored once,
  after the freeze, with stronger references as the decision manifest notes.
- **End-to-end timing.** Timings above exclude prior generation except in the earlier 8-graph
  end-to-end runs (about 1.1–1.5 s per graph). The freeze should report end-to-end time for all
  24 graphs per size.
- **Spatial out-of-distribution cells.** The R/C/RC cells for the policy are still outstanding
  (corrective task R11).
- **Seeds and configuration.** Each frozen artifact's config is in its run directory; the manifest
  should copy them with hashes.
