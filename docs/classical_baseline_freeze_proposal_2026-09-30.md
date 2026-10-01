# Classical baseline freeze proposal — 2026-09-30

Task F9 of the [F1-gap follow-up list](autonomous_f1gap_followup_tasks_2026-09-30.md). It started
as a proposal; the [freeze record](#freeze-record-2026-10-01) below closes it. Quantum and
quantum-inspired refinement was gated on that freeze, as the
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

The N100 entry is the primary run (training seed 42) of the 16-start policy, not the best of its
three seeds on this panel. Choosing the panel-best seed would select on development data.

## Policy versus diffusion decoding

Same graphs, same cached champion prior. Intervals are 95% bootstrap intervals over graphs.

| Size | Policy gap % | Diffusion-only gap % | Paired difference (pp) |
|---|---:|---:|---:|
| 20 | 9.19 [7.75, 10.63] | 21.33 | −12.1 [−17.6, −6.6] |
| 50 | 21.51 [19.66, 23.40] | 29.61 | −8.1 [−12.9, −3.4] |
| 100, original policy | 26.95 [25.40, 28.57] | 28.51 | −1.6 [−5.1, 2.0] |
| 100, 16-start policy, seed 42 (primary) | 24.73 | 28.51 | −3.8 [−7.2, −0.5] |
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

**Policy retrain (task F12, 2026-10-01).** Six N50 policies were trained with the corrected
sampler, three against the old champion prior and three against the depot-aware prior. The new
prior's three-seed paired change was −0.39 pp [−1.48, +0.76], which fails the declared rule. The
depot-aware prior helps diffusion-only decoding but not the policy pipeline, so the frozen N50 keeps
the existing policy and its original champion prior. The F11 prior stays the best diffusion-only
N50 prior.

## Freeze record (2026-10-01)

The baseline in the table at the top is frozen as `baseline-v1.0`. The
[freeze manifest](evidence/baseline_freeze_20261001/freeze_manifest.json) pairs commit `460b2efb5`
with the sha256 of every checkpoint and config, copies the six training configs, and records the
evaluation protocol and every result file below. `src/` and `configs/` match that commit. The
evaluation entry point, `scripts/solve_with_baseline.py`, was committed in `aca6c0aec`,
byte-identical to the hash in the manifest, which completes the freeze.

Protocol: the prior is regenerated per graph (50-step `posterior_mixture_v2`, graph-ID seed), the
policy decodes greedily from 16 starts, and the best start is kept. Timing is CUDA-synchronised
prior plus policy time per graph on an RTX 3060 Ti, excluding model loading. Intervals are 95%
bootstrap intervals over graphs. Every solution on every set below is feasible.

| Set | Graphs | N20 gap % | N50 gap % | N100 gap % |
|---|---:|---:|---:|---:|
| Development panel (used for selection) | 72 | 9.19 [7.78, 10.71] | 21.51 [19.69, 23.36] | 24.73 [23.42, 26.05] |
| Validation | 45 | 8.31 [6.76, 9.97] | 21.08 [19.38, 22.87] | 27.26 [25.41, 28.83] |
| R/C/RC spatial OOD cells | 72 | 8.29 [6.31, 11.02] | 19.20 [17.41, 20.93] | 22.09 [19.79, 24.35] |
| **Reserved test, scored once** | 96 | **8.56 [7.32, 9.86]** | **21.88 [20.63, 23.13]** | **27.20 [25.99, 28.42]** |

The reserved test is scored against strengthened references: the best of the instance label and
two seeded PyVRP runs (10, 20 and 40 s per size). The other sets use the instance labels. The
reserved-test result is the baseline's independent test claim, and that set must not be reused for
selection.

End-to-end time per graph on the development panel, measured on an idle machine:

| Size | Prior s | Policy s | Total s |
|---|---:|---:|---:|
| 20 | 0.70 | 0.08 | 0.78 |
| 50 | 0.69 | 0.17 | 0.86 |
| 100 | 0.75 | 0.31 | 1.06 |

The 50-step prior dominates the runtime at every size.

Spatial OOD cells, policy gap % per regime (8 graphs per cell), with diffusion-only champion
decoding on the same graphs for comparison:

| Regime | N20 policy / diffusion | N50 policy / diffusion | N100 policy / diffusion |
|---|---:|---:|---:|
| Random (R) | 8.23 / 17.01 | 22.54 / 27.54 | 26.67 / 32.76 |
| Clustered (C) | 10.51 / 17.35 | 15.30 / 19.83 | 16.08 / 26.34 |
| Mixed (RC) | 6.14 / 18.67 | 19.75 / 30.19 | 23.52 / 30.33 |

The policy beats diffusion-only decoding in every cell. No regime is much worse than the
development panel.
Clustered graphs are the easiest for both methods.

## Former open criteria

- **Commit and hashes.** Done: the manifest pairs commit `460b2efb5` with every hash, and the
  evaluation script is in commit `aca6c0aec` with the recorded hash.
- **Independent evaluation.** Done: the reserved test was scored once, against strengthened
  references.
- **End-to-end timing.** Done for all 72 panel graphs.
- **Spatial out-of-distribution cells.** Done (corrective task R11).
- **Seeds and configuration.** Done: copies of the six configs and their hashes are in the
  manifest folder.
