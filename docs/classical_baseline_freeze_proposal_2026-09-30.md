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

The F1-gap probes produced depot-aware joint priors with higher matrix F1 than the original
`paper_cmd` checkpoint, but their decoded routes are worse than the per-size champions:

| Prior, diffusion decoding at 50 steps | N20 gap % | N50 gap % | N100 gap % |
|---|---:|---:|---:|
| Per-size champions (current prior) | 21.33 | 29.61 | 28.51 |
| Per-size Task 5 candidates | 17.57 | 25.07 | 26.72 |
| Depot-aware joint, distance edges (F2) | 42.40 | 37.65 | 39.63 |
| Depot-aware joint, trainable GAT | 24.88 | 38.46 | 45.12 |

F1 gains need not improve decoded routes, as the corrective work already found. The Task 5
candidates decode better than the champions but failed their declared promotion rule, so the
champions remain the prior. Changing the prior would also require retraining the policies.

The full-length corrected paper reconstruction (task F5, 50 epochs on 50,000 unfiltered labels)
confirms this. Its prior has the highest matrix F1 of any checkpoint here (0.531 at 50 steps), yet
it decodes to the worst routes: 79.9% / 50.6% / 42.8% mean gap at N20 / N50 / N100. It is not a
candidate prior for the classical baseline.

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
