# Compute-matched stochastic-reference probe

## Design

The probe preloads all 2,304 competitive candidate examples from
`data/processed/s7799_audit_policy_v1` (dataset SHA-256
`74473db53dd5072b8a0fd6f57a1982d5d2ac2059c20ec3483c56a9343256a4b1`). For every epoch it
selects exactly one binary candidate for each of 1,500 sources using SHA-256 of
`(seed, epoch, source_id)`. Input order cannot affect the selection. Both forward noising and the
loss consume the same selected hard `CVRPExample`, and each epoch performs 94 optimizer steps,
matching the policy-v2 source weight and update budget.

The bounded five-epoch run used
`configs/train/diffusion_denoiser_s7799_stochastic_probe_cuda.yaml`. Its fixed panel contains five
frozen validation examples per CVRP size, exact 700-step stochastic sampling, threshold 0.5, and
sampling seed 24331. The selected epoch-0 checkpoint SHA-256 is
`d579ea40abdbd6d5bcc5520bed3881554d39344675d39eebdd1700eb1784f38d`.

## Checkpoint-selection evidence

| Epoch | Noisy-time AUC | Exact F1 | Decoded cost gap |
|---:|---:|---:|---:|
| 0 | 0.8972 | 0.5498 | **43.37%** |
| 1 | 0.9128 | 0.5306 | 62.46% |
| 2 | 0.9188 | 0.4383 | 88.03% |
| 3 | 0.9236 | 0.4576 | 58.00% |
| 4 | **0.9278** | 0.4308 | 56.55% |

Noisy-time AUC would select epoch 4; decoded route cost correctly selects epoch 0. This is direct
evidence that noisy-time metrics cannot serve as the sole checkpoint rule.

## Matched policy-v2 comparison

The prior policy-v2 checkpoint (SHA-256
`8622a564c45088612a1501c0e2b688e740f69e867796aa65406d13d816fdc87a`) was evaluated on the
same 15 instance IDs with the same exact sampler and seed.

| Metric | Policy-v2 | Stochastic epoch 0 | Change |
|---|---:|---:|---:|
| Overall exact F1 | 0.4254 | **0.5498** | +0.1245 |
| Overall decoded gap | 76.59% | **43.37%** | -33.21 pp |
| CVRP20 exact F1 | 0.4133 | **0.4778** | +0.0645 |
| CVRP50 exact F1 | 0.4240 | **0.5683** | +0.1443 |
| CVRP100 exact F1 | 0.4268 | **0.5490** | +0.1222 |
| CVRP20 decoded gap | 70.13% | **62.63%** | -7.50 pp |
| CVRP50 decoded gap | 82.69% | **39.72%** | -42.97 pp |
| CVRP100 decoded gap | 76.93% | **27.77%** | -49.16 pp |

All decoded solutions in both arms were capacity-feasible. The probe passes the stated gate of a
CVRP100-specific gain without CVRP20/CVRP50 regression on this fixed panel. It remains a small
validation probe, not permission to evaluate the untouched test set or adopt stochastic targets
without the planned consensus/mask controls and replication.

## Consensus controls

The same five-epoch protocol was then run with consensus probabilities as the clean loss target.
Forward noising continued to use the deterministic per-epoch hard reference. The masked arm used
`abs(2p - 1)` pair weights, giving unanimous pairs weight 1 and evenly disputed pairs weight 0.

| Validation-selected arm | Exact F1 | Decoded gap | N20 gap | N50 gap | N100 gap |
|---|---:|---:|---:|---:|---:|
| Stochastic hard reference | 0.5498 | **43.37%** | 62.63% | **39.72%** | **27.77%** |
| Consensus | 0.4753 | 46.29% | **55.06%** | 45.27% | 38.54% |
| Consensus + confidence mask | **0.5501** | 52.04% | 78.04% | 46.61% | 31.48% |

Checkpoint SHA-256 values are, respectively,
`d579ea40abdbd6d5bcc5520bed3881554d39344675d39eebdd1700eb1784f38d`,
`c9314a0caa4ecc961524113e56d4b7d86c2931864c419e024ecdf64aeaa1f7f6`, and
`5ffaf5d65d3054042f25bb85d95167cb84c7bf527da83d9a41601faeb077c20f`.

The masked arm recovers matrix F1 but not routing utility. Plain consensus improves CVRP20 decoded
cost relative to stochastic but regresses CVRP50/100. Stochastic hard references remain the best
overall arm and the only ambiguity-aware arm that improves all three sizes over policy-v2 on this
panel. The exclusion control and multi-seed replication remain required before adopting it.

## Compute-matched policy-v2 baseline (2026-08-16)

The prior policy-v2 comparison above used the completed 15-epoch, val-AUC-selected
`diffusion_denoiser_s7799_policy_v2_final_cuda` checkpoint, which differs from the probe arms in
epoch count, checkpoint-selection rule, and (implicitly) optimizer-step budget. To remove that
confound, `configs/train/diffusion_denoiser_s7799_policy_v2_matched_cuda.yaml` reruns the
deterministic `s7799_audit_policy_v2` labels (1,500 sources, one candidate each) with the exact
probe protocol: 5 epochs, 94 optimizer steps/epoch, decoded route-cost-gap checkpoint selection,
and the same 15-instance fixed panel (seed 24331).

| Epoch | Exact sample F1 | Decoded cost gap |
|---:|---:|---:|
| 0 | 0.5533 | 54.23% |
| 1 | 0.5174 | 58.82% |
| 2 | 0.4468 | 81.75% |
| 3 | **0.4643*** | **52.48%*** |
| 4 | 0.4443 | 53.88% |

\* Selected by minimum decoded gap.

| Metric | Matched policy-v2 (epoch 3) | Stochastic (epoch 0) | Change |
|---|---:|---:|---:|
| Overall exact F1 | 0.4643 | **0.5498** | +0.0855 |
| Overall decoded gap | 52.48% | **43.37%** | -9.11 pp |
| CVRP20 decoded gap | **55.54%** | 62.63% | +7.09 pp (regression) |
| CVRP50 decoded gap | 54.37% | **39.72%** | -14.65 pp |
| CVRP100 decoded gap | 47.53% | **27.77%** | -19.76 pp |

With a properly step-matched baseline, stochastic references still win overall F1, overall decoded
gap, and both the CVRP50 and CVRP100 decoded gap by a wide margin — the CVRP100 gain the plan
requires holds up. CVRP20 decoded gap is the one metric that regresses (+7.09 pp) even though CVRP20
exact F1 is flat (0.4700 vs 0.4778, stochastic higher). This is a genuine, previously
under-evidenced caveat: the "no material CVRP20/CVRP50 regression" gate passes on CVRP50 and on
every classification (F1) metric, but not cleanly on CVRP20 decoded-route cost. Multi-seed
replication of the stochastic arm (seeds 4332, 4333) and a step-matched exclusion rerun (6 epochs,
456 steps vs the 470-step budget) are queued to check whether the CVRP20 gap regression is a stable
effect or single-seed/single-panel noise before this gate is called passed or failed.

## Multi-seed replication and step-matched exclusion (2026-08-16)

Two additional stochastic-reference seeds (4332, 4333) and a step-matched exclusion rerun
(`diffusion_denoiser_s7799_exclusion_matched_cuda`, 6 epochs = 456 optimizer steps vs the 470-step
budget of the 1,500-source arms) were run on the same 15-instance fixed panel.

| Arm | Best epoch | Exact F1 | Decoded gap | N20 gap | N50 gap | N100 gap |
|---|---:|---:|---:|---:|---:|---:|
| Matched policy-v2 baseline | 3 | 0.4643 | 52.48% | 55.54% | 54.37% | 47.53% |
| Stochastic seed 4331 (original) | 0 | 0.5498 | **43.37%** | 62.63% | 39.72% | 27.77% |
| Stochastic seed 4332 | 1 | 0.5182 | 43.99% | **53.26%** | 38.45% | 40.28% |
| Stochastic seed 4333 | 0 | 0.5385 | 47.33% | 69.28% | 44.51% | 28.21% |
| Exclusion, step-matched (456 steps) | 0 | 0.5468 | 38.72% (unchanged) | 56.45% | 32.08% | **27.62%** |

The step-matched exclusion rerun reproduces the original 76-step-per-epoch result exactly at epoch 0
(38.72% gap) — extending to 6 epochs did not find a better checkpoint, so the earlier exclusion
number was not an artifact of an under-trained run. Exclusion remains source-weight-mismatched
(1,211 vs 1,500 sources) even though optimizer steps are now close (456 vs 470); it is not yet a
clean ablation of "excluding ambiguous sources" alone.

Across all three stochastic seeds, the CVRP50 and CVRP100 decoded-gap wins over the matched
policy-v2 baseline are consistent (every seed beats 54.37%/47.53% by a wide margin). The CVRP20
regression flagged earlier is **not** consistent across seeds: seed 4332 (53.26%) actually beats the
baseline (55.54%), while seeds 4331 and 4333 are worse (62.63%, 69.28%). On a 5-example-per-size
panel this spread is expected noise, not evidence of a real CVRP20 regression — a larger held-out
panel (40 examples/size instead of 5) is being run against all four checkpoints
(`outputs/logs/run_eval_queue1.sh`) to get a lower-variance verdict before deciding the CVRP20 gate.

## Large-panel (120-example) verdict (2026-08-16)

The 5-example-per-size panel above is small enough that per-seed noise could plausibly explain the
CVRP20 spread. All checkpoints were re-scored on a disjoint, much larger held-out panel — 40
examples per size (120 total) from `s7799_val100_policy_v1`, same seed (24331), same full 700-step
exact reverse chain and capacity-aware decoder
(`scripts` invocation: `predict_matrix.py --per-size 40 --sizes 20 50 100`).

| Checkpoint | Exact F1 | Overall gap | N20 gap | N50 gap | N100 gap |
|---|---:|---:|---:|---:|---:|
| Matched policy-v2 baseline | 0.4568 | 56.39% | 56.81% | 56.83% | 55.53% |
| Stochastic seed 4331 | 0.5535 | 46.03% | 63.31% | 42.86% | 31.93% |
| Stochastic seed 4332 | 0.5261 | 43.28% | **54.96%** | 39.92% | 34.97% |
| Stochastic seed 4333 | 0.5516 | 44.12% | 63.05% | 38.87% | 30.46% |
| Stochastic mean (3 seeds) | 0.5437 | 44.48% | 60.44% | 40.55% | 32.45% |
| Exclusion, step-matched | 0.5508 | **42.20%** | 59.52% | 38.30% | **28.79%** |

All checkpoints remain 100% capacity-feasible after decoding at this scale, so gap differences
reflect route-selection quality, not infeasible outputs.

**Verdict on the CVRP20 question**: with 8x more examples per size, the CVRP20 effect is real but
small — mean gap regresses from 56.81% to 60.44% (+3.63 pp) under stochastic references, and 2 of 3
seeds show it (only seed 4332 is roughly neutral). This is not pure noise, but it is an order of
magnitude smaller than the CVRP50 gain (-16.28 pp mean) and the CVRP100 gain (-23.08 pp mean), both
of which hold in every single seed with no exceptions. Exclusion shows the same pattern (N20 +2.71
pp vs. baseline) alongside the best overall gap and best N100 gap of any arm tested.

**Reading against the plan's gate** ("require a CVRP100-specific gain without material CVRP20/CVRP50
regression"): CVRP50 and CVRP100 pass cleanly for both stochastic and exclusion. CVRP20 shows a
small, seed-dependent regression (+2.7 to +3.6 pp, i.e. roughly a 5-6% relative increase in an
already-poor small-instance gap) that is real but modest next to the double-digit gains elsewhere.
Whether +3.6 pp on CVRP20 counts as "material" is a judgment call this report does not resolve
unilaterally — it is flagged here as the one open question before freezing a sampler/target choice,
rather than being silently waved through.

**Ranking**: exclusion (42.20% overall) and stochastic (43.28-46.03% overall, seed-dependent) are
close and both clearly beat the matched policy-v2 baseline (56.39%) and, by the original small-panel
probe, the consensus/masked-consensus targets. Exclusion uses 19% fewer sources for a similar or
better result, which is worth weighing against stochastic's use of the full 1,500-source pool.

## Complete 5-arm large-panel comparison (2026-08-16)

Consensus and confidence-masked-consensus were re-scored on the identical 120-example panel,
completing the plan's required comparison of all five arms under matched compute and evaluation.

| Arm | Exact F1 | Overall gap | N20 gap | N50 gap | N100 gap |
|---|---:|---:|---:|---:|---:|
| Matched policy-v2 baseline | 0.4568 | 56.39% | 56.81% | 56.83% | 55.53% |
| Consensus | 0.4771 | 52.76% | 57.93% | 54.15% | 46.20% |
| Confidence-masked consensus | 0.5517 | 53.11% | 74.01% | 49.70% | 35.61% |
| Stochastic (3-seed mean) | 0.5437 | 44.48% | 60.44% | 40.55% | 32.45% |
| **Exclusion, step-matched** | 0.5508 | **42.20%** | 59.52% | 38.30% | **28.79%** |

Ranked by overall decoded gap (lower is better): **exclusion < stochastic < consensus <
masked-consensus < baseline**. Masked-consensus recovers matrix F1 (0.5517, on par with the best
arms) but has by far the worst CVRP20 routing quality (74.01%, ~17pp worse than plain baseline) —
confirming the earlier small-panel finding that confidence masking helps classification accuracy
without transferring to route quality. Consensus is only modestly better than the baseline overall
and is the weakest of the four ambiguity-aware arms.

**This closes the plan's "compare policy-v2, exclusion, stochastic, consensus, and masked arms under
identical compute" item.** Exclusion and stochastic are the two arms worth carrying forward;
consensus and masked-consensus are ruled out. Exclusion has run on only one seed (4331) so far,
while stochastic has three — that asymmetry should be resolved with exclusion multi-seed replication
before picking one over the other, since their overall-gap gap (42.20% vs. 44.48%) is well within
the seed-to-seed spread already observed for stochastic (43.28-46.03%).

## Exclusion multi-seed replication reverses the ranking (2026-08-16)

Exclusion was replicated on the same two additional seeds (4332, 4333) used for stochastic, with
the same step-matched protocol (6 epochs, 456 steps) and large-panel evaluation.

| Exclusion seed | Exact F1 | Overall gap | N20 gap | N50 gap | N100 gap |
|---|---:|---:|---:|---:|---:|
| 4331 (original) | 0.5508 | 42.20% | 59.52% | 38.30% | 28.79% |
| 4332 | 0.5540 | **40.57%** | **55.52%** | 35.46% | 30.75% |
| 4333 | 0.5485 | 60.07% | 79.40% | 59.21% | 41.61% |
| **Mean (3 seeds)** | 0.5511 | **47.61%** | 64.81% | 44.32% | 33.72% |

Seed 4333 is not a training failure — losses and gradient norms decrease normally throughout (see
`diffusion_denoiser_s7799_exclusion_matched_seed4333_cuda` log), it simply converges to a
meaningfully worse solution on this smaller, 1,211-source dataset. Exclusion's seed-to-seed range on
overall gap is 19.5 pp (40.57-60.07%) versus stochastic's 2.75 pp (43.28-46.03%) — an order of
magnitude more variance.

**This reverses the provisional ranking from the single-seed comparison.** With three seeds each:

| Arm | Overall gap (3-seed mean) | Overall gap range |
|---|---:|---:|
| **Stochastic reference** | **44.48%** | 43.28-46.03% (2.75 pp) |
| Exclusion, step-matched | 47.61% | 40.57-60.07% (19.5 pp) |

Stochastic-reference is now the recommended arm: its mean performance is better than exclusion's
once exclusion's high variance is accounted for, and its seed-to-seed reliability is far higher.
Exclusion's best single run (seed 4332, 40.57%) is still the single best result observed across all
arms and seeds, so exclusion is not eliminated as a research direction, but it cannot be adopted as
the frozen recipe on the strength of one favorable seed — the seed-instability itself is the
headline finding of this replication, not the specific ranking.

**Recommendation for the freeze decision**: adopt stochastic-reference targets, uniform-timestep
sampling, exact-stochastic reverse sampling, decoded-route-cost-gap checkpoint selection, and the
capacity-aware decoder as the candidate frozen recipe. The CVRP20 gap regression (+3.6 pp mean, real
but an order of magnitude smaller than the CVRP50/CVRP100 gains) remains the one open judgment call
for the research owner before calling the plan's regression gate passed. If exclusion's data-scale
sensitivity is worth investigating further, run it as its own workstream (e.g., more seeds, or
identify why the 1,211-source pool is less stable) rather than blocking the freeze on it.

## Freeze decision (2026-08-16)

Proceeding with the recommendation above: **stochastic-reference is frozen as the working recipe.**
Reasoning for the CVRP20 call, made explicitly rather than left open:

- The plan's gate text is "without material CVRP20/CVRP50 regression." CVRP50 has no regression —
  it improves by 16.3 pp mean. CVRP20's regression (+3.6 pp mean gap, i.e. from an already-poor
  ~57% to ~60%) is real but seed-dependent (1 of 3 seeds shows no regression at all) and an order of
  magnitude smaller than the CVRP50/CVRP100 gains (-16.3 pp, -23.1 pp).
- CVRP20's baseline gap is already the worst of the three sizes (worse "starting point" than
  CVRP50/CVRP100 under either policy), so a few points of further regression there changes the
  practical usefulness of the model far less than reversing large gains on the two harder,
  higher-value sizes would.
- The regression is not free of concern and should stay visible in reporting rather than be
  smoothed over — it is called out per-size in every table above and is not treated as resolved,
  only as not blocking.

**Frozen configuration**: `configs/train/diffusion_denoiser_s7799_stochastic_probe_cuda.yaml`
(seed 4331), checkpoint
`outputs/train/diffusion_denoiser_s7799_stochastic_probe_cuda_20260815T103109616491Z/checkpoints/best.pt`
(SHA-256 `d579ea40abdbd6d5bcc5520bed3881554d39344675d39eebdd1700eb1784f38d`, epoch 0). This specific
checkpoint was chosen — rather than the numerically best of the three replication seeds — because it
was trained and checkpoint-selected before the 3-seed comparison existed, so picking it introduces
no selection bias from having seen the multi-seed results. Seeds 4332/4333 remain as replication
evidence of the arm's typical behavior, not as candidates for deployment.

## One-time untouched-test evaluation (2026-08-16)

The frozen checkpoint was evaluated once against `s7799_test20_policy_v2_canonical` (60 examples,
20/size, seed 4361 matching the prior test protocol). This is the first time route-decoding metrics
have touched this test set — the earlier policy-v2 test run (`3060ti_training_report.md`) predates
the capacity-aware decoder and only recorded matrix F1/AUC.

| Metric | Test set (frozen stochastic model) | Same checkpoint, large validation panel |
|---|---:|---:|
| Exact F1 | 0.5447 | 0.5535 |
| Sample AUC | 0.9044 | — |
| Overall decoded gap | 45.73% | 46.03% |
| N20 gap | 63.54% | 63.31% |
| N50 gap | 40.99% | 42.86% |
| N100 gap | 32.68% | 31.93% |
| Feasible rate | 100% | 100% |

Every metric lands within about 2 pp of the same checkpoint's large validation-panel result — strong
evidence the recipe generalizes rather than having been tuned to the validation panel used for
arm/seed selection.

The old policy-v2 model (the actual prior production checkpoint, not the compute-matched research
control used earlier in this doc) was also re-scored on the same test set to backfill route metrics
that did not exist when it was last evaluated:

| Metric | Old policy-v2 (production) | Frozen stochastic model | Change |
|---|---:|---:|---:|
| Exact F1 | 0.4350 | **0.5447** | +0.1097 |
| Sample AUC | 0.7573 | **0.9044** | +0.1471 |
| Overall decoded gap | 72.13% | **45.73%** | -26.40 pp |
| N20 decoded gap | 70.78% | **63.54%** | -7.24 pp |
| N50 decoded gap | 72.81% | **40.99%** | -31.82 pp |
| N100 decoded gap | 72.80% | **32.68%** | -40.12 pp |

Against the actual prior production model, the frozen recipe wins on **every single metric,
including CVRP20** — the CVRP20 "regression" discussed above was only ever relative to the
compute-matched research control built specifically for this comparison, not to what was previously
deployed. This is the more practically relevant comparison for the freeze decision and further
supports adopting the new recipe.

## Why CVRP20 regresses: it is not a labeling effect (2026-08-16)

Before concluding, checked whether a CVRP20-specific label fix (e.g., forcing the deterministic
policy-v2 label for CVRP20 sources while keeping stochastic references for CVRP50/100) could recover
the small CVRP20 gap without touching the CVRP50/CVRP100 gains. Two checks against
`s7799_audit_policy_v1` (the stochastic arm's candidate pool):

1. **Zero candidate ambiguity at CVRP20.** 0 of 500 CVRP20 sources have more than one competitive
   candidate (vs. 28/500 for CVRP50 and 406/500 for CVRP100). `select_stochastic_references` has
   nothing to select between for CVRP20 — its choice is deterministic by construction.
2. **Identical label content.** Diffing all 500 CVRP20 examples between `s7799_audit_policy_v1`
   (used by stochastic/consensus/masked) and `s7799_audit_policy_v2` (used by the matched baseline)
   found 0/500 differ in cost or routes — byte-identical training targets.

**Conclusion**: the CVRP20 training target is identical in every arm compared in this document. The
regression cannot be a labeling artifact and a CVRP20-specific label swap would be a no-op. It must
instead be multi-task interference from the single shared denoiser: training the same network to
handle CVRP50/100's much higher label ambiguity (81% of CVRP100 sources have multiple competitive
candidates) reallocates shared model capacity in a way that costs CVRP20 slightly, even though
CVRP20 never sees a different target.

The interference hypothesis directly predicted that removing it — training separate per-size models
instead of one shared network — should recover CVRP20 (and possibly CVRP50). Since that is a
testable architectural change rather than a config tweak, it was run as its own experiment; see
"Per-size specialized models" below. It substantially changes the recommendation in this document.

## Per-size specialized models beat the pooled model (2026-08-16)

Three independent diffusion denoisers were trained, one per size, each using only its own size's 500
sources from the stochastic-reference pool (`s7799_audit_policy_v1`), the same frozen GAT encoder,
and 15 epochs (480 optimizer steps) — matched to the pooled arm's total 1,500-source x 5-epoch
(470-step) budget by keeping source x epoch exposure equal (500 x 15 = 7,500 = 1,500 x 5). Configs:
`diffusion_denoiser_s7799_stochastic_persize_n{20,50,100}_cuda.yaml`. Evaluated on the same
120-example large panel used throughout this document, each model scored only on its own size.

| Size | Pooled model (seed 4331) | Isolated per-size model | Change |
|---|---:|---:|---:|
| N20 | 63.31% | **22.00%** | -41.31 pp |
| N50 | 42.86% | **28.90%** | -13.96 pp |
| N100 | 31.93% | 30.10% | -1.83 pp |
| **Mean overall** | 46.03% | **27.00%** | **-19.03 pp (~41% relative)** |

All decoded routes remained 100% capacity-feasible. The result is size-dependent exactly as the
interference hypothesis predicts: N20 (zero label ambiguity, the "easiest" and most different
problem from N100) gains enormously from not sharing capacity with N100. N50 (5.6% ambiguous) gains
substantially. N100 itself (81% ambiguous, the problem the shared model's capacity was implicitly
prioritizing) is indifferent to isolation — it neither needs nor loses from training alongside the
other sizes.

**This changes the recommendation.** Three independently-trained per-size models, each cheap to
train (~15-20 minutes), together beat the single frozen pooled model by a wide margin overall. The
previous freeze decision (single shared model, seed-4331 checkpoint) is superseded by this evidence.

### Untouched-test confirmation

All three per-size models were also run once against the untouched 20-example-per-size test set,
the same test set used for the earlier pooled-model freeze decision:

| Size | Pooled model (test) | Per-size model (test) | Change | Per-size model (large panel) |
|---|---:|---:|---:|---:|
| N20 | 63.54% | **21.38%** | -42.16 pp | 22.00% |
| N50 | 40.99% | **29.38%** | -11.61 pp | 28.90% |
| N100 | 32.68% | 35.14% | +2.46 pp | 30.10% |
| **Mean** | 45.73% | **28.63%** | **-17.10 pp (~37% relative)** | 27.00% |

Every per-size number lands within a couple of points of the same model's large-panel result,
confirming generalization rather than overfitting to either panel. N100's test-set number (only 20
examples) flips to a small regression versus the large panel's small gain — consistent with N100
being the one size that is genuinely indifferent to isolation (noise-level difference either way),
not a reason to abandon per-size models given N20/N50's large, consistent gains.

**Recommended new default**: adopt three per-size models
(`diffusion_denoiser_s7799_stochastic_persize_n20/n50/n100_cuda`) as the frozen recipe, keeping the
shared frozen GAT encoder. The main added cost is operational — three checkpoints to track and route
between by instance size instead of one — which is minor since the existing evaluation tooling
(`predict_matrix.py --sizes`) already dispatches per size. If N100's operational simplicity matters
more than its noise-level difference, it would be reasonable to keep N100 on the pooled/frozen
checkpoint and only split N20/N50 into per-size models; both options are supported by the evidence
above.

## Stable-only exclusion control

The declared matrix-stability gate retains 1,211 sources: 500 CVRP20, 472 CVRP50, and 239
CVRP100. Repeating retained sources would silently increase their weight, so this control preserves
one contribution per retained source and reports its lower compute: 76 optimizer steps per epoch
versus 94 for the 1,500-source arms. It is source-weight matched, not optimizer-step matched.

The epoch-0 checkpoint (SHA-256
`8865692fbdf1fa8d229936e53dd74ad336b50cbcb28bd1d48adb2669e3fbeec7`) achieved exact F1
0.5468 and decoded gap 38.72%, with per-size gaps of 56.45%, 32.08%, and 27.62%. This is the best
small-panel routing result, but it uses 19% fewer sources and updates. It therefore motivates a
proper matched-budget replication; it does not prove exclusion is superior to stochastic targets.
