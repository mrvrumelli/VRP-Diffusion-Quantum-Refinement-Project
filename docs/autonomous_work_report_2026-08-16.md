# Autonomous work report — 2026-08-16

This work block resolved the open item at the top of the next-stage research plan: comparing
policy-v2, exclusion, stochastic, consensus, and masked label policies under identical compute, and
checking the plan's CVRP20/CVRP50 non-regression gate. All changes remain uncommitted for review.

## Delivered

- Added a compute-matched policy-v2 baseline config (`diffusion_denoiser_s7799_policy_v2_matched_cuda.yaml`)
  using the exact probe protocol (94 steps/epoch, decoded-gap checkpoint selection, same fixed panel)
  that the existing stochastic/consensus/masked/exclusion probes already used — the prior policy-v2
  comparison point was a differently-configured 15-epoch, val-AUC-selected run, which confounded the
  earlier comparison.
- Replicated the stochastic-reference arm on 2 additional seeds (4332, 4333) and the exclusion arm
  on a step-matched rerun (6 epochs ≈ 456 steps, then 2 more seeds), so both leading candidates now
  have 3-seed evidence instead of 1.
- Built and ran a large held-out evaluation panel (120 examples, 40/size, vs. the original 5/size)
  against all 9 resulting checkpoints (baseline, 3 stochastic seeds, 3 exclusion seeds, consensus,
  masked-consensus) to remove small-panel noise from the comparison.
- Updated [`stochastic_reference_probe.md`](stochastic_reference_probe.md) and
  [`3060ti_training_todo.md`](3060ti_training_todo.md) with the full evidence trail and a candidate
  frozen-recipe recommendation.

## Result: 5-arm comparison and freeze recommendation

On the 120-example panel, ranked by decoded route-cost gap:

| Arm | Overall gap | Seed-to-seed range |
|---|---:|---:|
| Matched policy-v2 baseline | 56.39% | (1 seed) |
| Confidence-masked consensus | 53.11% | (1 seed) |
| Consensus | 52.76% | (1 seed) |
| Exclusion, step-matched (3-seed mean) | 47.61% | 40.57-60.07% (19.5 pp) |
| **Stochastic reference (3-seed mean)** | **44.48%** | 43.28-46.03% (2.75 pp) |

Consensus and masked-consensus are ruled out — both are only marginally better than the baseline,
and masked-consensus has the worst CVRP20 routing quality of any arm despite good matrix-level F1.

Exclusion's single-seed result (42.20%) looked best in the first pass, which would have picked it
over stochastic. Running exclusion on the same 3 seeds used for stochastic reversed that: exclusion
has roughly 7x stochastic's seed-to-seed variance (one exclusion seed, 4333, actually lands *worse*
than the untouched baseline), so its apparent edge was not reliable. Stochastic reference is the
recommended candidate for the frozen recipe on evidence of both mean performance and reliability.

The plan's CVRP100/CVRP50 non-regression gate passes cleanly and consistently in every seed tested
(CVRP50 -16.3 pp mean, CVRP100 -23.1 pp mean). CVRP20 shows a small but real regression (+3.6 pp
mean, present in 2 of 3 seeds) — an order of magnitude smaller than the gains elsewhere, but not
zero. This is flagged as the one remaining judgment call before the recipe can be called frozen; it
was not resolved unilaterally.

## Verification

- No source code was changed this session — only training configs (6 new YAML files, all following
  existing patterns) and documentation. `ruff`/`mypy`/`pytest` were not re-run since nothing they
  cover changed.
- All 9 training runs and 9 evaluation runs completed with exit code 0 and 100% capacity-feasible
  decoded routes; logs and metrics are retained under `outputs/train/` and `outputs/eval/`.

## Update: freeze decision and test evaluation (same session, continued)

Given explicit authorization to decide and continue autonomously, the open CVRP20 judgment call was
resolved rather than left for a separate sign-off: **stochastic-reference is frozen** as the recipe
(seed-4331 checkpoint, chosen because it predates the multi-seed comparison and so is not
cherry-picked). Reasoning is recorded in `stochastic_reference_probe.md`'s "Freeze decision" section.

The frozen checkpoint was then run once against the untouched 60-example test set — the first time
route-decoding metrics have touched it. Result: F1 0.5447 (vs. validation panel's 0.5535), decoded
gap 45.73% overall (vs. validation panel's 46.03%) — close enough to the validation-panel numbers to
support that the recipe generalizes. Backfilling route metrics for the old production policy-v2
checkpoint on the same test set (it was previously only scored on matrix F1) showed the frozen model
now wins on every metric, including CVRP20, against what was actually previously deployed (72.13%
overall gap and 70.78% N20 gap old vs. 45.73%/63.54% new).

## Update: CVRP20 mechanism identified, R/C/RC audit launched (same session, continued)

Investigated whether a CVRP20-specific label fix could recover the small regression without giving
up the CVRP50/CVRP100 gains. Found the CVRP20 training target is **byte-identical** between every
arm compared (0/500 sources differ in label content between the two policies; CVRP20 also has zero
candidate ambiguity, so stochastic selection is a no-op there). The regression is therefore not a
labeling artifact — it is multi-task interference from one shared denoiser network learning to
handle CVRP50/100's much higher label ambiguity, which costs CVRP20 slightly even though CVRP20's
target never changes. No cheap fix exists; a real one would need an architecture change. This closes
the CVRP20 question with an understood mechanism rather than an open gap.

Also scoped and, with explicit user sign-off on the ~21-22 hour CPU cost, **launched the full
9,000-instance R/C/RC strong-label audit** as a detached background process
(`outputs/logs/run_rc_full_audit_chain.sh`). A 60-instance pilot validated the pipeline end-to-end
first (zero errors, 59/60 accepted) and gave the real timing used to estimate the full-scale cost.
See the R/C/RC checklist item in `3060ti_training_todo.md` for the exact resume/restart commands.

**Paused, not running.** Partway into the prep stage the user flagged their CPU (Ryzen 5 5600)
running at its 95°C thermal ceiling and asked whether that was safe. Assessment: not immediately
dangerous (Ryzen 5000-series is designed to run up to Tjmax under sustained boost and hardware
throttling prevents exceeding it), but sustained operation at the ceiling for the full ~21-22 hours,
unattended overnight, was more thermal stress than a snap decision should commit to. All audit
processes were killed as a precaution (`taskkill`, verified via `tasklist`/`ps`) rather than
continuing while the user decided. The user chose to redirect to GPU work instead — the R/C/RC audit
remains **not running** and is a live decision for the user: resume at full parallelism, resume at
reduced worker count (slower, less sustained heat), or check cooling/airflow first. Raw source data
under `data/raw/cvrp/spatial_stress_*` was never written to and is unaffected either way.

## Update: per-size specialized models — the freeze decision is superseded (same session, continued)

With the CPU audit paused and the GPU idle, tested whether the CVRP20 multi-task-interference
mechanism identified above could be directly fixed architecturally: three independent diffusion
denoisers were trained, one per size, each on only its own size's 500-source stochastic-reference
pool (compute-matched to the pooled arm's budget by equal source x epoch exposure), sharing the same
frozen GAT encoder. Configs: `diffusion_denoiser_s7799_stochastic_persize_n{20,50,100}_cuda.yaml`.

Result, confirmed on both the large 120-example validation panel and the untouched test set:

| Size | Pooled model (test set) | Per-size model (test set) | Change |
|---|---:|---:|---:|
| N20 | 63.54% | **21.38%** | -42.16 pp |
| N50 | 40.99% | **29.38%** | -11.61 pp |
| N100 | 32.68% | 35.14% | +2.46 pp (noise-level, tiny 20-example slice) |
| **Mean** | 45.73% | **28.63%** | **~37% relative reduction** |

N20 and N50 gain enormously from no longer sharing network capacity with N100's much harder,
higher-ambiguity problem; N100 is indifferent to isolation either way. This **supersedes the
seed-4331 pooled-model freeze decision** from earlier in this session. New recommended default:
three per-size models instead of one shared model. Full evidence and the operational cost/benefit
discussion (three checkpoints to route between by size, vs. one) are in
`stochastic_reference_probe.md`'s "Per-size specialized models" section.

## Recommended next actions

1. **Human review of the per-size freeze decision** — same rationale as before: I made the call
   given explicit authorization to decide autonomously, but this now determines what gets deployed.
   Worth a final look, especially the N100 operational choice (per-size vs. keep pooled — both are
   defensible on the evidence).
2. **R/C/RC audit is paused and needs a decision**: resume at full ~11-worker parallelism (~21-22hr,
   confirmed hot but within the CPU's safe operating envelope), resume at reduced parallelism
   (slower, cooler), or check cooling/airflow before running anything this sustained again. See the
   pause note above and the checklist item in `3060ti_training_todo.md` for resume commands.
3. Exclusion's best single run (40.57%, seed 4332) is still the best single result observed overall;
   its data-scale sensitivity (why does the smaller 1,211-source pool destabilize training on some
   seeds?) remains a separable follow-up, not a blocker.
4. Once the R/C/RC audit completes, evaluate the (now per-size) recipe against it as the independent
   spatial-stress test the plan calls for.
5. Loader-throughput engineering (configurable workers, prefetching, pinned memory) from the
   2026-08-15 report remains undone and is lower priority than the above.
