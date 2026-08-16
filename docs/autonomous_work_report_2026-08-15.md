# Autonomous work report — 2026-08-15

This work block converted the highest-priority model-diagnosis and data-readiness TODOs into
reproducible code, tests, configs, and durable evidence. All changes remain uncommitted for review.

## Delivered

- Added capacity-aware matrix-to-route decoding and route-level evaluation, including feasibility,
  repair, cost gap, vehicle count, route size, runtime, and post-decoding matrix metrics.
- Diagnosed exact reverse sampling on a fixed CVRP20 panel. Exact stochastic transitions beat the
  deterministic and stride-7 alternatives; the settings and evidence are recorded in
  [`reverse_sampling_diagnosis.md`](reverse_sampling_diagnosis.md).
- Added fixed-panel full-chain checkpoint selection and frozen oracle best-of-K ambiguity metrics.
- Added deterministic stochastic-reference sampling, fractional consensus targets, confidence
  masks, and an audited-source exclusion arm without weakening binary stored matrices.
- Replaced ordinary eager corpus loading with an indexed JSON dataset and bounded LRU cache.
  Newly saved matrices use validated packbits/base64 storage while historical JSON matrices remain
  readable.
- Regenerated and validated all 9,000 R/C/RC plain-CVRP spatial-stress instances. The validation
  covers local distance, global dispersion, grid entropy, silhouette, and representative plots.
- Verified zero exact input overlap between the 3,000 independent R instances and the 200,001-item
  `s7799` corpus. See [`spatial_stress_validation.md`](spatial_stress_validation.md).

## Probe outcome

All probe checkpoints were compared on the same fixed full-chain panel, but the exclusion arm had
fewer optimizer steps and is therefore not a final compute-matched ablation.

| Arm | Best epoch | Exact sample F1 | Decoded cost gap |
|---|---:|---:|---:|
| Existing policy-v2 baseline | — | 0.4254 | 76.59% |
| Stochastic reference | 0 | **0.5498** | 43.37% |
| Consensus target | 3 | 0.4753 | 46.29% |
| Confidence-masked consensus | 0 | 0.5501 | 52.04% |
| Audit exclusion | 0 | 0.5468 | **38.72%** |

The current evidence supports stochastic-reference training as the cleanest model-side gain.
Exclusion has the best observed route gap, but needs an optimizer-step-matched rerun before it can
be compared fairly. Full per-size results, hashes, and limitations are in
[`stochastic_reference_probe.md`](stochastic_reference_probe.md).

## Verification

- `ruff check .`: passed
- `mypy src`: passed for 46 source files
- Touched-file `ruff format --check`: passed for 23 files
- Full CPU suite: **352 passed**
- `git diff --check`: passed

## Recommended next actions

1. Run policy-v2, stochastic, exclusion, consensus, and masked-consensus arms with identical source
   exposure and optimizer-step budgets.
2. Require an improvement on CVRP100 without material CVRP20/CVRP50 regression before freezing the
   sampler, target policy, checkpoint rule, and decoder.
3. Label the independent R/C/RC sets with the stronger multi-seed policy before model evaluation.
4. Add configurable loader workers, prefetching, pinned memory, and non-blocking CUDA transfers,
   retaining safe single-worker defaults and benchmarking before enabling them broadly.

