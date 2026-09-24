# Project findings and action plan — 2026-09-24

## Executive status

The project has a working and well-tested classical CVRP research pipeline: synthetic data
generation, heuristic and audited labeling, GAT pretraining, constraint-matrix diffusion,
capacity-aware decoding, and a dual-pointer REINFORCE policy. The main IID corpus contains
200,001 instances (66,667 each for CVRP20/50/100), with deterministic train/validation/test
splits of 180,000/10,002/9,999 examples.

The strongest controlled training evidence still comes from 500 strongly audited sources per
size, not the full corpus. The 100/250/500-per-size learning curve was still improving at 500, so
additional audited data is justified as a bounded learning-curve experiment. It is not justified
to audit all 180,000 training examples before establishing that another modest increase improves
decoded route quality.

The current recommended diffusion baseline is three size-specialized stochastic-reference
denoisers sharing a frozen GAT encoder. The policy improves the single-decode route gap at N20 and
N50 but has not yet beaten diffusion-only decoding at N100. Learned routes are consistently
capacity-feasible, but their cost gaps remain far above strong PyVRP solutions. Quantum and
quantum-inspired local refinement have not yet been implemented.

## Evidence available on 2026-09-24

### Data and labels

- Main generated corpus: 200,001 IID examples.
- Main split: 60,000 train, 3,334 validation, and 3,333 test examples per size.
- Strong IID audit: 1,500 sources, 500 per size.
- Stable matrix acceptance: 493/500 at N20, 472/500 at N50, and 239/500 at N100.
- Independent spatial stress corpus: 9,000 examples across R/C/RC, three sizes, and 1,000
  examples per distribution-size cell.
- The full 9,000-source spatial pool was subsequently materialized into audited training-label
  sets despite the older written pause note. The policy-v1 manifest records 13,974 competitive
  labels from 9,000 sources (3,000/3,215/7,759 at N20/N50/N100); policy v2 records 9,007 labels
  (3,000/3,000/3,007). Both identify source hash
  `626006f44e752c2a7ffe497e34b52f2f8a33a794a2ec9bd8426bfc140664c4fb`.
- The original `outputs/label_audit/rc_full` audit bundle referenced by those manifests is no
  longer present in this checkout. The recoverable provenance, label multiplicities, merged-set
  hashes, and trained-checkpoint hashes are now archived in
  [`rc_full_artifact_reconstruction_2026-09-24.md`](rc_full_artifact_reconstruction_2026-09-24.md).
  Raw stable-acceptance, candidate-runtime, and failure statistics remain unrecoverable without a
  backup or rerun and are not inferred from the materialized labels.
- Size-specific merged training sets were used for later `rcfull` models: 3,500 N20 examples,
  3,751 N50 examples, and 9,027 N100 label examples. Completed 15-epoch runs record dataset hashes
  and training metrics. The first N20 attempt has no final metrics and was superseded by the later
  completed N20 run.

### Diffusion training

- The original pooled policy-v2 model completed full-chain validation and untouched-test
  evaluation but was superseded.
- Stochastic-reference training was more reliable across seeds than stable-only exclusion.
- Three size-specialized denoisers reduced the large-panel mean decoded gap from 46.03% to
  27.00% and the small untouched-test mean from 45.73% to 28.63%.
- The best controlled learning curve ended at 500 audited sources per size and had not clearly
  saturated.

### Policy training

- End-to-end policy runs completed at N20/N50/N100 with 100% capacity feasibility.
- Fair K=1 gaps were 18.05% at N20, 28.01% at N50, and 31.54% at N100.
- These improve over size-specialized diffusion at N20 and N50; N100 remains behind the 30.10%
  diffusion-only large-panel result and needs a bounded tuning study.

### Evaluation and engineering health

- Capacity-aware route metrics, matrix metrics, ambiguity evaluation, synthetic evaluation,
  spatial validation, OR comparisons, CVRPLIB parsing, and failure analysis exist.
- Current committed OR and CVRPLIB reports are smoke-scale, not final benchmarks.
- On 2026-09-24, `pytest -q` passed 427 tests and `mypy src` passed for 50 source files.
- `ruff check .` reported 27 violations concentrated in recently added evaluation scripts/tests;
  this is the first no-training repair item.

## Ordered action plan

### A. No-training work — do first

1. **Completed 2026-09-24:** restored Ruff and mypy quality gates; the full CPU suite passes.
2. **Reconstruction archived 2026-09-24:** surviving manifests now permanently document all 9,000
   materialized sources, both label policies, per-cell multiplicity, merged-set hashes, and model
   checkpoint hashes. The missing raw audit statistics are explicitly bounded rather than guessed.
3. **Candidate completed 2026-09-24:**
   `configs/eval/classical_baseline_v1_candidate.yaml` freezes and hash-validates the exact GAT,
   per-size diffusion and policy checkpoints, inference choices, and current validation/test
   datasets. It remains a candidate until the declared large-evaluation and environment gates pass.
4. **Completed 2026-09-24:** consolidated standalone and policy reverse-diffusion sampling behind
   one tested reverse-chain implementation.
5. **Completed 2026-09-24:** the policy now builds the local-attention prior once per encoding and
   shares it with the local encoder and pointer mask.
6. **Completed implementation 2026-09-24:** full-chain diffusion evaluation and greedy policy
   dataset solving support optional same-size batching while retaining batch size 1 as the
   historical compatibility default. Policy summaries now record effective batch size and
   throughput. A representative CUDA batch-size sweep remains evaluation work, not training work.
7. **Completed 2026-09-24:** canonical routing evaluation now records positive-edge TP/FP/FN,
   precision/recall, vehicle delta/inflation, singleton collapse, route-size histograms, and repair
   events. Full-chain evaluation inherits these metrics automatically.
8. **Confidence tooling completed 2026-09-24:** routing summaries can report deterministic seeded
   percentile-bootstrap intervals for feasible decoded cost gaps, overall and per size. The
   full-chain CLI defaults to 95% intervals with 10,000 resamples and archives those settings.
   Matched multi-method panels and their runtime comparisons remain a separate compute step.

### B. Short evaluation work — no retraining, but requires model/runtime compute

1. Confirm the N20 and N50 policy gains on a larger frozen panel.
2. Evaluate frozen policy, diffusion-only, PyVRP, and OR-Tools under declared matched conditions.
3. Expand CVRPLIB beyond the one-instance parser smoke test.
4. Evaluate existing `rcfull` checkpoints on IID and all R/C/RC cells after artifact provenance is
   reconciled.

### C. Training and labeling work

1. Add approximately 500–1,500 new audited IID sources per size, preserving competitive
   multi-reference solutions, then run a 500/1,000/2,000-per-size learning curve.
2. Stop label expansion when frozen-panel decoded route-gap gains flatten; do not use noisy-time
   AUC or matrix F1 alone as the stopping criterion.
3. Run a bounded N100 policy study covering longer training, early stopping, learning-rate
   scheduling, entropy regularization, normalization, number of starts, and curriculum options.
4. Target a clear N100 gate: beat the 30.10% diffusion-only large-panel gap without losing
   feasibility.

### D. Refinement and quantum work — only after the classical baseline is frozen

1. Implement deterministic local-neighborhood extraction and a matched classical refiner.
2. Define and exhaustively validate small QUBO formulations on 20–50 hand-checkable cases.
3. Build leakage-free development/validation neighborhood sets.
4. Screen quantum-inspired and quantum approaches in simulation before hardware experiments.
5. Report CMD-only, CMD plus classical refinement, CMD plus quantum refinement, and quantum plus
   classical polish from identical initial solutions and under matched budgets.

## Completion criteria for `baseline-v1.0`

`baseline-v1.0` is ready only when the exact artifacts and hashes are frozen, static quality gates
pass, the policy-versus-diffusion choice is explicit for every size, IID and spatial evaluation is
large enough to report uncertainty, classical baselines use declared comparable budgets, and all
results include feasibility, route quality, vehicle inflation, runtime, seeds, and configuration.
