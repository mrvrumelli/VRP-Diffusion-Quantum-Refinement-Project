# Phase 3 faithful diffusion status — 2026-09-24

**Correction, 2026-09-30 (R3):** Earlier mathematical-completeness claims below are superseded. The legacy sampler normalized a soft clean prior instead of mixing normalized hard-state posteriors (eq. 9). The explicit `posterior_mixture_v2` path now implements the mixture and passes independent binary-chain enumeration. Legacy `skipped_posterior` retains historical behavior. This fixes inference mathematics; it does not establish trained-model reproduction. See [execution evidence](corrective_execution_2026-09-30.md).

## Result

The Phase 3 engineering implementation and automated mathematical gates are complete. The
empirical reproduction gate is still open because no trained `paper_cmd` GAT/denoiser checkpoint
or frozen paper-compatible evaluation panel exists in the repository yet.

Phase 3 therefore has two distinct statuses:

- **Implementation:** complete.
- **Paper F1 curve:** blocked on the Phase 2 data artifacts and a full diffusion training run.

The policy path may be exercised by unit and smoke tests, but it must not be reported as a
validated paper reproduction until the remaining empirical gate passes.

## Configuration and architecture audit

`configs/train/diffusion_denoiser_paper_cmd.yaml` now enforces:

- 1,000 diffusion timesteps with a linear beta schedule from 0.0001 to 0.02;
- hidden and timestep embedding dimensions of 128;
- a frozen five-layer, eight-head pretrained GAT;
- five anisotropic denoising layers;
- BatchNorm in the edge and node updates from equations 11 and 13;
- paper-compatible `e^0 = x_t` edge input, without the robust path's distance feature;
- batch size 32 and 50 epochs;
- 50-step inference using the arbitrary-interval skipped posterior.

The alignment contract rejects `paper_cmd` diffusion configurations that change the normalization
or edge-input mode. Legacy and `ours_robust` models retain LayerNorm and the
`[noisy_matrix, distance]` edge input by default, so their checkpoint format and behavior are not
silently reclassified as paper-compatible.

## Skipped-transition implementation

For arbitrary schedule states `-1 <= s < t`, the implementation computes
`q(x_s | x_t, x_0)` from the cumulative bit-flip signal before `s` and the complete transition
product from `s` to `t`. State `s=-1` denotes the clean matrix. The historical repeated one-step
approximation remains available only as the explicit `one_step_approx` compatibility sampler;
new standalone and policy inference defaults use `skipped_posterior`.

The training-time sample evaluator now consumes both `sample_eval.num_inference_steps` and
`sample_eval.sampler`. Previously these keys were present in the paper YAML but were ignored,
which meant the intended 50-step curve could not control checkpoint evaluation.

## Automated evidence

Focused verification on 2026-09-24:

- 81 focused diffusion, denoiser, inference, training, alignment, and checkpoint tests passed;
- the complete CPU repository suite passed with 507 tests;
- Ruff passed on all touched Python files;
- Mypy passed on all 54 source files.

The tests independently establish that:

1. the skipped posterior equals the adjacent analytical posterior for `s=t-1`;
2. arbitrary skipped probabilities match brute-force enumeration of toy two-state chains;
3. every stored reverse transition is binary, symmetric, and zero-diagonal;
4. stochastic skipped sampling is bitwise repeatable under a fixed seed;
5. BatchNorm and `e^0=x_t` are active in the paper denoiser;
6. paper-mode denoiser checkpoints round-trip with their architecture intact;
7. checkpoint-selection evaluation receives the requested step count and sampler.

## Remaining empirical gate

The repository contains no trained `paper_cmd` diffusion checkpoint and no frozen Phase 2
paper-compatible test panel. Consequently, it is not yet possible to claim the paper's qualitative
inference-step curve or approximately 0.823 F1 at 50 steps.

After Phase 2 data generation and GAT pretraining, train the denoiser with:

```powershell
python -m vrp_diffusion_quantum.train.train_diffusion `
  --config configs/train/diffusion_denoiser_paper_cmd.yaml
```

Then evaluate a fixed panel at several inference budgets with the exact skipped posterior:

```powershell
python -m vrp_diffusion_quantum.inference.predict_matrix `
  --checkpoint outputs/paper_cmd/<run>/checkpoints/best.pt `
  --val-dir <frozen-paper-compatible-test-directory> `
  --per-size 1000 `
  --sizes 20 50 100 `
  --inference-steps 5 10 20 50 100 200 1000 `
  --sampler skipped_posterior `
  --device cuda `
  --output-dir outputs/paper_cmd/<run>/inference_step_curve
```

This writes `inference_step_curve.json`. Phase 3 closes only after the curve is reviewed against
the paper, the 50-step result is approximately 0.823 F1, and the checkpoint, panel manifest,
dataset hashes, seed, and environment are archived together.
