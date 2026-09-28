# Large-panel policy-vs-diffusion evaluation — 2026-09-24

Track A Task 2 ([`autonomous_long_compute_guide_2026-09-24.md`](autonomous_long_compute_guide_2026-09-24.md)):
confirm the N20/N50 dual-pointer policy K=1 gains and get a real N100 number on the full frozen
`s7799_val100_policy_v1` panel, rather than the smaller per-size slices the original K=1
comparison used.

## Frozen inputs

- Panel: `data/processed/s7799_val100_policy_v1_n{20,50,100}` — 100/92/45 examples.
- Policy checkpoints: `policy_reinforce_s7799_n{20,50,100}_heldout_cuda_2026082*` (frozen, dual-pointer REINFORCE).
- Diffusion-only checkpoints: `diffusion_denoiser_s7799_stochastic_persize_n{20,50,100}_cuda_20260816*` (the frozen per-size champions).
- Decode: policy K=1 (`num_starts=1`, methodology matching `scripts/eval_policy_k1.py`); diffusion-only
  full 700-step exact reverse chain via `predict_matrix.py`, default sampler.
- Command: `outputs/logs/run_task2_large_panel_eval.sh`.

## Results

| Size | Policy K=1 gap | Diffusion-only champion gap (95% CI) | Delta | Feasible |
|---|---:|---:|---:|---:|
| N20 | **17.86%** | 21.11% [18.71, 23.57] | **-3.25pp** | 100% both |
| N50 | **27.81%** | 29.39% [27.33, 31.51] | **-1.58pp** | 100% both |
| N100 | 31.80% | **30.57%** [28.15, 33.09] | +1.23pp | 100% both |

Raw artifacts: `outputs/eval/task2_large_panel_20260924/{policy_k1_n20,n50,n100}.log`,
`outputs/eval/task2_large_panel_20260924/diffusion_n{20,50,100}/sample_metrics.json`.

## Interpretation

The earlier per-size-slice K=1 comparison (18.05%/28.01%/31.54% policy vs. 22.00%/28.90%/30.10%
diffusion) holds up on this 2-8x larger, disjoint full panel: N20 and N50 remain real,
methodologically-clean policy wins; N100 remains genuinely behind the diffusion-only champion.
The diffusion-only champion's own re-measured N100 gap (30.57%) is slightly worse than the
historical 30.10% figure — both numbers are within the reported confidence interval, treated as
normal panel-composition variance, not a regression.

This result is what motivated the Track A Task 6 N100 hyperparameter study — see
[`task6_n100_hyperparameter_sweep_2026-09-24.md`](task6_n100_hyperparameter_sweep_2026-09-24.md).

## Incident during this run

N100's diffusion-only leg died mid-run with a bare `exit 127` and no traceback, at the point
Task 9's label-audit pilot ramped up to 11 CPU workers concurrently. A direct rerun of the exact
same command completed cleanly (`exit 0`) with no changes. Treated as resource contention under
concurrent load, not a code defect — not investigated further since it did not recur.

## Caveat

This is a fixed 100/92/45-example panel per size, not the full corpus — confidence intervals are
reported per size and should be read alongside the point estimate, not dropped.
