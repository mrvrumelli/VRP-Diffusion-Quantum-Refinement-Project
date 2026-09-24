# Fixed CVRP20 reverse-sampling diagnosis

## Frozen inputs

- Checkpoint: `diffusion_denoiser_s7799_policy_v2_final_cuda`, epoch 9, SHA-256
  `8622a564c45088612a1501c0e2b688e740f69e867796aa65406d13d816fdc87a`.
- Validation source: `data/processed/s7799_val100_policy_v1`, SHA-256
  `26d3e71377c0b3554fb6fe4f5f109114438156775836887dadfa74594e3d9fd6`.
- Panel: 20 deterministic CVRP20 examples selected with seed 4350.
- Schedule: 700 timesteps; threshold 0.5; all arms use identical checkpoint and membership.
- Command: `python scripts/diagnose_reverse_sampling.py --checkpoint <best.pt> --val-dir
  data/processed/s7799_val100_policy_v1 --output
  outputs/eval/reverse_sampling_cvrp20/comparison.json --sizes 20 --per-size 20 --seed 4350
  --device cuda`.

The JSON artifact records all 20 instance IDs, resolved paths, hashes, settings, per-arm metrics,
and runtimes.

## Results

| Sampler | F1 | Precision | Recall | Decoded cost gap | Vehicles | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Exact stochastic, prior 0.50 | 0.4162 | 0.5227 | 0.3458 | 62.36% | 7.20 | 183.2 s |
| Exact deterministic, prior 0.50 | 0.3800 | 0.4635 | 0.3219 | 76.22% | 7.70 | 182.8 s |
| Exact stochastic, prior 0.2542 | 0.4162 | 0.5227 | 0.3458 | 62.36% | 7.20 | 188.6 s |
| Approximate stride-7 stochastic | 0.3809 | 0.4752 | 0.3178 | 77.39% | 7.75 | 27.2 s |

All decoded outputs were capacity-feasible. The reference-density prior produced the same final
metrics as the 0.50 prior, so the 700-step chain erased this initialization change. Deterministic
posterior thresholding and the approximate stride-7 chain both materially regressed matrix F1 and
routing cost. Exact stochastic sampling therefore remains the frozen baseline.

The 62.36% decoded cost gap and excess vehicle count show that feasibility repair alone does not
make the predicted partition useful. The next training configuration selects checkpoints on a
fixed exact full-chain panel using decoded cost gap, while retaining exact F1 in the same evidence
row: `configs/train/diffusion_denoiser_s7799_policy_v2_full_chain_cuda.yaml`.
