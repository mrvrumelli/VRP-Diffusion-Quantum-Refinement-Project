# Policy dataset-batching profile — 2026-09-24

## Result

Same-size policy batching substantially improves end-to-end inference throughput on the RTX 3060
Ti. In a 24-instance N20 smoke profile, batch size 8 was approximately 6.0 times faster than the
historical batch-size-1 path.

| Batch size | Instances/s | Mean runtime/instance (s) | Feasible | Mean gap (%) | Routes differing from batch 1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.277 | 0.7831 | 24/24 | 15.946 | — |
| 4 | 4.638 | 0.2156 | 24/24 | 16.357 | 2/24 |
| 8 | 7.688 | 0.1301 | 24/24 | 15.946 | 0/24 |

Relative throughput was 3.63x at batch size 4 and 6.02x at batch size 8.

## Frozen profile conditions

- GPU: NVIDIA GeForce RTX 3060 Ti, 8 GiB.
- Dataset: `data/processed/s7799_val100_policy_v1_n20`.
- Selection: 24 instances, seed 240924.
- Policy checkpoint SHA-256:
  `6b2a840c897f4c8d55b368005c918bd4616ad030a5b9357f946efd034a308877`.
- Denoiser checkpoint SHA-256:
  `6ff75ed0ae7c4d2981ab41440b9dd5787d83ab18ae9ad712987bbd9941f02b0c`.
- Prior: 50 diffusion inference steps.
- Decode: greedy, one start, one augmentation, one candidate per instance.
- Confidence interval: seeded 95% percentile bootstrap, 10,000 resamples.

Raw run artifacts are under `outputs/profiling/policy_batch_n20_b{1,4,8}`.

## Interpretation

CUDA batched kernels need not be bitwise equivalent to a sequence of batch-size-1 kernels. Here,
batch size 4 changed two greedy route choices and shifted the mean gap by 0.41 percentage points;
batch size 8 happened to reproduce every batch-size-1 route. This small smoke profile is not enough
to claim that batch size 8 is universally output-equivalent.

Consequently:

- batch size is part of the evaluation configuration and must be reported;
- baseline-v1.0 remains frozen at batch size 1 for historical comparability;
- larger panels may use batching for throughput, but comparisons must use the same fixed batch size
  for every checkpoint/method run being compared;
- a larger N20/N50/N100 sweep should choose production batch sizes based on memory, throughput, and
  route-stability evidence.
