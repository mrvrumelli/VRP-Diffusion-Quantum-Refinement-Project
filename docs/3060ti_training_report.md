# RTX 3060 Ti post-audit training report

## Outcome

The autonomous bounded workflow completed on Windows 11 with an RTX 3060 Ti (8 GiB), Python
3.12.10, and PyTorch 2.11.0+cu128. Original corpus files were never overwritten. Every label set,
subset, audit, run, and evaluation was written to a new versioned directory.

The selected model is the epoch-9 `best.pt` checkpoint under
`outputs/train/diffusion_denoiser_s7799_policy_v2_final_cuda_20260815T025052655130Z`. Its frozen
configuration is `configs/train/diffusion_denoiser_s7799_policy_v2_final_cuda.yaml`.

## Data and audit evidence

| Artifact | Count | Hash / integrity evidence |
|---|---:|---|
| Training audit source | 1,500 (500/size) | `42e8cd106e434e66eb57a7e1521edb90259355e55c17456ff1298fd2bff17e7f` |
| Selected policy-v2 train labels | 1,500 (500/size) | `842b4bbfee69f1802bc559daf51c7f655d5fe1c70cccbc0dba10930bc8289487` |
| Frozen accepted validation labels | 237 (100/92/45) | `26d3e71377c0b3554fb6fe4f5f109114438156775836887dadfa74594e3d9fd6` source hash recorded by subset materialization |
| Untouched test source | 60 (20/size) | examples hash `84c6037e6d81a61a90a2915df2d5d71184942e9d46a70641ff92c840bfc8dd94` |

The training audit completed 6,000 PyVRP and 446 OR-Tools runs with zero errors. Matrix acceptance
was 493/500 for CVRP20, 472/500 for CVRP50, and 239/500 for CVRP100. The bounded 80- and
120-second CVRP100 follow-ups showed that more solver time does not reliably resolve partition
ambiguity. The final untouched-test audit completed 240 PyVRP and 28 OR-Tools runs with zero
errors; matrix acceptance was 20/20, 18/20, and 9/20 by increasing size.

## CUDA and reliability gates

- CUDA preflight and backward pass succeeded; compute capability 8.6, CUDA 12.8, cuDNN 9.1.9.
- Four CUDA tests passed.
- End-to-end one-epoch GAT and diffusion smoke passed.
- Diffusion peak allocated memory was about 904 MB (reserved about 1.03 GB); no OOM occurred.
- Resume was verified from epoch 0 into epoch 1 with optimizer/scaler/model state and a new
  checkpoint.
- Windows launcher path handling, manifest filtering, AMP overflow recovery in both trainers, and
  CP1254 inference output were fixed as encountered.

## Controlled selection

Policy v2 was chosen after the compute-matched label comparison documented in
`label_audit_s7799_decision.md`. With that data and encoder fixed, five-epoch diffusion results were:

| Variant | Validation AUC | BCE | F1 |
|---|---:|---:|---:|
| High timestep baseline, lr 3e-4 | 0.9128 | 0.2258 | 0.5864 |
| **Uniform timesteps, lr 3e-4** | **0.9261** | **0.2111** | **0.6150** |
| Uniform/high control, lr 1e-4 | 0.9023 | 0.2461 | 0.5620 |
| Uniform/high control, lr 5e-4 | 0.9133 | 0.2248 | 0.5867 |
| Positive-weight power 0.25 | 0.9116 | 0.2174 | 0.5838 |

Uniform sampling replicated at seeds 4330/4331/4332 with AUC 0.9261/0.9275/0.9186 and F1
0.6150/0.6238/0.6003. At seed 4331, high sampling reached only AUC 0.9167 and F1 0.5954.

The fixed five-epoch learning curve justified using the full audited pool:

| Sources per size | Total train examples | Validation AUC | F1 |
|---:|---:|---:|---:|
| 100 | 300 | 0.8954 | 0.5505 |
| 250 | 750 | 0.9165 | 0.5960 |
| 500 | 1,500 | 0.9261 | 0.6150 |

## Final model and exact full-chain evaluation

The selected seed was resumed to a 15-epoch ceiling and stopped after epoch 13. Best validation
occurred at epoch 9: noisy-time AUC 0.9453, BCE 0.1812, and F1 0.6695. Exact reverse-chain
evaluation used all 700 steps and fixed hard threshold 0.5:

| Frozen set | Examples | AUC | F1 | CVRP20 F1 | CVRP50 F1 | CVRP100 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 60 (20/size) | 0.7502 | 0.4238 | 0.4162 | 0.4319 | 0.4219 |
| Untouched test | 60 (20/size) | 0.7573 | 0.4350 | 0.4206 | 0.4522 | 0.4305 |

## Limitations

- Exact full-chain quality is materially below noisy-time denoising metrics; the latter must not be
  presented as end-to-end routing quality.
- The untouched test has only 20 examples per size. Strict matrix acceptance is only 9/20 for
  CVRP100, so its score has both sampling uncertainty and reference ambiguity.
- This evaluation measures constraint-matrix recovery, not decoded route feasibility or route-cost
  optimality. A decoder/quantum-refinement comparison was not available in this training path.
- Only 500 strongly audited training sources per size exist. Scaling beyond 1,500 total examples
  would require new versioned label generation and another audit gate.
