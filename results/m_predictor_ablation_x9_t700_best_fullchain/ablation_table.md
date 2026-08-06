# Ablation: P2.1 vs P3 (best full-chain checkpoint)

**Selection:** max full-chain hard F1 over available checkpoints.

| ckpt | epoch | full-chain F1 | one-shot F1 |
| --- | --- | --- | --- |
| best.pt | 1 | 0.4570 | 0.5808 |
| **last.pt (selected)** | **17** | **0.4789** | **0.5909** |

Checkpoint: `outputs/train/diffusion_denoiser_20260806T042656342740Z/checkpoints/last.pt` · stride=1.

| method | f1 | auc | precision | recall | pair_acc | f1_n20 | f1_n50 | f1_n100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2.1_supervised | 0.5571 | 0.9106 | 0.5176 | 0.6031 | 0.8696 | 0.6203 | 0.5808 | 0.5433 |
| P3_one_shot | 0.5909 | 0.9223 | 0.5228 | 0.6794 | 0.8765 | 0.6184 | 0.6081 | 0.5833 |
| P3_full_chain | 0.4789 | 0.7159 | 0.4608 | 0.4985 | 0.8508 | 0.5631 | 0.4572 | 0.4795 |

## Deltas vs P2.1

| method | ΔF1 | ΔAUC |
| --- | --- | --- |
| P3_one_shot | +0.0338 | +0.0117 |
| P3_full_chain | -0.0781 | -0.1947 |
