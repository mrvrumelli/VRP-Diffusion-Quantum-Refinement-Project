# Single-size normalization control — 2026-09-30

Declared before launch, after the mixed-size diagnostics and source-count curve.
This is a bounded follow-up to R9, not a newly selected baseline.

- Hypothesis: mixed-size batch traversal alone explains the observed BatchNorm instability.
- Compare LayerNorm (`baseline`) and BatchNorm (`bn_shuffled`) on N100 only.
- Reuse exactly the existing 32 N100 training sources and eight development graphs.
- Train both arms from scratch with seeds 4331 and 4332, 600 updates, batch 4,
  width 64, three jointly trained GAT/denoiser layers, weighted BCE, absolute inputs.
- Keep final-update checkpoints, corrected 50-step generation, seed 9001, batch 1.
- Record generated F1, mean instance-relative route gap and fixed-noise calibration.
- Four runs only. No normalization promotion based on this eight-graph diagnostic.
  If BatchNorm remains inconsistent, reject batch ordering as a sufficient explanation;
  do not claim that normalization alone explains the full model's F1 deficit.

Command:

```powershell
.venv/Scripts/python.exe scripts/run_corrective_learning_checks.py --arms baseline bn_shuffled --seeds 4331 4332 --updates 600 --single-size 100 --output outputs/corrective_normalization_n100_20260930
```
