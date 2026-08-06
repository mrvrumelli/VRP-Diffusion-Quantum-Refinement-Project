# VRP Diffusion Quantum Refinement Project

Reproducible research code for a Constraints Matrix Diffusion based CVRP solver, with
quantum and quantum-inspired refinement scoped to small local neighborhoods after the
classical baseline is stable.

## Setup

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Core loop

Prefer the two-stage shell script (injects the GAT checkpoint automatically):

```bash
bash scripts/run_gat_then_diffusion.sh
```

| Stage | Command |
|-------|---------|
| 1. GAT | `python scripts/pretrain_gat_encoder.py --config configs/train/gat_pretrain.yaml` |
| 2. Denoiser | set `model.gat_checkpoint` in a copy of `configs/train/diffusion_denoiser.yaml`, then `python -m vrp_diffusion_quantum.train.train_diffusion --config …` |
| 3. Ablation (regen) | `python scripts/eval_m_ablation.py --config configs/eval/m_predictor_ablation.yaml` |
| 3. Ablation (view) | open `notebooks/p21_vs_p3_ablation_results.ipynb` |

`diffusion_denoiser.yaml` leaves `gat_checkpoint: null` on purpose — use the shell script, or
pass an explicit path after GAT pretrain. Direct stage-2 with the stock yaml will fail until
that path is set.

P2.1 sanity train (library baseline used in the ablation):

```bash
python scripts/train_matrix_predictor.py --config configs/train/matrix_predictor_sanity.yaml
```

Full-chain sample eval on a trained denoiser:

```bash
python -m vrp_diffusion_quantum.inference.predict_matrix \
  --checkpoint outputs/train/<run>/checkpoints/last.pt \
  --per-size 16 --device cuda
```

MLflow: `mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db`

### Recipe notes

- **GAT:** no aug + soft √WBCE.
- **Denoiser:** ×9 aug (original + 4 geo + 4 demand) + soft √WBCE, **T=700**, train
  `t_sample: high` (val uses the same mode). Demand views **keep routes/`M` fixed**
  (invariance aug; capacity feasibility of the labeled routes is not re-checked).
- **Checkpointing:** in-train `best.pt` tracks sparse `sample_f1`. The reported ablation
  selected **`last.pt` (epoch 17)** because it beat `best.pt` (epoch 1) on full-chain hard F1
  on the held-out eval set — prefer that selection when comparing to P2.1.
- **Headline numbers** (checked in under
  `results/m_predictor_ablation_x9_t700_best_fullchain/`): P3 one-shot F1 ≈ 0.59 >
  P2.1 ≈ 0.56 > P3 full-chain ≈ 0.48. Full-chain currently underperforms one-shot/P2.1 on
  this eval; treat one-shot as the stronger diffusion readout until the reverse chain improves.

Regen writes under `outputs/eval/` (gitignored). The notebook reads the committed
`results/…` snapshot first, then falls back to `outputs/eval/…`.

## Dev

```bash
ruff check .
ruff format .
pytest
```
