# VRP Diffusion Quantum Refinement Project

Reproducible research code for a Constraints Matrix Diffusion based CVRP solver, with
quantum and quantum-inspired refinement scoped to small local neighborhoods after the
classical baseline is stable.

## Setup

Use Python 3.12.

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

Before running the ablation, copy its config and set the training, validation-selection, and
untouched test directories plus a `best.pt` checkpoint selected during training.

Create those directories from exported, labeled `CVRPExample` JSONs with a seeded,
size-stratified split. The command refuses to overwrite a non-empty destination and records the
source hash, exact membership, per-size counts, and split hashes in `split_manifest.json`:

```bash
python scripts/make_splits.py \
  --source data/raw/cvrp/<run>/examples \
  --output data/raw/cvrp/<run>/splits \
  --seed 42
```

Generate the separate P3.6 shifted-demand source dataset from its committed config, then label
and convert it through the dataset app before splitting/evaluating it:

```bash
python -m vrp_diffusion_quantum.data.generate_cvrp \
  --config configs/data/cvrp_shifted_demands.yaml
```

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

P3.6 generalization stress test (configure a CVRP20-only checkpoint, disjoint validation/test
paths, and a shifted-demand dataset first):

```bash
python scripts/evaluate_diffusion_generalization.py \
  --config configs/eval/diffusion_generalization.yaml
```

The run writes a generalization table with F1 drop relative to in-distribution CVRP20, runtime,
dataset hashes, and the ten worst examples per case.

MLflow: `mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db`

### Recipe notes

- **GAT:** no aug + soft √WBCE.
- **Denoiser:** x9 label-preserving geometric augmentation (original + 4 D4 views + 4
  45-degree-offset rotations) + soft sqrt-WBCE, **T=700**; `t_sample: high` (validation uses
  the same mode).
- **Checkpointing:** `best.pt` is selected only from validation metrics during training. The
  ablation tunes soft decision thresholds on the validation split and reports final metrics once
  on a separate test split.

Regeneration writes a timestamped run under `outputs/eval/` with config, hashes, seed, commit,
runtime, metrics, and logs. Generated checkpoints and result snapshots remain gitignored.

## Dev

```bash
ruff check .
ruff format .
pytest
```
