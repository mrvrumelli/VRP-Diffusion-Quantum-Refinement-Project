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
bash scripts/run_gat_then_diffusion.sh \
  configs/train/gat_pretrain_s7799.yaml \
  configs/train/diffusion_denoiser_s7799.yaml
```

The `s7799` configs use the local, gitignored
`cvrp_s7799_n20-50-100_x66667/` corpus. The unsuffixed configs are templates for dataset runs
created under `data/raw/cvrp/`; update their dataset paths before using them.

| Stage | Command |
|-------|---------|
| 1. GAT | `python scripts/pretrain_gat_encoder.py --config configs/train/gat_pretrain_s7799.yaml` |
| 2. Denoiser | set `model.gat_checkpoint` in a copy of `configs/train/diffusion_denoiser_s7799.yaml`, then `python -m vrp_diffusion_quantum.train.train_diffusion --config …` |
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
  --seed 42 \
  --materialization hardlink
```

Hard-linked splits occupy almost no additional data space but must be treated as read-only because
the source and split entries refer to the same file content. Use `--materialization copy` when the
source and output are on different filesystems or independent writable files are required.

Audit split counts, overlap, manifest metadata, and logical size before training. Add
`--verify-hashes` for the slower full-content verification:

```bash
python scripts/audit_dataset.py \
  --splits cvrp_s7799_n20-50-100_x66667/splits
```

Create a deterministic, storage-efficient subset for learning curves or label audits:

```bash
python scripts/select_dataset_subset.py \
  --source cvrp_s7799_n20-50-100_x66667/splits/train \
  --output data/processed/label_audit_s7799 \
  --sizes 20 50 100 --per-size 500 --seed 4201
```

Generate the separate P3.6 shifted-demand source dataset from its committed config, then label
and convert it through the dataset app before splitting/evaluating it:

```bash
python -m vrp_diffusion_quantum.data.generate_cvrp \
  --config configs/data/cvrp_shifted_demands.yaml
```

Generate the independent plain-CVRP spatial stress sets (1,000 examples per size and distribution):

```bash
python -m vrp_diffusion_quantum.data.generate_cvrp --config configs/data/cvrp_spatial_r.yaml
python -m vrp_diffusion_quantum.data.generate_cvrp --config configs/data/cvrp_spatial_c.yaml
python -m vrp_diffusion_quantum.data.generate_cvrp --config configs/data/cvrp_spatial_rc.yaml
```

Here R means random, C clustered, and RC random/clustered. These names describe only the spatial
stress regimes: they are plain CVRP data, not genuine Solomon CVRPTW instances with time windows.

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
  --val-dir cvrp_s7799_n20-50-100_x66667/splits/val \
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

### RTX 3060 Ti training tonight

On the desktop, create a fresh virtual environment, install the CUDA-enabled PyTorch build using
the current command from the [official PyTorch selector](https://docs.pytorch.org/get-started/locally/),
then install this project. Do not copy the Mac `.venv`. Copy the gitignored
`cvrp_s7799_n20-50-100_x66667/` corpus beside the README.

After activating that environment, the complete preflight, corpus audit, deterministic subset
creation, GAT pretrain, and diffusion run is one command:

```bash
bash scripts/run_3060ti_training.sh
```

The committed recipe uses AMP, gradient clipping, and gradient accumulation with a conservative
per-step batch size. It budgets 30 minutes for GAT and 2.5 hours for diffusion. Runtime limits are
checked after a completed epoch, so expect roughly 3–4 hours depending on epoch duration. Live logs
are written under `outputs/`, and each stage writes timestamped configs, metrics, environment data,
and checkpoints under `outputs/train/`.

If the diffusion stage is interrupted after it has written `last.pt`, resume the newest run with:

```bash
bash scripts/run_3060ti_training.sh --resume-latest
```

For a short GPU verification before the long run, use `python scripts/check_cuda.py
--require-cuda` and `pytest -m cuda`.

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
