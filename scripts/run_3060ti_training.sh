#!/usr/bin/env bash
# Preflight, prepare deterministic subsets, and run the bounded 3060 Ti training recipe.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VRP_PYTHON_BIN="${VRP_PYTHON_BIN:-python}"
SOURCE_SPLITS="cvrp_s7799_n20-50-100_x66667/splits"
SUBSET_ROOT="data/processed/s7799_1k_per_size"
MODE="${1:-run}"

if [[ "$MODE" != "run" && "$MODE" != "--resume-latest" ]]; then
  echo "Usage: bash scripts/run_3060ti_training.sh [--resume-latest]" >&2
  exit 2
fi

mkdir -p outputs
"$VRP_PYTHON_BIN" scripts/check_cuda.py \
  --require-cuda --output outputs/cuda_environment.json

if [[ "$MODE" == "--resume-latest" ]]; then
  LATEST_CHECKPOINT=""
  for candidate in outputs/train/diffusion_denoiser_s7799_1k_cuda_*/checkpoints/last.pt; do
    [[ -f "$candidate" ]] || continue
    if [[ -z "$LATEST_CHECKPOINT" || "$candidate" -nt "$LATEST_CHECKPOINT" ]]; then
      LATEST_CHECKPOINT="$candidate"
    fi
  done
  if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "ERROR: no diffusion last.pt checkpoint found to resume." >&2
    exit 1
  fi
  RUN_DIR="$(cd "$(dirname "$LATEST_CHECKPOINT")/.." && pwd)"
  echo "Resuming checkpoint: $LATEST_CHECKPOINT"
  "$VRP_PYTHON_BIN" -m vrp_diffusion_quantum.train.train_diffusion \
    --config "$RUN_DIR/config.yaml" --resume "$LATEST_CHECKPOINT"
  exit 0
fi

if [[ ! -d "$SOURCE_SPLITS/train" || ! -d "$SOURCE_SPLITS/val" ]]; then
  echo "ERROR: missing $SOURCE_SPLITS." >&2
  echo "Copy the gitignored s7799 corpus to the repository first." >&2
  exit 1
fi

"$VRP_PYTHON_BIN" scripts/audit_dataset.py \
  --splits "$SOURCE_SPLITS" --json-output outputs/s7799_data_audit.json

if [[ ! -f "$SUBSET_ROOT/train/subset_manifest.json" ]]; then
  "$VRP_PYTHON_BIN" scripts/select_dataset_subset.py \
    --source "$SOURCE_SPLITS/train" \
    --output "$SUBSET_ROOT/train" \
    --sizes 20 50 100 --per-size 1000 --seed 4210
fi

if [[ ! -f "$SUBSET_ROOT/val/subset_manifest.json" ]]; then
  "$VRP_PYTHON_BIN" scripts/select_dataset_subset.py \
    --source "$SOURCE_SPLITS/val" \
    --output "$SUBSET_ROOT/val" \
    --sizes 20 50 100 --per-size 250 --seed 4211
fi

export VRP_PYTHON_BIN
bash scripts/run_gat_then_diffusion.sh \
  configs/train/gat_pretrain_s7799_1k_cuda.yaml \
  configs/train/diffusion_denoiser_s7799_1k_cuda.yaml
