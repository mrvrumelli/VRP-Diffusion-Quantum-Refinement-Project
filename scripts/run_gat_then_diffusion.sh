#!/usr/bin/env bash
# GAT pretrain → diffusion denoise (frozen GAT).
#
#   bash scripts/run_gat_then_diffusion.sh
#   bash scripts/run_gat_then_diffusion.sh configs/train/gat_pretrain.yaml configs/train/diffusion_denoiser.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs
export PYTHONUNBUFFERED=1
VRP_PYTHON_BIN="${VRP_PYTHON_BIN:-python}"

GAT_CFG="${1:-configs/train/gat_pretrain.yaml}"
DIFF_CFG="${2:-configs/train/diffusion_denoiser.yaml}"
GAT_EXP="$(basename "$GAT_CFG" .yaml)"
DIFF_EXP="$(basename "$DIFF_CFG" .yaml)"
GAT_LOG="outputs/${GAT_EXP}_live.log"
DIFF_LOG="outputs/${DIFF_EXP}_live.log"
TMP_CFG="outputs/${DIFF_EXP}_runtime.yaml"

echo "[1/2] GAT pretrain ($GAT_CFG)..."
"$VRP_PYTHON_BIN" scripts/pretrain_gat_encoder.py --config "$GAT_CFG" \
  2>&1 | tee "$GAT_LOG"

GAT_CKPT=""
for candidate in outputs/train/${GAT_EXP}_*/checkpoints/gat_encoder_best.pt; do
  [[ -f "$candidate" ]] || continue
  if [[ -z "$GAT_CKPT" || "$candidate" -nt "$GAT_CKPT" ]]; then
    GAT_CKPT="$candidate"
  fi
done
if [[ -z "$GAT_CKPT" ]]; then
  echo "ERROR: gat_encoder_best.pt not found under outputs/train/${GAT_EXP}_*" >&2
  exit 1
fi
echo "Using GAT checkpoint: $GAT_CKPT"

"$VRP_PYTHON_BIN" - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("$DIFF_CFG").read_text())
cfg["model"]["gat_checkpoint"] = "$GAT_CKPT"
out = Path("$TMP_CFG")
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", out)
PY

echo "[2/2] Diffusion denoise (frozen GAT) ($TMP_CFG)..."
"$VRP_PYTHON_BIN" -m vrp_diffusion_quantum.train.train_diffusion --config "$TMP_CFG" \
  2>&1 | tee "$DIFF_LOG"

echo "Pipeline finished."
