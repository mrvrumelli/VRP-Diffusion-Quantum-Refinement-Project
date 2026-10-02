#!/usr/bin/env bash
# Refinement extension grid: more rounds (cap 30) x neighborhood size / count, dev then validation.
set -u
cd "$(dirname "$0")/../.."
OUT=outputs/phase6_20261001/extension
mkdir -p $OUT
ARMS="default:5:8:10 medium:7:12:10 large:10:16:10 default_more:5:8:30 large_more:10:16:30"
for SET in development:frozen_panel validation:frozen_validation; do
  NAME=${SET%%:*}; SRC=outputs/baseline_freeze_20261001/${SET##*:}
  for ARM in $ARMS; do
    IFS=: read -r TAG R E P <<< "$ARM"
    .venv/Scripts/python.exe scripts/run_refinement_loop.py --solutions $SRC --output $OUT/${NAME}_${TAG} \
      --rounds 30 --max-reorder-size $R --max-exchange-size $E --max-per-type $P \
      --classical-restarts 10 --configs classical classical_restarts sa_qubo --workers 10 \
      > $OUT/${NAME}_${TAG}.log 2>&1 && echo "DONE ${NAME}_${TAG}" || echo "FAILED ${NAME}_${TAG}"
  done
done
echo EXTENSION_DONE
