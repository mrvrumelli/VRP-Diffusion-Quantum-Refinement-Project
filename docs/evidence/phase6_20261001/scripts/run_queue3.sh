#!/usr/bin/env bash
# Q5: classical polish on top of converged SA solutions (default neighborhoods), dev and val.
set -u
cd "$(dirname "$0")/../.."
P6=outputs/phase6_20261001
until grep -q QUEUE2_DONE "$1"; do sleep 30; done
for SET in development:frozen_panel validation:frozen_validation; do
  NAME=${SET%%:*}; SRC=outputs/baseline_freeze_20261001/${SET##*:}
  START=$P6/extension/${NAME}_start_after_sa
  .venv/Scripts/python.exe $P6/build_polish_start.py $P6/extension/${NAME}_default/rows.jsonl sa_qubo $SRC $START
  .venv/Scripts/python.exe scripts/run_refinement_loop.py --solutions $START --output $P6/extension/${NAME}_sa_then_classical \
    --rounds 30 --classical-restarts 10 --configs classical classical_restarts --workers 10 \
    > $P6/extension/${NAME}_sa_then_classical.log 2>&1 && echo "DONE polish_$NAME" || echo "FAILED polish_$NAME"
done
echo QUEUE3_DONE
