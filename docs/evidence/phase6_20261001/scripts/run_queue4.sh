#!/usr/bin/env bash
# Reordered queue: finish dev default_more, Q5 polish (dev), validation default, Q5 polish (val),
# then the remaining grid arms. Every loop run resumes from rows.partial.jsonl.
set -u
cd "$(dirname "$0")/../.."
P6=outputs/phase6_20261001
OUT=$P6/extension
DEV=outputs/baseline_freeze_20261001/frozen_panel
VAL=outputs/baseline_freeze_20261001/frozen_validation
run_arm () {
  NAME=$1; SRC=$2; TAG=$3; R=$4; E=$5; P=$6
  .venv/Scripts/python.exe scripts/run_refinement_loop.py --solutions $SRC --output $OUT/${NAME}_${TAG} \
    --rounds 30 --max-reorder-size $R --max-exchange-size $E --max-per-type $P \
    --classical-restarts 10 --configs classical classical_restarts sa_qubo --workers 10 \
    >> $OUT/${NAME}_${TAG}.log 2>&1 && echo "DONE ${NAME}_${TAG}" || echo "FAILED ${NAME}_${TAG}"
}
polish () {
  NAME=$1; SRC=$2
  START=$OUT/${NAME}_start_after_sa
  .venv/Scripts/python.exe $P6/build_polish_start.py $OUT/${NAME}_default/rows.jsonl sa_qubo $SRC $START
  .venv/Scripts/python.exe scripts/run_refinement_loop.py --solutions $START --output $OUT/${NAME}_sa_then_classical \
    --rounds 30 --classical-restarts 10 --configs classical classical_restarts --workers 10 \
    >> $OUT/${NAME}_sa_then_classical.log 2>&1 && echo "DONE polish_$NAME" || echo "FAILED polish_$NAME"
}
run_arm development $DEV default_more 5 8 30
polish development $DEV
run_arm validation $VAL default 5 8 10
polish validation $VAL
run_arm development $DEV large_more 10 16 30
run_arm validation $VAL medium 7 12 10
run_arm validation $VAL large 10 16 10
run_arm validation $VAL default_more 5 8 30
run_arm validation $VAL large_more 10 16 30
echo QUEUE4_DONE
