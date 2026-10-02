#!/usr/bin/env bash
# Final test (protocol: docs/phase6_final_test_protocol_2026-10-02.md). Scored once (--once).
set -u
cd "$(dirname "$0")/../.."
until grep -q QUEUE5_DONE "$1"; do sleep 30; done
P6=outputs/phase6_20261001
OUT=$P6/final
mkdir -p $OUT
PY=.venv/Scripts/python.exe
run () {  # set-name src tag extra-args...
  NAME=$1; SRC=$2; TAG=$3; shift 3
  $PY scripts/run_refinement_loop.py --solutions $SRC --output $OUT/${NAME}_${TAG} --rounds 30 --workers 10 --once "$@" \
    > $OUT/${NAME}_${TAG}.log 2>&1 && echo "DONE final_${NAME}_${TAG}" || echo "FAILED final_${NAME}_${TAG}"
}
for SET in reserved:outputs/baseline_freeze_20261001/reserved_test_scored ood:outputs/baseline_freeze_20261001/frozen_ood; do
  NAME=${SET%%:*}; SRC=${SET##*:}
  run $NAME $SRC t1 --configs classical
  run $NAME $SRC t2_t4_t5 --max-per-type 30 --classical-restarts 10 --configs classical_restarts sa_qubo sa_qubo+classical_restarts
  run $NAME $SRC t3 --max-reorder-size 10 --max-exchange-size 16 --max-per-type 30 --classical-restarts 10 --configs classical_restarts
  run $NAME $SRC t6_t9 --max-reorder-size 4 --max-exchange-size 6 --classical-restarts 10 \
    --configs classical_restarts classical_exact qaoa random_qubo qaoa+classical_restarts --qaoa-reps 1 --max-qubits 16
done
echo QUEUE6_DONE
