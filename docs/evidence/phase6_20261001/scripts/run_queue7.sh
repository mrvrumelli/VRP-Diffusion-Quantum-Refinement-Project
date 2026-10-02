#!/usr/bin/env bash
# Clean timing pass after the final test: 12-graph subset, one worker per run, numeric libraries
# single-threaded, at most four runs at once on six physical cores.
set -u
cd "$(dirname "$0")/../.."
until grep -q QUEUE6_DONE "$1"; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P6=outputs/phase6_20261001
SUB=$P6/timing_subset
OUT=$P6/timing
mkdir -p $OUT
PY=.venv/Scripts/python.exe
t () {  # tag extra-args...
  TAG=$1; shift
  $PY scripts/run_refinement_loop.py --solutions $SUB --output $OUT/$TAG --rounds 30 --workers 1 "$@" \
    > $OUT/$TAG.log 2>&1 && echo "DONE timing_$TAG" || echo "FAILED timing_$TAG"
}
( t t1_t10 --configs classical; t exact_small --max-reorder-size 4 --max-exchange-size 6 --configs classical_exact ) &
( t t3 --max-reorder-size 10 --max-exchange-size 16 --max-per-type 30 --classical-restarts 10 --configs classical_restarts ) &
( t t2_t4_t5 --max-per-type 30 --classical-restarts 10 --configs classical_restarts sa_qubo sa_qubo+classical_restarts ) &
( t t6_t7 --max-reorder-size 4 --max-exchange-size 6 --configs qaoa random_qubo --qaoa-reps 1 --max-qubits 16 ) &
wait
echo QUEUE7_DONE
