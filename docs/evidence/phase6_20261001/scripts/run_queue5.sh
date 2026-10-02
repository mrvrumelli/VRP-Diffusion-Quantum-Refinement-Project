#!/usr/bin/env bash
# Gap-filling queue (after queue4): QAOA in the loop, per-selector ablation, bias sweep, seeds,
# time-matched Q5 control, matched experiment with SQA.
set -u
cd "$(dirname "$0")/../.."
until grep -q QUEUE4_DONE "$1"; do sleep 30; done
P6=outputs/phase6_20261001
OUT=$P6/extension
DEV=outputs/baseline_freeze_20261001/frozen_panel
VAL=outputs/baseline_freeze_20261001/frozen_validation
PY=.venv/Scripts/python.exe
loop () {  # name src tag extra-args...
  NAME=$1; SRC=$2; TAG=$3; shift 3
  $PY scripts/run_refinement_loop.py --solutions $SRC --output $OUT/${NAME}_${TAG} --rounds 30 --workers 10 "$@" \
    >> $OUT/${NAME}_${TAG}.log 2>&1 && echo "DONE ${NAME}_${TAG}" || echo "FAILED ${NAME}_${TAG}"
}
QAOA_ARGS="--max-reorder-size 4 --max-exchange-size 6 --classical-restarts 10 --configs classical classical_restarts sa_qubo qaoa random_qubo qaoa+classical --qaoa-reps 1 --qaoa-restarts 4 --qaoa-shots 1024 --qaoa-keep 50 --max-qubits 16"
loop development $DEV qaoa_small $QAOA_ARGS
mkdir -p $P6/selection
for SET in development:$DEV validation:$VAL; do
  NAME=${SET%%:*}; SRC=${SET##*:}
  for TYPE in uncertain_m low_confidence_edges; do
    $PY scripts/run_selection_ablation.py --solutions $SRC --output $P6/selection/${NAME}_${TYPE} \
      --solvers classical sa_qubo --seeds 0 1 2 --rounds 30 --workers 10 \
      --diffusion-types $TYPE --groups random_any random_adjacent > $P6/selection/${NAME}_${TYPE}.log 2>&1 \
      && echo "DONE selection_${NAME}_${TYPE}" || echo "FAILED selection_${NAME}_${TYPE}"
  done
done
mkdir -p $P6/bias_sweep
for NAME in development validation; do
  $PY scripts/run_bias_sweep.py --set $P6/neighborhoods_${NAME}.json --output $P6/bias_sweep/$NAME --workers 10 \
    > $P6/bias_sweep/$NAME.log 2>&1 && echo "DONE bias_$NAME" || echo "FAILED bias_$NAME"
done
loop validation $VAL qaoa_small $QAOA_ARGS
for SEED in 1 2; do
  loop development $DEV default_seed$SEED --seed $SEED --classical-restarts 10 --configs classical_restarts sa_qubo
  loop validation $VAL default_seed$SEED --seed $SEED --classical-restarts 10 --configs classical_restarts sa_qubo
done
loop development $DEV default_restarts20 --classical-restarts 20 --configs classical_restarts
loop validation $VAL default_restarts20 --classical-restarts 20 --configs classical_restarts
for NAME in development validation; do
  $PY scripts/run_refinement_experiment.py --set $P6/neighborhoods_${NAME}.json --output $P6/experiment_${NAME}_sqa \
    --budget 0.25 --sweeps 100 --alpha 0.5 --seed 0 --workers 8 > $P6/experiment_${NAME}_sqa.log 2>&1 \
    && echo "DONE sqa_experiment_$NAME" || echo "FAILED sqa_experiment_$NAME"
done
echo QUEUE5_DONE
