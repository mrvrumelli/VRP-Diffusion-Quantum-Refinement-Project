#!/usr/bin/env bash
# Queue after reprioritising: Q4 selection ablation (dev, val), then the remaining grid arms.
set -u
cd "$(dirname "$0")/../.."
P6=outputs/phase6_20261001
mkdir -p $P6/selection
for SET in development:frozen_panel validation:frozen_validation; do
  NAME=${SET%%:*}; SRC=outputs/baseline_freeze_20261001/${SET##*:}
  .venv/Scripts/python.exe scripts/run_selection_ablation.py --solutions $SRC --output $P6/selection/$NAME \
    --solvers classical sa_qubo --seeds 0 1 2 --rounds 30 --workers 10 > $P6/selection/$NAME.log 2>&1 \
    && echo "DONE selection_$NAME" || echo "FAILED selection_$NAME"
done
OUT=$P6/extension
run_arm () {
  NAME=$1; SRC=$2; TAG=$3; R=$4; E=$5; P=$6
  .venv/Scripts/python.exe scripts/run_refinement_loop.py --solutions $SRC --output $OUT/${NAME}_${TAG} \
    --rounds 30 --max-reorder-size $R --max-exchange-size $E --max-per-type $P \
    --classical-restarts 10 --configs classical classical_restarts sa_qubo --workers 10 \
    >> $OUT/${NAME}_${TAG}.log 2>&1 && echo "DONE ${NAME}_${TAG}" || echo "FAILED ${NAME}_${TAG}"
}
run_arm development outputs/baseline_freeze_20261001/frozen_panel default_more 5 8 30
run_arm development outputs/baseline_freeze_20261001/frozen_panel large_more 10 16 30
for ARM in default:5:8:10 medium:7:12:10 large:10:16:10 default_more:5:8:30 large_more:10:16:30; do
  IFS=: read -r TAG R E P <<< "$ARM"
  run_arm validation outputs/baseline_freeze_20261001/frozen_validation $TAG $R $E $P
done
echo QUEUE2_DONE
