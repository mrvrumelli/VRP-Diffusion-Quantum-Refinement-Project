#!/usr/bin/env bash
# Launch scripts/run_data_pipeline.py in the background and keep the Mac awake while it runs,
# so a large generate/solve/export run can be left going overnight unattended.
#
# It's safe to just close this over: the pipeline checkpoints after every solved instance, so
# if the run does get interrupted, rerun scripts/run_data_pipeline.py with the same --run-name
# and it resumes instead of starting over (see that script's docstring).
#
# Note: `caffeinate` blocks idle/system sleep but cannot override a closed laptop lid unless the
# Mac is on power with an external display attached (clamshell mode). On battery, or lid closed
# with no external display, the OS will still sleep. Safest bet: power connected, lid open (or
# clamshell mode set up), display sleep is fine -- only idle/system sleep needs blocking here.
#
# Usage:
#   scripts/run_overnight.sh [run_data_pipeline.py args...]
#   scripts/run_overnight.sh --run-name big_run --num 20000 --workers 4
#
# Check on it:
#   tail -f <printed log file>
# Stop it:
#   kill <printed pid>

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

OUT_DIR="$ROOT_DIR/data/raw/cvrp"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/.overnight_${STAMP}.out"
PID_FILE="$OUT_DIR/.overnight_${STAMP}.pid"

if command -v caffeinate >/dev/null 2>&1; then
  RUNNER=(caffeinate -ims)
else
  echo "warning: caffeinate not found (not macOS?) -- system sleep may interrupt the run" >&2
  RUNNER=()
fi

nohup "${RUNNER[@]}" "$PYTHON" "$ROOT_DIR/scripts/run_data_pipeline.py" "$@" \
  > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "started pid=$PID"
echo "early stdout/stderr: $LOG_FILE"
echo "once it starts, per-run progress is also in <run_dir>/run.log (run_dir is logged near the top of $LOG_FILE)"
echo "check progress:  tail -f $LOG_FILE"
echo "stop it:         kill $PID"
