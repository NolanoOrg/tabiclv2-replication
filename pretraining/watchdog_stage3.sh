#!/bin/bash
# Watchdog for the Stage-3 trainer: restart-with-resume + health auto-halt.
# Halts if: loss ema > 0.9 past step 500, multiplier > 48 with ema > 0.65
# (stage-2 converged mult ~39 is the healthy baseline), or 'loss nan'.
# Paths are resolved relative to this script; override with env vars:
#   REPO_DIR (repo root, for the venv and module paths), RUN_DIR (run outputs).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(dirname "$HERE")}
RUN_DIR=${RUN_DIR:-$HERE}

STEPS=10000
CKPT=$RUN_DIR/ckpt_stage3
GEN=10.0.0.7:29900
cd "$REPO_DIR"
[ -f "$REPO_DIR/.venv/bin/activate" ] && source "$REPO_DIR/.venv/bin/activate"
LOG=$RUN_DIR/logs/run_stage3_full.log
HALT=$RUN_DIR/HALT3
WLOG=/tmp/watchdog_stage3.log

health_check() {
  LINE=$(grep '^step' "$LOG" 2>/dev/null | tail -1)
  [ -z "$LINE" ] && return 0
  STEP=$(echo "$LINE" | awk '{print $2}')
  EMA=$(echo "$LINE" | grep -oP 'ema \K[0-9.n]+')
  MULT=$(echo "$LINE" | grep -oP 'mult \K[0-9.]+')
  if echo "$LINE" | grep -q 'loss nan'; then echo "nan"; return 1; fi
  if [ -n "$EMA" ] && [ "$EMA" != "nan" ] && [ "${STEP:-0}" -gt 500 ]; then
    awk -v e="$EMA" 'BEGIN{exit !(e>0.9)}' && { echo "ema=$EMA"; return 1; }
  fi
  if [ -n "$MULT" ] && [ -n "$EMA" ] && [ "$EMA" != "nan" ]; then
    awk -v m="$MULT" -v e="$EMA" 'BEGIN{exit !(m>48 && e>0.65)}' \
      && { echo "mult=$MULT ema=$EMA"; return 1; }
  fi
  return 0
}

while true; do
  if [ -f "$HALT" ]; then
    echo "[watchdog $(date -u +%FT%T)] HALT3 present — not restarting" >> $WLOG
    sleep 300; continue
  fi
  if grep -q "^Done" "$LOG" 2>/dev/null; then
    echo "[watchdog $(date -u +%FT%T)] stage 3 complete — exiting" >> $WLOG
    exit 0
  fi
  REASON=$(health_check) || {
    echo "[watchdog $(date -u +%FT%T)] HEALTH HALT: $REASON" >> $WLOG
    echo "$(date -u +%FT%T) $REASON" > "$HALT"
    pgrep -f "python -u pretraining/stage3_[t]rain.py" | xargs -r kill -9
    continue
  }
  if ! pgrep -f "python -u pretraining/stage3_[t]rain.py" >/dev/null; then
    echo "[watchdog $(date -u +%FT%T)] trainer down — (re)starting with --resume auto" >> $WLOG
    PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python -u pretraining/stage3_train.py \
      --steps $STEPS --remote_gen $GEN --checkpoint_dir $CKPT \
      --save_every 250 --log_every 5 --resume auto >> "$LOG" 2>&1 &
  fi
  sleep 60
done
