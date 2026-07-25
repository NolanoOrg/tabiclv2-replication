#!/bin/bash
# Watchdog v2 for the Stage-1 trainer: restart-with-resume PLUS health auto-halt.
# Halts (and STOPS restarting) if: loss ema > 0.9 past warmup, multiplier > 30
# WITH ema > 0.65 (healthy plain-WD training visits mult ~21-23 at peak LR —
# wdtest.tsv; mult alone no longer halts), or 'loss nan' appears.
# Usage: nohup bash watchdog_trainer.sh <steps> <ckpt_dir> <gen_host:port> &
# Paths are resolved relative to this script; override with env vars:
#   REPO_DIR (repo root, for the venv and module paths), RUN_DIR (run outputs).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(dirname "$HERE")}
RUN_DIR=${RUN_DIR:-$HERE}

STEPS=${1:-500000}; CKPT=${2:-$RUN_DIR/ckpt_stage1}; GEN=${3:-10.0.0.7:29700}
cd "$REPO_DIR"
[ -f "$REPO_DIR/.venv/bin/activate" ] && source "$REPO_DIR/.venv/bin/activate"
LOG=$RUN_DIR/logs/run_stage1_full.log
HALT=$RUN_DIR/HALT
WLOG=/tmp/watchdog_trainer.log

health_check() {
  LINE=$(grep '^step' "$LOG" 2>/dev/null | tail -1)
  [ -z "$LINE" ] && return 0
  STEP=$(echo "$LINE" | awk '{print $2}')
  EMA=$(echo "$LINE" | grep -oP 'ema \K[0-9.n]+')
  MULT=$(echo "$LINE" | grep -oP 'mult \K[0-9.]+')
  if echo "$LINE" | grep -q 'loss nan'; then echo "nan"; return 1; fi
  if [ -n "$EMA" ] && [ "$EMA" != "nan" ] && [ "${STEP:-0}" -gt 12000 ]; then
    awk -v e="$EMA" 'BEGIN{exit !(e>0.9)}' && { echo "ema=$EMA"; return 1; }
  fi
  if [ -n "$MULT" ] && [ -n "$EMA" ] && [ "$EMA" != "nan" ]; then
    awk -v m="$MULT" -v e="$EMA" 'BEGIN{exit !(m>30 && e>0.65)}' \
      && { echo "mult=$MULT ema=$EMA"; return 1; }
  fi
  return 0
}

while true; do
  if [ -f "$HALT" ]; then
    echo "[watchdog $(date -u +%FT%T)] HALT file present — not restarting" >> $WLOG
    sleep 300; continue
  fi
  if grep -q "^Done" "$LOG" 2>/dev/null; then
    echo "[watchdog $(date -u +%FT%T)] run complete — exiting" >> $WLOG
    exit 0
  fi
  REASON=$(health_check) || {
    echo "[watchdog $(date -u +%FT%T)] HEALTH HALT: $REASON — killing trainer, writing HALT" >> $WLOG
    echo "$(date -u +%FT%T) $REASON" > "$HALT"
    ps -eo pid,cmd | grep "[s]tage1_train.py --steps $STEPS" | awk '{print $1}' | xargs -r kill -9
    continue
  }
  if ! pgrep -f "stage1_train.py --steps $STEPS" >/dev/null; then
    echo "[watchdog $(date -u +%FT%T)] trainer down — restarting with --resume auto" >> $WLOG
    PYTHONHASHSEED=0 nohup python -u pretraining/stage1_train.py \
      --steps $STEPS --log_every 100 --warmup 10000 \
      --wd_mode plain --weight_decay 0.1 \
      --compile_blocks 1 --fp32_col_attn 1 --remote_gen $GEN \
      --checkpoint_dir $CKPT --save_every 5000 --resume auto >> "$LOG" 2>&1 &
  fi
  sleep 60
done
