#!/bin/bash
# Watchdog for gen_server on the CPU node. Restarts with crash recovery
# (resumes stream at the newest RNG snapshot; clients SEEK to what they need).
# Usage: nohup bash watchdog_server.sh <port> <seed> <jobs> <depth> &
PORT=${1:-29700}; SEED=${2:-42}; JOBS=${3:-60}; DEPTH=${4:-16}
cd "$(dirname "$0")"
source ../.venv/bin/activate
while true; do
  if ! pgrep -f "gen_server.py --port $PORT" >/dev/null; then
    echo "[watchdog $(date -u +%FT%T)] gen_server down — restarting" >> /tmp/watchdog_server.log
    if [ -f /tmp/gen_state.pkl ]; then
      PYTHONHASHSEED=0 nohup python -u gen_server.py --port $PORT --seed $SEED \
        --jobs $JOBS --depth $DEPTH --resume_from_statefile >> /tmp/gen_server_prod.log 2>&1 &
    else
      PYTHONHASHSEED=0 nohup python -u gen_server.py --port $PORT --seed $SEED \
        --jobs $JOBS --depth $DEPTH >> /tmp/gen_server_prod.log 2>&1 &
    fi
  fi
  sleep 30
done
