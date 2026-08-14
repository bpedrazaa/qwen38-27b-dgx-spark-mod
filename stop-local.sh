#!/usr/bin/env bash
set -euo pipefail
PIDFILE="${PIDFILE:-$HOME/qwen38-vllm.pid}"
PORT="${PORT:-8000}"
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping PID $PID"
    kill "$PID" || true
    for i in $(seq 1 40); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo "Force killing $PID"
      kill -9 "$PID" || true
      sleep 1
    fi
  fi
  rm -f "$PIDFILE"
fi
# catch leftover engine if pidfile was stale
PIDS="$(ss -lntp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p {print}' || true)"
pkill -f "vllm serve .*qwen38-27b-nvfp4" 2>/dev/null || true
echo "Stopped local vLLM (port ${PORT})"
