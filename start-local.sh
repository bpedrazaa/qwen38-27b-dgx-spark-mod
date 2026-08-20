#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-$HOME/llm/qwen38-27b-nvfp4}"
MODEL_ID="${MODEL_ID:-Qwen3.8-27B}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-7}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
ENABLE_MTP="${ENABLE_MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":4,\"video\":1}}"
MEDIA_IO_KWARGS="${MEDIA_IO_KWARGS:-{\"video\":{\"num_frames\":-1}}}"
VLLM_BIN="${VLLM_BIN:-$HOME/venvs/vllm-026/bin/vllm}"
LOG="${LOG:-$HOME/qwen38-vllm.log}"
PIDFILE="${PIDFILE:-$HOME/qwen38-vllm.pid}"

[ -x "$VLLM_BIN" ] || { echo "Missing $VLLM_BIN"; exit 1; }
[ -f "$MODEL_DIR/model.safetensors" ] || { echo "Missing weights"; exit 1; }
[ "$(stat -c%s "$MODEL_DIR/model.safetensors")" = "22568192096" ] || { echo "Weights incomplete: $(stat -c%s "$MODEL_DIR/model.safetensors") / 22568192096 bytes"; exit 1; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running PID $(cat "$PIDFILE")"
  exit 0
fi

export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_CACHE_ROOT="$HOME/.cache/vllm-qwen38"
export TRITON_CACHE_DIR="$HOME/.cache/triton-qwen38"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor-qwen38"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

SPEC_ARGS=()
if [ "${ENABLE_MTP}" = "1" ]; then
  SPEC_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
fi

nohup "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$MODEL_ID" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
  --media-io-kwargs "$MEDIA_IO_KWARGS" \
  "${SPEC_ARGS[@]}" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "PID $(cat "$PIDFILE"), log $LOG, ctx=${MAX_MODEL_LEN}, vision=on, MTP=${ENABLE_MTP} k=${NUM_SPECULATIVE_TOKENS}"
