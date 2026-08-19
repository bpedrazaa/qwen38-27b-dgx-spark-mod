#!/usr/bin/env bash
# Serve Qwen3.8-27B with Inco DFlash 2 on one DGX Spark / GB10.
set -euo pipefail

IMAGE="${IMAGE:-qwen38-dflash2:v0.27.1-aarch64}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-dflash2}"
TARGET_MODEL="${TARGET_MODEL:-joshebbs/qwen3.8-27b-uncensored-nvfp4-modelopt}"
DRAFT_MODEL="${DRAFT_MODEL:-incoai/Qwen3.8-27B-DFlash2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
TARGET_DIR="${TARGET_DIR:-}"
DRAFT_DIR="${DRAFT_DIR:-}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
mkdir -p "$HF_HOME"

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "Image $IMAGE not found. Build it first:"
  echo "  docker build -f Dockerfile.dflash2 -t $IMAGE ."
  exit 1
}

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

TARGET_ARG="$TARGET_MODEL"
DRAFT_ARG="$DRAFT_MODEL"
MOUNTS=(-v "$HF_HOME:/root/.cache/huggingface")
if [[ -n "$TARGET_DIR" ]]; then
  [[ -f "$TARGET_DIR/config.json" ]] || { echo "TARGET_DIR has no config.json: $TARGET_DIR" >&2; exit 1; }
  TARGET_ARG=/models/base
  MOUNTS+=(-v "$TARGET_DIR:/models/base:ro")
fi
if [[ -n "$DRAFT_DIR" ]]; then
  [[ -f "$DRAFT_DIR/config.json" ]] || { echo "DRAFT_DIR has no config.json: $DRAFT_DIR" >&2; exit 1; }
  DRAFT_ARG=/models/draft
  MOUNTS+=(-v "$DRAFT_DIR:/models/draft:ro")
fi

ARGS=(
  "$TARGET_ARG"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-prefix-caching
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --limit-mm-per-prompt.image 2
  --limit-mm-per-prompt.video 0
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_ARG\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"draft_sample_method\":\"probabilistic\"}"
)
if [[ -n "$CHAT_TEMPLATE" ]]; then
  [[ -f "$CHAT_TEMPLATE" ]] || { echo "CHAT_TEMPLATE not found: $CHAT_TEMPLATE" >&2; exit 1; }
  MOUNTS+=(-v "$CHAT_TEMPLATE:/models/chat_template.jinja:ro")
  ARGS+=(--chat-template /models/chat_template.jinja)
fi

echo "Starting $CONTAINER_NAME: target=$TARGET_ARG draft=$DRAFT_ARG k=$NUM_SPECULATIVE_TOKENS ctx=$MAX_MODEL_LEN"
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${MOUNTS[@]}" \
  "$IMAGE" "${ARGS[@]}" >/dev/null

for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Ready: http://127.0.0.1:$PORT/v1 (about $((i * 5))s)"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for the API" >&2
docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
exit 1
