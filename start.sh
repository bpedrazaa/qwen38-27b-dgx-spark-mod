#!/usr/bin/env bash
# Start unsloth/Qwen3.8-27B-NVFP4 on NVIDIA DGX Spark / GB10 via vLLM.
# Dense 27B NVFP4 quantization — compressed-tensors (modelopt).
set -euo pipefail

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}"
IMAGE="${IMAGE:-eugr/spark-vllm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-27b-nvfp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-1}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-}"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${HF_HOME:-${WORK_DIR}/.cache/huggingface}"
PID_FILE="${WORK_DIR}/.vllm.pid"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_HOME}" "${WORK_DIR}/.runtime"

if docker ps --format "{{.Names}}" | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} is already running."
  curl -sS "${READY_URL}" | head -c 400 || true
  echo
  exit 0
fi

if docker ps -a --format "{{.Names}}" | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing stopped container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

hf_cache_repo_dir() {
  echo "${HF_HOME}/hub/models--${1//\//--}"
}

model_is_cached() {
  local cache_dir snapshot
  cache_dir="$(hf_cache_repo_dir "${1}")"
  [[ -d "${cache_dir}/snapshots" ]] || return 1
  for snapshot in "${cache_dir}"/snapshots/*/; do
    [[ -d "${snapshot}" ]] || continue
    [[ -f "${snapshot}/config.json" ]] || continue
    if [[ -f "${snapshot}/model.safetensors" ]] \
      || [[ -f "${snapshot}/model.safetensors.index.json" ]] \
      || compgen -G "${snapshot}/"*.safetensors >/dev/null; then
      return 0
    fi
  done
  return 1
}

download_model() {
  local model_id="$1"
  echo "Downloading ${model_id} into ${HF_HOME}"
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" HF_TOKEN="${HF_TOKEN:-}" \
      hf download "${model_id}" ${HF_TOKEN:+--token "${HF_TOKEN}"}
    return
  fi
  docker run --rm \
    --entrypoint python3 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    "${IMAGE}" \
    -c "import os; from huggingface_hub import snapshot_download; snapshot_download('${model_id}', token=os.environ.get('HF_TOKEN') or None)"
}

MODEL_MOUNT=""
MODEL_ARG="${MODEL_ID}"
if [[ -n "${LOCAL_MODEL_DIR}" && -d "${LOCAL_MODEL_DIR}" ]]; then
  MODEL_MOUNT="-v ${LOCAL_MODEL_DIR}:/models/qwen38-27b-nvfp4:ro"
  MODEL_ARG="/models/qwen38-27b-nvfp4"
elif [[ -d "${HOME}/llm/qwen38-27b-nvfp4" && -f "${HOME}/llm/qwen38-27b-nvfp4/config.json" ]]; then
  MODEL_MOUNT="-v ${HOME}/llm/qwen38-27b-nvfp4:/models/qwen38-27b-nvfp4:ro"
  MODEL_ARG="/models/qwen38-27b-nvfp4"
elif ! model_is_cached "${MODEL_ID}"; then
  download_model "${MODEL_ID}"
else
  echo "Model cache found for ${MODEL_ID}"
fi

LANG_FLAG=""
[[ "${LANGUAGE_ONLY}" == "1" ]] && LANG_FLAG="--language-model-only"

cat > "${WORK_DIR}/.runtime/serve.sh" <<EOF
#!/bin/bash
set -euo pipefail
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_CACHE_ROOT=/tmp/vllm_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
mkdir -p /tmp/vllm_cache /tmp/triton_cache /tmp/torchinductor_cache
exec vllm serve ${MODEL_ARG} \\
  --served-model-name ${MODEL_ID} \\
  --host ${HOST} \\
  --port ${PORT} \\
  ${LANG_FLAG} \\
  --tensor-parallel-size 1 \\
  --trust-remote-code \\
  --quantization compressed-tensors \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --max-num-seqs ${MAX_NUM_SEQS} \\
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \\
  --gpu-memory-utilization ${GPU_MEM_UTIL} \\
  --enable-chunked-prefill \\
  --enable-prefix-caching \\
  --reasoning-parser qwen3 \\
  --tool-call-parser qwen3_coder \\
  --enable-auto-tool-choice
EOF
chmod +x "${WORK_DIR}/.runtime/serve.sh"

echo "Pulling image ${IMAGE} (if needed)"
docker pull "${IMAGE}" >/dev/null || true

echo "Starting ${CONTAINER_NAME} on port ${PORT} (max_num_seqs=${MAX_NUM_SEQS})"
# shellcheck disable=SC2086
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --network host \
  --ipc=host \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e HF_HOME=/cache/huggingface \
  -v "${HF_HOME}:/cache/huggingface" \
  -v "${WORK_DIR}/.runtime:/runtime:ro" \
  ${MODEL_MOUNT} \
  "${IMAGE}" \
  bash /runtime/serve.sh >/dev/null

docker inspect -f "{{.State.Pid}}" "${CONTAINER_NAME}" > "${PID_FILE}" || true

echo "Waiting for ${READY_URL}"
for i in $(seq 1 120); do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    echo "Ready after ~$((i * 5))s"
    curl -sS "${READY_URL}"
    echo
    echo "Logs: docker logs -f ${CONTAINER_NAME}"
    echo "Stop:  ./stop.sh"
    exit 0
  fi
  if ! docker ps --format "{{.Names}}" | grep -qx "${CONTAINER_NAME}"; then
    echo "Container exited during startup. Last logs:"
    docker logs --tail 80 "${CONTAINER_NAME}" || true
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for API. Last logs:"
docker logs --tail 80 "${CONTAINER_NAME}" || true
exit 1
