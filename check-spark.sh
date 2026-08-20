#!/usr/bin/env bash
# Pre-launch sanity check for a Spark box before running ./start.sh.
# Read-only: makes no changes, safe to run anytime.
set -uo pipefail

PORT="${PORT:-8001}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-27b-nvfp4}"
MODEL_DIR="${MODEL_DIR:-$HOME/llm/qwen38-27b-nvfp4}"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${HF_HOME:-${WORK_DIR}/.cache/huggingface}"

section() { printf '\n== %s ==\n' "$1"; }

section "Ports (8000 = common NVSM exporter collision, ${PORT} = target)"
ss -ltn | grep -E ':(8000|'"${PORT}"')\s' || echo "  nothing on 8000 or ${PORT}"

section "Existing containers"
docker ps -a

section "GPU"
nvidia-smi || echo "  nvidia-smi not available"

section "Memory"
free -h

section "Disk"
df -h /
docker info 2>/dev/null | grep -i "docker root dir" || true

section "Model weights (${MODEL_DIR})"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "  found: ${MODEL_DIR}"
  ls -la "${MODEL_DIR}"/*.safetensors 2>/dev/null || echo "  no .safetensors in dir (index/sharded layout?)"
else
  echo "  not found at ${MODEL_DIR} (start.sh will fall back to HF cache or auto-download)"
fi

section "HF cache (${HF_HOME})"
if [[ -d "${HF_HOME}" ]]; then
  du -sh "${HF_HOME}" 2>/dev/null
else
  echo "  no repo-local HF cache yet at ${HF_HOME}"
fi

section "This checkout"
echo "  path:   ${WORK_DIR}"
git -C "${WORK_DIR}" remote -v 2>/dev/null || echo "  not a git repo"
git -C "${WORK_DIR}" log -1 --oneline 2>/dev/null
git -C "${WORK_DIR}" status --porcelain 2>/dev/null | sed 's/^/  /' || true
