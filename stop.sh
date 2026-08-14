#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-27b-nvfp4}"
if docker ps -a --format "{{.Names}}" | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}"
  echo "Stopped ${CONTAINER_NAME}"
else
  echo "No container named ${CONTAINER_NAME}"
fi
