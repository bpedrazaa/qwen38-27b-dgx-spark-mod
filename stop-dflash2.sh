#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-dflash2}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "Stopped $CONTAINER_NAME"
