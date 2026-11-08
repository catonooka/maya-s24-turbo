#!/bin/bash
# Serve DSV41 EXL3 2.9bpw on 3090 with optional extra vllm args.
# Usage: serve-dsv41.sh [extra vllm serve args...]
docker rm -f dsv41 >/dev/null 2>&1
exec docker run -d --name dsv41 \
  --gpus '"device=0"' --shm-size 8g --memory 300g --cpus 30 \
  --cap-add IPC_LOCK --cap-add SYS_PTRACE --ulimit memlock=-1 \
  -p 127.0.0.1:8000:8000 \
  -v ~/models:/models:ro \
  -v ~/dsv41-cache:/root/.cache \
  -e DSV41_PACK=/models/DSV41-EXL3-3090-local \
  -e DSV41_CT_LAYERS=35 \
  -e DSV41_CT_LOG_EVERY=100 -e DSV41_EC_MARGIN_MB=300 \
  ghcr.io/0xsero/dsv41-flash-offload:d141-main "$@"
