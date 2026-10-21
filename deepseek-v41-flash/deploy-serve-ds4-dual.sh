#!/bin/bash
# Serve DeepSeek-V4.1-Flash via the maya repo (full dashboard/API) on both 3090s.
# Container wrapper: ubuntu:24.04 (glibc match for maya-ds4-container) + CUDA libs + unlimited memlock.
# API on http://0.0.0.0:8080 (OpenAI-compatible), dashboard on the same port.
docker rm -f maya-serve >/dev/null 2>&1 || true
exec docker run -d --name maya-serve \
  --gpus all --shm-size 16g --memory 300g --cpus 30 \
  --cap-add IPC_LOCK --cap-add SYS_PTRACE --ulimit memlock=-1 \
  -p 0.0.0.0:8080:8080 \
  -v ~:~ \
  -v /usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-libs:ro \
  --entrypoint /bin/bash ubuntu:24.04 \
  -c 'export DEBIAN_FRONTEND=noninteractive LD_LIBRARY_PATH=/usr/local/cuda-libs:$LD_LIBRARY_PATH;
      apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq python3 python3-pip >/dev/null 2>&1;
      pip3 install --break-system-packages -q jinja2 fastapi uvicorn psutil regex requests pillow numpy >/dev/null 2>&1;
      cd ~/maya-ds4-main &&
      python3 serve/server.py --engine strata --config maya-ds4.json --port 8080 2>&1'
