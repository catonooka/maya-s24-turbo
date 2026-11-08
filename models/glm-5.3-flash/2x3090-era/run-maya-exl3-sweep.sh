#!/bin/bash
# Sweepable EXL3 launcher: env overrides from /tmp/maya-sweep.env (written by the sweep harness),
# falling back to the tuned config's own env.
cd /home/aisever/project-maya
set -a; [ -f /tmp/maya-sweep.env ] && . /tmp/maya-sweep.env; set +a
exec ./run-maya-exl3-3.05bpw.sh
