#!/bin/bash
# S24 launcher: stops EXL3 unit if up, boots S24 config given as $1 (default mtp) in system slice
set -e
CFG="${1:-exl3.json}"
NAME="glm3-$(basename "$CFG" .json)"
sudo systemctl stop "$NAME.service" glm3.service 'glm3-exl3-*.service' 'glm3-s24-*.service' 2>/dev/null || true
sudo systemctl reset-failed "$NAME.service" 2>/dev/null || true
cd /home/aisever/maya-glm3
sudo systemd-run --uid=1000 --gid=1000 --unit="$NAME" --slice=system.slice -p LimitMEMLOCK=infinity -p WorkingDirectory=/home/aisever/maya-glm3 \
  /home/aisever/project-maya/.venv/bin/python /home/aisever/maya-glm3/maya.py --config /home/aisever/maya-glm3/"$CFG" --port 8080
