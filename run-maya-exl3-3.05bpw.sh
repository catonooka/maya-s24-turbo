#!/bin/sh
# starts the Project Maya dashboard and API (written by maya.py; ./maya.sh does the same)
cd /home/aisever/project-maya || exit 1
exec /home/aisever/project-maya/.venv/bin/python /home/aisever/project-maya/maya.py --config /home/aisever/project-maya/maya-exl3-3.05bpw.json --port 8080 "$@"
