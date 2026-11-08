#!/bin/sh
# starts the Project Maya dashboard and API (written by maya.py; ./maya.sh does the same)
cd /home/USER/project-maya || exit 1
exec /home/USER/project-maya/.venv/bin/python /home/USER/project-maya/maya.py --config /home/USER/project-maya/maya-exl3-3.05bpw.json --port 8080 "$@"
