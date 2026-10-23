#!/bin/sh
# starts the Project Maya dashboard and API (written by maya.py; ./maya.sh does the same)
cd /home/aisever/project-maya || exit 1
exec /home/aisever/project-maya/.venv/bin/python /home/aisever/project-maya/serve/server.py --engine strata --config /home/aisever/project-maya/maya-maya-s-v2-iq2_xxs.json --port 8080 "$@"
