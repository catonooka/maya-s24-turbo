#!/bin/bash
# Item 2: DSpark coverage sweep (fires after item-1 chain completes; GPU must be free)
set -e; cd ~/maya-s24-turbo/dsv41-3090/deepseek-campaign-1024
run_point () {
  local TAG=$1 POOL=$2; shift 2
  sudo systemctl stop maya-ds4 2>/dev/null || true
  sudo systemctl reset-failed maya-ds4 2>/dev/null || true
  sleep 3
  sudo systemd-run --unit=maya-ds4 --working-directory=/home/aisever/maya-ds4-main "$@" bash serve-mixed.sh
  python3 sweep_decode.py "$TAG" "$POOL"
}
run_point cum75 12 env MAYA_DS4_DSPARK_CUM=0.75
run_point cum90 13 env MAYA_DS4_DSPARK_CUM=0.9
echo DSPARK-SWEEP-DONE
