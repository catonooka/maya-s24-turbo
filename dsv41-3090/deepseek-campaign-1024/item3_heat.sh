#!/bin/bash
# Item 3: cache-heat knob sweep (fires after item 2; GPU must be free)
# Defaults under test: DECAY 0.9, BONUS 1.0, RECENT_FILLS 2 / NEED 2 / WIN 8
set -e; cd ~/maya-s24-turbo/dsv41-3090/deepseek-campaign-1024
run_point () {
  local TAG=$1 POOL=$2; shift 2
  sudo systemctl stop maya-ds4 2>/dev/null || true
  sudo systemctl reset-failed maya-ds4 2>/dev/null || true
  sleep 3
  sudo systemd-run --unit=maya-ds4 --working-directory=/home/aisever/maya-ds4-main "$@" bash serve-mixed.sh
  python3 sweep_decode.py "$TAG" "$POOL"
}
run_point decay95 14 env DS4_CACHE_DECAY=0.95
run_point decay80 15 env DS4_CACHE_DECAY=0.8
run_point bonus15 16 env DS4_CACHE_BONUS=1.5
run_point recent16 17 env DS4_RECENT_WIN=16 DS4_RECENT_FILLS=4
echo HEAT-SWEEP-DONE
