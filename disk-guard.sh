#!/bin/bash
# disk-guard.sh — STRICT pre-download check. Usage: disk-guard.sh <GB-needed> [label]
# Rule: a model download/pack/import only starts if projected free-after >= 150 GB.
# ~/cores is the known trap (285 GB of stale crash dumps once ate 30% of the disk).
NEED_GB="${1:-0}"
LABEL="${2:-download}"
FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
AFTER_GB=$((FREE_GB - NEED_GB))
CORES_GB=0
[ -d "$HOME/cores" ] && CORES_GB=$(du -sBG "$HOME/cores" 2>/dev/null | tr -dc '0-9')
echo "disk-guard: free=${FREE_GB}G need=${NEED_GB}G ($LABEL) -> free-after=${AFTER_GB}G (floor 150G)"
if [ "$CORES_GB" -ge 5 ]; then
  echo "disk-guard: WARNING ~/cores holds ${CORES_GB}G of crash dumps — review and rm (stale maya cores are post-mortem artifacts, safe once diagnosed)"
fi
if [ "$AFTER_GB" -lt 150 ]; then
  echo "disk-guard: REFUSED — free-after ${AFTER_GB}G is below the 150G floor. Clean up first (~/cores, stale packs, hf .cache blobs)."
  exit 1
fi
echo "disk-guard: OK"
exit 0
