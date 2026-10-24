#!/bin/bash
# Flip DS4 between decode-shape (TP1) and prefill-shape (TP2). Usage: sudo bash flip-tp.sh 1|2
set -e
TP="$1"; [ "$TP" = "1" ] || [ "$TP" = "2" ] || { echo "usage: sudo bash flip-tp.sh 1|2"; exit 1; }
sudo systemctl stop maya-ds4 2>/dev/null || true
sleep 5
python3 - <<EOF
import json
p="/home/aisever/maya-ds4-main/maya-ds4.json"
c=json.load(open(p))
i=c["args"].index("--tp"); c["args"][i+1]="$TP"
json.dump(c,open(p,"w"),indent=1)
print("config now --tp $TP")
EOF
sudo systemctl reset-failed maya-ds4 2>/dev/null || true
sudo systemd-run --uid=1000 --gid=1000 --unit=maya-ds4 --working-directory=/home/aisever/maya-ds4-main bash serve-v1124-baseline.sh
echo "flipped to TP$TP - engine boots in ~4-6 min"
