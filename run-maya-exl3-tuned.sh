#!/bin/bash
# EXL3-3.05bpw tuned launcher (2x3090): env lives in config json (RAM_RESIDENT + KV_INT8 only —
# PROMOTE_MIN/SPEC vars measured HARMFUL on EXL3: promotions must churn freely, see notes)
cd /home/aisever/project-maya
exec ./run-maya-exl3-3.05bpw.sh
