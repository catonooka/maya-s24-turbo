#!/bin/bash
# Maya launcher v1.0.24 tuned: RAM-resident, INT8 KV, promote 6, draft-keep 2
export STRATA_GLM_RAM_RESIDENT=1
export STRATA_GLM_PROMOTE_MIN=6
export STRATA_GLM_KV_INT8=1
export STRATA_GLM_MTP_KEEP=2
cd /home/aisever/project-maya
exec ./run-maya-maya-s-v2-iq2_xxs.sh
