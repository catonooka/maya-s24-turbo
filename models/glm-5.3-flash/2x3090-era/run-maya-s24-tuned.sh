#!/bin/bash
# Maya-S24 launcher v1.1.5: RAM-resident, INT8 KV, promote 6, mtp-keep 2, spec-sample+cancel
export STRATA_GLM_RAM_RESIDENT=1
export STRATA_GLM_PROMOTE_MIN=6
export STRATA_GLM_KV_INT8=1
export STRATA_GLM_MTP_KEEP=2
export STRATA_GLM_SPEC_SAMPLE=1
export STRATA_GLM_SPEC_CANCEL=1
cd /home/aisever/project-maya
exec ./run-maya-maya-s24.sh
