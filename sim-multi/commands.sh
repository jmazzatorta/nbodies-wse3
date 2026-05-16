#!/bin/bash
# Compile the multistep N-body CSL kernel for fabric 32x32.
#
# Usage:
#   ./commands.sh [SIZE] [N_STEPS] [DT_NUM] [DT_DEN]
#
# SIZE:    small (default) | medium | medium4 | large | full
# N_STEPS: number of leapfrog integration steps (default 10)
# DT_NUM:  numerator of timestep (default 1)
# DT_DEN:  denominator of timestep (default 10000)
#          -> DT = DT_NUM / DT_DEN = 0.0001 by default
#
# Examples:
#   ./commands.sh                       # small, 10 steps, dt=1/10000=0.0001
#   ./commands.sh small 100 1 10000     # small, 100 steps, dt=0.0001
#   ./commands.sh small 10 1 1000       # small, 10 steps, dt=0.001
#   ./commands.sh medium 50 5 10000     # medium, 50 steps, dt=0.0005

export PATH="$HOME/Documents/thesis/sdk:$PATH"
set -e

SIZE="${1:-small}"
N_STEPS="${2:-10}"
DT_NUM="${3:-1}"
DT_DEN="${4:-10000}"

if [ "${SIZE}" = "full" ]; then
    N_TOTAL=16384
    N_LOCAL=16
elif [ "${SIZE}" = "large" ]; then
    N_TOTAL=8192
    N_LOCAL=8
elif [ "${SIZE}" = "medium4" ]; then
    N_TOTAL=4096
    N_LOCAL=4
elif [ "${SIZE}" = "medium" ]; then
    N_TOTAL=2048
    N_LOCAL=2
else
    N_TOTAL=1024
    N_LOCAL=1
fi

DT_VAL=$(python3 -c "print(${DT_NUM} / ${DT_DEN})")
echo "[INFO] SIZE=${SIZE}: N_TOTAL=${N_TOTAL}, N_LOCAL=${N_LOCAL}"
echo "[INFO] N_STEPS=${N_STEPS}, DT=${DT_NUM}/${DT_DEN}=${DT_VAL}"

cslc --arch=wse3 ./layout.csl \
  --fabric-dims=39,34 \
  --fabric-offsets=4,1 \
  --params=N_TOTAL_BODIES:${N_TOTAL} \
  --params=N_LOCAL:${N_LOCAL} \
  --params=N_STEPS:${N_STEPS} \
  --params=DT_NUM:${DT_NUM} \
  --params=DT_DEN:${DT_DEN} \
  --params=MEMCPY_H2D_ID:0 \
  --params=MEMCPY_D2H_ID:1 \
  --max-inlined-iterations 500 \
  -o out --memcpy --channels 1
