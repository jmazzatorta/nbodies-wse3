#!/bin/bash
# Compile the multistep N-body benchmark kernel for a SIDE x SIDE fabric.
#
# Usage:
#   ./commands.sh SIDE N_LOCAL N_STEPS DT_NUM DT_DEN
#
# Examples:
#   ./commands.sh 32 1 10 1 10000      # 32x32, 1 body/PE, 10 steps, dt=0.0001
#   ./commands.sh 32 4 100 1 10000     # 32x32, 4 bodies/PE, 100 steps, dt=0.0001
#   ./commands.sh 64 16 50 5 10000     # 64x64, 16 bodies/PE, 50 steps, dt=0.0005
#
# Notes:
#   - On simulated fabric, fabric-dims must be at least (side+7, side+2).
#   - On real CS-3, override via CS_FABRIC_DIMS, e.g.:
#       export CS_FABRIC_DIMS=757,996  # WSE-2 fabric (placeholder; check your system)

export PATH="$HOME/Documents/thesis/sdk:$PATH"
set -e

SIDE="${1:-32}"
N_LOCAL="${2:-1}"
N_STEPS="${3:-10}"
DT_NUM="${4:-1}"
DT_DEN="${5:-10000}"

# Simulator fabric dimensions: side + memcpy overhead.
SIMFAB_X=$((SIDE + 7))
SIMFAB_Y=$((SIDE + 2))
FABRIC_DIMS=${CS_FABRIC_DIMS:-${SIMFAB_X},${SIMFAB_Y}}

DT_VAL=$(python3 -c "print(${DT_NUM} / ${DT_DEN})")
echo "[INFO] Compiling: SIDE=${SIDE}, N_LOCAL=${N_LOCAL}"
echo "[INFO]            N_STEPS=${N_STEPS}, DT=${DT_NUM}/${DT_DEN}=${DT_VAL}"
echo "[INFO]            fabric-dims=${FABRIC_DIMS}"
echo "[INFO] Total PE in program rectangle: $((SIDE * SIDE))"
echo "[INFO] Total bodies (no padding): $((SIDE * SIDE * N_LOCAL))"

cslc --arch=wse3 ./layout.csl \
  --fabric-dims=${FABRIC_DIMS} \
  --fabric-offsets=4,1 \
  --params=side:${SIDE} \
  --params=N_LOCAL:${N_LOCAL} \
  --params=N_STEPS:${N_STEPS} \
  --params=DT_NUM:${DT_NUM} \
  --params=DT_DEN:${DT_DEN} \
  --params=MEMCPY_H2D_ID:0 \
  --params=MEMCPY_D2H_ID:1 \
  --max-inlined-iterations 500 \
  -o out --memcpy --channels 1

echo "[INFO] Compilation complete. Output in ./out/"
