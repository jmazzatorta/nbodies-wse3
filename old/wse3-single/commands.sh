#!/bin/bash
# Compile the N-body benchmark kernel for a SIDE x SIDE fabric with N_LOCAL bodies per PE.
#
# Usage:
#   ./commands.sh SIDE N_LOCAL
#   ./commands.sh 32 4         # 32x32 fabric, 4 bodies per PE (4096 total)
#   ./commands.sh 64 16        # 64x64 fabric, 16 bodies per PE (65536 total)
#   ./commands.sh 128 1        # 128x128 fabric, 1 body per PE (16384 total)
#
# Notes:
#   - On simulated fabric, fabric-dims must be at least (side+7, side+2)
#   - On real CS-3 / WSE-3 (~900,000 PE in ~950x950 layout), set the actual
#     wafer fabric dimensions via the CS_FABRIC_DIMS env var, e.g.:
#         export CS_FABRIC_DIMS=...,...  # WSE-3 actual fabric 

export PATH="$HOME/Documents/thesis/sdk:$PATH"
set -e

SIDE=${1:-32}
N_LOCAL=${2:-1}

# Compute simulator fabric dimensions: side + memcpy overhead.
SIMFAB_X=$((SIDE + 7))
SIMFAB_Y=$((SIDE + 2))

# Use CS_FABRIC_DIMS if set (for real wafer), else simfab dimensions.
FABRIC_DIMS=${CS_FABRIC_DIMS:-${SIMFAB_X},${SIMFAB_Y}}

echo "[INFO] Compiling: SIDE=${SIDE}, N_LOCAL=${N_LOCAL}, fabric-dims=${FABRIC_DIMS}"
echo "[INFO] Total PE in program rectangle: $((SIDE * SIDE))"
echo "[INFO] Total bodies (no padding): $((SIDE * SIDE * N_LOCAL))"

cslc --arch=wse3 ./layout.csl \
  --fabric-dims=${FABRIC_DIMS} \
  --fabric-offsets=4,1 \
  --params=side:${SIDE} \
  --params=N_LOCAL:${N_LOCAL} \
  --params=MEMCPY_H2D_ID:0 \
  --params=MEMCPY_D2H_ID:1 \
  --max-inlined-iterations 500 \
  -o out --memcpy --channels 1

echo "[INFO] Compilation complete. Output in out/"
