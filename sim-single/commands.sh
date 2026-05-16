#!/bin/bash
# Compile the N-body CSL kernel for fabric 32x32.
#
# Default: small test (N_LOCAL=1, 1 body per PE, 1024 bodies total).
#   This makes compute_chunk_forces do 1 pair per chunk instead of 256,
#   so the simulator finishes in minutes instead of hours.
# Use 'full' as first arg for the production-size run (N_LOCAL=16, 16384 bodies).

export PATH="$HOME/Documents/thesis/sdk:$PATH"
set -e

if [ "${1:-small}" = "full" ]; then
    N_TOTAL=16384
    N_LOCAL=16
    echo "[INFO] FULL configuration: N_TOTAL=${N_TOTAL}, N_LOCAL=${N_LOCAL}"
elif [ "${1:-small}" = "large" ]; then
    N_TOTAL=8192
    N_LOCAL=8
    echo "[INFO] LARGE configuration: N_TOTAL=${N_TOTAL}, N_LOCAL=${N_LOCAL}"
elif [ "${1:-small}" = "medium4" ]; then
    N_TOTAL=4096
    N_LOCAL=4
    echo "[INFO] MEDIUM4 configuration: N_TOTAL=${N_TOTAL}, N_LOCAL=${N_LOCAL}"
elif [ "${1:-small}" = "medium" ]; then
    N_TOTAL=2048
    N_LOCAL=2
    echo "[INFO] MEDIUM configuration: N_TOTAL=${N_TOTAL}, N_LOCAL=${N_LOCAL}"
else
    N_TOTAL=1024
    N_LOCAL=1
    echo "[INFO] SMALL configuration: N_TOTAL=${N_TOTAL}, N_LOCAL=${N_LOCAL}"
fi

cslc --arch=wse3 ./layout.csl \
  --fabric-dims=39,34 \
  --fabric-offsets=4,1 \
  --params=N_TOTAL_BODIES:${N_TOTAL} \
  --params=N_LOCAL:${N_LOCAL} \
  --params=MEMCPY_H2D_ID:0 \
  --params=MEMCPY_D2H_ID:1 \
  --max-inlined-iterations 500 \
  -o out --memcpy --channels 1
