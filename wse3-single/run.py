#!/usr/bin/env cs_python
"""
N-body benchmark host launcher.

Reads side and N_LOCAL from the compiled out/out.json, loads bodies from
bodies.npy (must exactly fit side*side*N_LOCAL slots -- no padding here),
pushes data, launches compute_step, then pulls back forces + timing buffers.

Outputs:
  - Min/max/avg cycle counts across all PE for:
      t_self     = compute_self_forces + pack_own_chunk
      t_pipeline = drain + emit + compute_chunk_forces (entire ring)
      t_total    = t_self + t_pipeline
  - Wall-clock for push, launch, pull.
  - Sanity check: bodies with non-zero force.
"""
import argparse
import json
import os
import time

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)


# ---------------------------------------------------------------------------
# Ring topology 
# ---------------------------------------------------------------------------

def ring_pos_to_pe(rp, side):
    pe_y = rp // side
    rem  = rp % side
    pe_x = rem if (pe_y % 2 == 0) else (side - 1 - rem)
    return pe_x, pe_y


def build_per_pe_tensors(bodies, side, n_local):
    """Distribute bodies in ring order. Returns local_particles, n_pred_fwd,
    n_pred_bwd, and a list mapping global body index -> (pe_x, pe_y, slot)."""
    total_pe = side * side
    n_total  = bodies.shape[0]
    assert n_total == total_pe * n_local, (
        f"N_TOTAL ({n_total}) != side^2 * N_LOCAL ({total_pe} * {n_local})"
    )

    local_particles = np.zeros((side, side, n_local * 7), dtype=np.float32)
    n_pred_fwd      = np.zeros((side, side, 1),           dtype=np.uint32)
    n_pred_bwd      = np.zeros((side, side, 1),           dtype=np.uint32)

    body_to_pe = []   # body_to_pe[g] = (pe_x, pe_y, slot)
    L = total_pe

    for rp in range(L):
        pe_x, pe_y = ring_pos_to_pe(rp, side)
        chunk = bodies[rp * n_local : (rp + 1) * n_local]   # (n_local, 7)
        local_particles[pe_y, pe_x, :] = chunk.flatten()
        n_pred_fwd[pe_y, pe_x, 0] = rp
        n_pred_bwd[pe_y, pe_x, 0] = (L - 1) - rp
        for slot in range(n_local):
            body_to_pe.append((pe_x, pe_y, slot))

    return local_particles, n_pred_fwd, n_pred_bwd, body_to_pe


# ---------------------------------------------------------------------------
# Memcpy helpers
# ---------------------------------------------------------------------------

def h2d_per_pe_f32(runner, sym, data, side, words_per_pe):
    runner.memcpy_h2d(sym, data.reshape(-1),
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False)


def h2d_per_pe_u32(runner, sym, data, side, words_per_pe):
    runner.memcpy_h2d(sym, data.reshape(-1),
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False)


def d2h_per_pe_f32(runner, sym, side, words_per_pe):
    out = np.zeros([side * side * words_per_pe], dtype=np.float32)
    runner.memcpy_d2h(out, sym,
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False)
    return out.reshape(side, side, words_per_pe)


def d2h_per_pe_u16(runner, sym, side, words_per_pe):
    out_u32 = np.zeros([side * side * words_per_pe], dtype=np.uint32)
    runner.memcpy_d2h(out_u32, sym,
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_16BIT,
                      nonblock=False)
    # Each u32 holds one u16 in its low bits.
    out_u16 = (out_u32 & 0xFFFF).astype(np.uint16)
    return out_u16.reshape(side, side, words_per_pe)


# ---------------------------------------------------------------------------
# Timestamp decoding
# ---------------------------------------------------------------------------

def decode_ts48(ts_u16_arr):
    """Decode a length-3 u16 array (little-endian) into a 48-bit int."""
    lo  = int(ts_u16_arr[0])
    mid = int(ts_u16_arr[1])
    hi  = int(ts_u16_arr[2])
    return lo | (mid << 16) | (hi << 32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",   default="out", help="Compile output dir")
    parser.add_argument("--cmaddr", default=None,  help="IP:port for CS system (omit for sim)")
    parser.add_argument("--bodies", default="bodies.npy", help="Input particles file")
    parser.add_argument("--results-json", default=None,
                        help="Optional path to write results as JSON (for batch sweeps)")
    args = parser.parse_args()

    # Read compile-time parameters.
    with open(f"{args.name}/out.json") as f:
        compile_data = json.load(f)
    side    = int(compile_data["params"]["side"])
    n_local = int(compile_data["params"]["N_LOCAL"])
    total_pe = side * side
    n_total  = total_pe * n_local

    print("=" * 64)
    print("WSE-3 N-body BENCHMARK")
    print("=" * 64)
    print(f"  Fabric         : {side} x {side} = {total_pe} PE")
    print(f"  N_LOCAL        : {n_local}  (bodies per PE)")
    print(f"  N_TOTAL        : {n_total}")
    print(f"  Target         : {'real CS-3' if args.cmaddr else 'simulator'}")
    print()

    # Load bodies.
    if not os.path.exists(args.bodies):
        raise FileNotFoundError(
            f"{args.bodies} not found. Run 'python generate_bodies.py --n {n_total}' first."
        )
    bodies = np.load(args.bodies)
    assert bodies.shape == (n_total, 7), \
        f"bodies shape {bodies.shape} != ({n_total}, 7) -- regenerate with --n {n_total}"

    # Build per-PE tensors.
    print(f"[1/4] Building per-PE tensors...")
    local_particles, n_pred_fwd, n_pred_bwd, body_to_pe = \
        build_per_pe_tensors(bodies, side, n_local)

    # Connect.
    print(f"[2/4] Loading kernel...")
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    sym = {
        "local_particles":  runner.get_id("ptr_local_particles"),
        "local_forces":     runner.get_id("ptr_local_forces"),
        "done":             runner.get_id("ptr_done"),
        "n_pred_fwd":       runner.get_id("ptr_n_pred_fwd"),
        "n_pred_bwd":       runner.get_id("ptr_n_pred_bwd"),
        "time_buf_u16":     runner.get_id("ptr_time_buf_u16"),
    }
    runner.load()
    runner.run()

    # Push data.
    print(f"[3/4] Pushing data...")
    t_push0 = time.time()
    h2d_per_pe_f32(runner, sym["local_particles"], local_particles, side, n_local * 7)
    h2d_per_pe_u32(runner, sym["n_pred_fwd"], n_pred_fwd, side, 1)
    h2d_per_pe_u32(runner, sym["n_pred_bwd"], n_pred_bwd, side, 1)
    t_push1 = time.time()
    t_push = t_push1 - t_push0

    # Launch.
    print(f"[4/4] Launching compute_step...")
    t_launch0 = time.time()
    runner.launch("compute_step", nonblock=False)
    t_launch1 = time.time()
    t_launch = t_launch1 - t_launch0

    # Pull results.
    t_pull0 = time.time()
    forces_3d  = d2h_per_pe_f32(runner, sym["local_forces"], side, n_local * 3)
    timing_3d  = d2h_per_pe_u16(runner, sym["time_buf_u16"], side, 9)
    t_pull1 = time.time()
    t_pull = t_pull1 - t_pull0

    runner.stop()

    # ----- Reorder forces back to global body order -----
    global_forces = np.zeros((n_total, 3), dtype=np.float32)
    for g, (pe_x, pe_y, slot) in enumerate(body_to_pe):
        global_forces[g] = forces_3d[pe_y, pe_x, slot*3:slot*3+3]

    n_nonzero = int(np.any(global_forces != 0.0, axis=1).sum())

    # ----- Decode timing for every PE -----
    t_self_cycles     = np.zeros(total_pe, dtype=np.int64)
    t_pipeline_cycles = np.zeros(total_pe, dtype=np.int64)
    t_total_cycles    = np.zeros(total_pe, dtype=np.int64)

    for pe_y in range(side):
        for pe_x in range(side):
            ts = timing_3d[pe_y, pe_x]   # length 9
            start = decode_ts48(ts[0:3])
            self_ = decode_ts48(ts[3:6])
            end   = decode_ts48(ts[6:9])

            idx = pe_y * side + pe_x
            t_self_cycles[idx]     = self_ - start
            t_pipeline_cycles[idx] = end - self_
            t_total_cycles[idx]    = end - start

    def stats(arr):
        return int(arr.min()), int(arr.max()), int(arr.mean())

    self_min, self_max, self_avg = stats(t_self_cycles)
    pipe_min, pipe_max, pipe_avg = stats(t_pipeline_cycles)
    tot_min,  tot_max,  tot_avg  = stats(t_total_cycles)

    CLK_HZ = 850e6   # WSE-3 nominal clock
    def cyc_to_us(c): return c / CLK_HZ * 1e6
    def cyc_to_ms(c): return c / CLK_HZ * 1e3

    # ----- Output -----
    print()
    print("--- DEVICE TIMING (cycles) ---")
    print(f"  t_self     : min={self_min:>10d}  max={self_max:>10d}  avg={self_avg:>10d}")
    print(f"  t_pipeline : min={pipe_min:>10d}  max={pipe_max:>10d}  avg={pipe_avg:>10d}")
    print(f"  t_total    : min={tot_min:>10d}  max={tot_max:>10d}  avg={tot_avg:>10d}")
    print()
    print(f"--- DEVICE TIMING (ms @ {CLK_HZ/1e6:.0f} MHz) ---")
    print(f"  t_self     : min={cyc_to_ms(self_min):>8.3f}  max={cyc_to_ms(self_max):>8.3f}  avg={cyc_to_ms(self_avg):>8.3f}  ms")
    print(f"  t_pipeline : min={cyc_to_ms(pipe_min):>8.3f}  max={cyc_to_ms(pipe_max):>8.3f}  avg={cyc_to_ms(pipe_avg):>8.3f}  ms")
    print(f"  t_total    : min={cyc_to_ms(tot_min):>8.3f}  max={cyc_to_ms(tot_max):>8.3f}  avg={cyc_to_ms(tot_avg):>8.3f}  ms")
    print()
    print(f"--- WALL-CLOCK (host side) ---")
    print(f"  Push data       : {t_push:.3f} s")
    print(f"  Launch          : {t_launch:.3f} s")
    print(f"  Pull results    : {t_pull:.3f} s")
    print()
    print(f"--- THROUGHPUT (estimated from t_total max) ---")
    # Total pair interactions: N * (N-1)
    n_pairs = n_total * (n_total - 1)
    if tot_max > 0:
        pairs_per_sec = n_pairs / (tot_max / CLK_HZ)
        print(f"  Pair interactions : {n_pairs:.3e}")
        print(f"  Pairs/sec         : {pairs_per_sec:.3e}")
    print()
    print(f"--- SANITY ---")
    print(f"  Bodies with non-zero force : {n_nonzero}/{n_total}")
    print()

    # Optional JSON output for batch sweeps.
    if args.results_json:
        results = {
            "side": side,
            "n_local": n_local,
            "n_total": n_total,
            "total_pe": total_pe,
            "cycles": {
                "t_self":     {"min": self_min, "max": self_max, "avg": self_avg},
                "t_pipeline": {"min": pipe_min, "max": pipe_max, "avg": pipe_avg},
                "t_total":    {"min": tot_min,  "max": tot_max,  "avg": tot_avg},
            },
            "ms_at_850MHz": {
                "t_self":     {"min": cyc_to_ms(self_min), "max": cyc_to_ms(self_max), "avg": cyc_to_ms(self_avg)},
                "t_pipeline": {"min": cyc_to_ms(pipe_min), "max": cyc_to_ms(pipe_max), "avg": cyc_to_ms(pipe_avg)},
                "t_total":    {"min": cyc_to_ms(tot_min),  "max": cyc_to_ms(tot_max),  "avg": cyc_to_ms(tot_avg)},
            },
            "host_wallclock_s": {
                "push":   t_push,
                "launch": t_launch,
                "pull":   t_pull,
            },
            "n_pairs": n_pairs,
            "pairs_per_sec_estimate": (n_pairs / (tot_max / CLK_HZ)) if tot_max > 0 else None,
            "n_bodies_nonzero_force": n_nonzero,
        }
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.results_json}")


if __name__ == "__main__":
    main()
