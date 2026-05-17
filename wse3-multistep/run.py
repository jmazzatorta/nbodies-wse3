#!/usr/bin/env cs_python
"""
N-body MULTISTEP benchmark host launcher (production version).

Reads side, N_LOCAL, N_STEPS, DT_NUM/DT_DEN from out/out.json. Loads bodies
from bodies.npy (must exactly fit side*side*N_LOCAL slots), pushes data,
launches compute_step which runs N_STEPS leapfrog iterations on-device,
then pulls back the final particles, forces, and per-step timing buffer.

For each step s in 0..N_STEPS (inclusive of the final half-kick iteration),
the device records 3 timestamps:
    start[s] : at the top of step s (before compute_self_forces)
    self[s]  : after compute_self_forces + pack_own_chunk
    end[s]   : at the start of task_integrate for step s (after pipeline)

Per-PE deltas:
    t_self[s]     = self[s] - start[s]    (local self-forces + pack)
    t_pipeline[s] = end[s] - self[s]      (drain + emit + chunk_forces)
    t_total[s]    = end[s] - start[s]

The host aggregates min/max/avg across PEs for each step. Optional
--results-json writes a structured output for batch sweeps.
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
# Ring topology (must match layout.csl serpentine convention)
# ---------------------------------------------------------------------------

def ring_pos_to_pe(rp, side):
    pe_y = rp // side
    rem  = rp % side
    pe_x = rem if (pe_y % 2 == 0) else (side - 1 - rem)
    return pe_x, pe_y


def build_per_pe_tensors(bodies, side, n_local):
    """Distribute bodies in ring order. Returns local_particles, n_pred_fwd,
    n_pred_bwd, and the mapping global body index -> (pe_x, pe_y, slot)."""
    total_pe = side * side
    n_total  = bodies.shape[0]
    assert n_total == total_pe * n_local, (
        f"N_TOTAL ({n_total}) != side^2 * N_LOCAL ({total_pe} * {n_local})"
    )

    local_particles = np.zeros((side, side, n_local * 7), dtype=np.float32)
    n_pred_fwd      = np.zeros((side, side, 1),           dtype=np.uint32)
    n_pred_bwd      = np.zeros((side, side, 1),           dtype=np.uint32)

    body_to_pe = []
    L = total_pe
    for rp in range(L):
        pe_x, pe_y = ring_pos_to_pe(rp, side)
        chunk = bodies[rp * n_local : (rp + 1) * n_local]
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
    """Pull u16 values via MEMCPY_16BIT. Host buffer is u32; runtime
    populates only the low 16 bits."""
    out_u32 = np.zeros([side * side * words_per_pe], dtype=np.uint32)
    runner.memcpy_d2h(out_u32, sym,
                      0, 0, side, side, words_per_pe,
                      streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_16BIT,
                      nonblock=False)
    out_u16 = (out_u32 & 0xFFFF).astype(np.uint16)
    return out_u16.reshape(side, side, words_per_pe)


# ---------------------------------------------------------------------------
# Timestamp decoding
# ---------------------------------------------------------------------------

def decode_ts48(ts_u16_arr):
    """Decode 3 little-endian u16 into a 48-bit int."""
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
    parser.add_argument("--results-json", default="results.json",
                        help="Path to write structured results JSON.")
    parser.add_argument("--no-results-json", action="store_true",
                        help="Skip writing the JSON results file.")
    args = parser.parse_args()

    # ---- Read compile-time parameters ----
    with open(f"{args.name}/out.json") as f:
        compile_data = json.load(f)
    side     = int(compile_data["params"]["side"])
    n_local  = int(compile_data["params"]["N_LOCAL"])
    n_steps  = int(compile_data["params"]["N_STEPS"])
    dt_num   = int(compile_data["params"]["DT_NUM"])
    dt_den   = int(compile_data["params"]["DT_DEN"])
    dt       = dt_num / dt_den
    total_pe = side * side
    n_total  = total_pe * n_local

    # Timing buffer is 9 u16 per step * (N_STEPS+1) iterations.
    time_buf_words = 9 * (n_steps + 1)

    print("=" * 64)
    print("WSE-3 N-body MULTISTEP BENCHMARK")
    print("=" * 64)
    print(f"  Fabric         : {side} x {side} = {total_pe} PE")
    print(f"  N_LOCAL        : {n_local}  (bodies per PE)")
    print(f"  N_TOTAL        : {n_total}")
    print(f"  N_STEPS        : {n_steps}")
    print(f"  DT             : {dt_num}/{dt_den} = {dt}")
    print(f"  Target         : {'real CS-3' if args.cmaddr else 'simulator'}")
    print(f"  Time buffer    : {time_buf_words} u16 per PE")
    print()

    # ---- Load bodies (no padding allowed in bench) ----
    if not os.path.exists(args.bodies):
        raise FileNotFoundError(
            f"{args.bodies} not found. Run 'python generate_bodies.py --n {n_total}' first."
        )
    bodies = np.load(args.bodies)
    assert bodies.shape == (n_total, 7), (
        f"bodies shape {bodies.shape} != ({n_total}, 7). "
        f"Regenerate with --n {n_total}."
    )

    # ---- Build per-PE tensors ----
    print(f"[1/4] Building per-PE tensors...")
    local_particles, n_pred_fwd, n_pred_bwd, body_to_pe = \
        build_per_pe_tensors(bodies, side, n_local)

    # ---- Load runtime ----
    print(f"[2/4] Loading kernel...")
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True)
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

    # ---- Push data ----
    print(f"[3/4] Pushing data...")
    t_push0 = time.time()
    h2d_per_pe_f32(runner, sym["local_particles"], local_particles, side, n_local * 7)
    h2d_per_pe_u32(runner, sym["n_pred_fwd"], n_pred_fwd, side, 1)
    h2d_per_pe_u32(runner, sym["n_pred_bwd"], n_pred_bwd, side, 1)
    t_push = time.time() - t_push0

    # ---- Launch ----
    print(f"[4/4] Launching compute_step ({n_steps} leapfrog steps on-device)...")
    t_launch0 = time.time()
    runner.launch("compute_step", nonblock=False)
    t_launch = time.time() - t_launch0

    # ---- Pull results ----
    t_pull0 = time.time()
    particles_3d = d2h_per_pe_f32(runner, sym["local_particles"], side, n_local * 7)
    forces_3d    = d2h_per_pe_f32(runner, sym["local_forces"],    side, n_local * 3)
    timing_3d    = d2h_per_pe_u16(runner, sym["time_buf_u16"],    side, time_buf_words)
    t_pull = time.time() - t_pull0

    runner.stop()

    # ---- Reorder particles + forces back to global body order ----
    global_particles = np.zeros((n_total, 7), dtype=np.float32)
    global_forces    = np.zeros((n_total, 3), dtype=np.float32)
    for g, (pe_x, pe_y, slot) in enumerate(body_to_pe):
        global_particles[g] = particles_3d[pe_y, pe_x, slot*7:slot*7+7]
        global_forces[g]    = forces_3d[pe_y, pe_x, slot*3:slot*3+3]

    # ---- Decode timing for every PE, every step ----
    # For step s, we have 9 u16:
    #   [0..2]: start  -> tsc_start[s]
    #   [3..5]: self   -> tsc_self[s]
    #   [6..8]: end    -> tsc_end[s]
    # We have N_STEPS+1 steps; the last one (s == N_STEPS) only has start
    # and end recorded (no self_forces phase). Specifically, the device
    # records end[N_STEPS] but not start[N_STEPS]/self[N_STEPS], so for
    # step N_STEPS we only have a meaningful end timestamp (start/self
    # for step N_STEPS were taken at the end of step N_STEPS-1's integrate).
    #
    # In practice: start[s] and self[s] for s>0 are written by task_integrate
    # of step s-1 after incrementing current_step.  For s=N_STEPS (final
    # half-kick), the start/self slots are never written (default 0).

    t_self_cycles     = np.zeros((total_pe, n_steps + 1), dtype=np.int64)
    t_pipeline_cycles = np.zeros((total_pe, n_steps + 1), dtype=np.int64)
    t_total_cycles    = np.zeros((total_pe, n_steps + 1), dtype=np.int64)

    for pe_y in range(side):
        for pe_x in range(side):
            buf = timing_3d[pe_y, pe_x]   # shape (time_buf_words,)
            idx_pe = pe_y * side + pe_x
            for s in range(n_steps + 1):
                base = 9 * s
                start = decode_ts48(buf[base + 0 : base + 3])
                self_ = decode_ts48(buf[base + 3 : base + 6])
                end   = decode_ts48(buf[base + 6 : base + 9])
                t_self_cycles[idx_pe, s]     = self_ - start
                t_pipeline_cycles[idx_pe, s] = end - self_
                t_total_cycles[idx_pe, s]    = end - start

    # The last step (s = N_STEPS) has only the end timestamp meaningful
    # (the pipeline runs to compute a_N for the final half-kick). The
    # start/self slots for s=N_STEPS were not written by the device.
    # Therefore t_self[N_STEPS] and t_pipeline[N_STEPS] are not meaningful;
    # only t_total[N_STEPS] (relative to start[N_STEPS-1]'s end) makes sense.
    # We report the per-step stats for s=0..N_STEPS-1 (the "active" steps)
    # plus a separate breakdown for s=N_STEPS (the final pipeline iter).

    CLK_HZ = 850e6   # WSE-3 nominal clock
    def cyc_to_ms(c): return c / CLK_HZ * 1e3

    def stats(arr):
        return int(arr.min()), int(arr.max()), int(arr.mean())

    # ---- Per-step stats (steps 0..N_STEPS-1: "active" steps) ----
    print()
    print("--- DEVICE TIMING per step (cycles) ---")
    print(f"  step | t_self                          | t_pipeline                      | t_total")
    print(f"       | min        max        avg       | min        max        avg       | min        max        avg")
    print(f"  -----+---------------------------------+---------------------------------+---------------------------------")
    for s in range(n_steps):
        ts_min, ts_max, ts_avg = stats(t_self_cycles[:, s])
        tp_min, tp_max, tp_avg = stats(t_pipeline_cycles[:, s])
        tt_min, tt_max, tt_avg = stats(t_total_cycles[:, s])
        print(f"   {s:3d} | {ts_min:9d} {ts_max:9d} {ts_avg:9d} | "
              f"{tp_min:9d} {tp_max:9d} {tp_avg:9d} | "
              f"{tt_min:9d} {tt_max:9d} {tt_avg:9d}")

    # Aggregate across all steps.
    self_all = t_self_cycles[:, :n_steps].flatten()
    pipe_all = t_pipeline_cycles[:, :n_steps].flatten()
    tot_all  = t_total_cycles[:, :n_steps].flatten()
    self_min, self_max, self_avg = stats(self_all)
    pipe_min, pipe_max, pipe_avg = stats(pipe_all)
    tot_min,  tot_max,  tot_avg  = stats(tot_all)

    print()
    print("--- AGGREGATE (steps 0..N_STEPS-1) ---")
    print(f"  t_self     : min={self_min:>10d}  max={self_max:>10d}  avg={self_avg:>10d}   ({cyc_to_ms(self_avg):.3f} ms avg)")
    print(f"  t_pipeline : min={pipe_min:>10d}  max={pipe_max:>10d}  avg={pipe_avg:>10d}   ({cyc_to_ms(pipe_avg):.3f} ms avg)")
    print(f"  t_total    : min={tot_min:>10d}   max={tot_max:>10d}   avg={tot_avg:>10d}    ({cyc_to_ms(tot_avg):.3f} ms avg)")

    print()
    print(f"--- WALL-CLOCK (host side) ---")
    print(f"  Push data       : {t_push:.3f} s")
    print(f"  Launch          : {t_launch:.3f} s")
    print(f"  Pull results    : {t_pull:.3f} s")

    # Throughput estimate (one step worth of pair-interactions, per second).
    n_pairs = n_total * (n_total - 1)
    if tot_max > 0:
        pairs_per_sec = n_pairs / (tot_max / CLK_HZ)
        print()
        print(f"--- THROUGHPUT (1 step, device-only, from max t_total) ---")
        print(f"  Pair interactions per step : {n_pairs:.3e}")
        print(f"  Pairs/sec                  : {pairs_per_sec:.3e}")

    n_nonzero = int(np.any(global_forces != 0.0, axis=1).sum())
    print()
    print(f"--- SANITY ---")
    print(f"  Bodies with non-zero force : {n_nonzero}/{n_total}")

    # ---- Save device final state (compatible shape with reference.npy from validate_multistep.py) ----
    device_state = np.concatenate([
        global_particles[:, 0:3],   # positions
        global_particles[:, 3:6],   # velocities (after final half-kick)
        global_particles[:, 6:7],   # masses
        global_forces[:, 0:3],      # forces
    ], axis=1).astype(np.float32)
    np.save("device_final.npy", device_state)
    print(f"  Device final state saved to device_final.npy (shape {device_state.shape})")

    # ---- JSON output ----
    if not args.no_results_json:
        # Compute per-step stats for JSON.
        per_step = []
        for s in range(n_steps):
            ts_min, ts_max, ts_avg = stats(t_self_cycles[:, s])
            tp_min, tp_max, tp_avg = stats(t_pipeline_cycles[:, s])
            tt_min, tt_max, tt_avg = stats(t_total_cycles[:, s])
            per_step.append({
                "step": s,
                "t_self":     {"min": ts_min, "max": ts_max, "avg": ts_avg},
                "t_pipeline": {"min": tp_min, "max": tp_max, "avg": tp_avg},
                "t_total":    {"min": tt_min, "max": tt_max, "avg": tt_avg},
            })

        results = {
            "config": {
                "side": side,
                "n_local": n_local,
                "n_total": n_total,
                "total_pe": total_pe,
                "n_steps": n_steps,
                "dt_num": dt_num,
                "dt_den": dt_den,
                "dt": dt,
            },
            "per_step_cycles": per_step,
            "aggregate_cycles": {
                "t_self":     {"min": self_min, "max": self_max, "avg": self_avg},
                "t_pipeline": {"min": pipe_min, "max": pipe_max, "avg": pipe_avg},
                "t_total":    {"min": tot_min,  "max": tot_max,  "avg": tot_avg},
            },
            "ms_at_850MHz": {
                "t_self_avg":     cyc_to_ms(self_avg),
                "t_pipeline_avg": cyc_to_ms(pipe_avg),
                "t_total_avg":    cyc_to_ms(tot_avg),
            },
            "host_wallclock_s": {
                "push":   t_push,
                "launch": t_launch,
                "pull":   t_pull,
            },
            "throughput": {
                "n_pairs_per_step": n_pairs,
                "pairs_per_sec_est": (n_pairs / (tot_max / CLK_HZ)) if tot_max > 0 else None,
            },
            "sanity": {
                "n_bodies_nonzero_force": n_nonzero,
            },
        }
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.results_json}")


if __name__ == "__main__":
    main()
