#!/usr/bin/env cs_python
"""
N-body host launcher (single step).

Workflow:
  1. Load bodies from bodies.npy (or generate on the fly).
  2. Compute ring topology: each PE gets a ring_pos (0..1023) and
     flags has_fwd_neighbor / has_bwd_neighbor.
  3. Distribute 16 bodies per PE in ring-pos order.
  4. Push everything to the device.
  5. Launch compute_step.
  6. Pull forces back.
  7. Validate (if --validate).
"""
import argparse
import json
import time
import os
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)


# ---------------------------------------------------------------------------
# Ring topology helpers 
# ---------------------------------------------------------------------------

def pe_to_ring_pos(pe_x, pe_y, grid_side):
    """Serpentine row-major: even rows go east, odd rows go west."""
    if pe_y % 2 == 0:
        return pe_y * grid_side + pe_x
    else:
        return pe_y * grid_side + (grid_side - 1 - pe_x)


def ring_pos_to_pe(rp, grid_side):
    """Inverse of pe_to_ring_pos."""
    pe_y = rp // grid_side
    rem = rp % grid_side
    if pe_y % 2 == 0:
        pe_x = rem
    else:
        pe_x = grid_side - 1 - rem
    return pe_x, pe_y


# ---------------------------------------------------------------------------
# Reference O(N^2) gravity force 
# ---------------------------------------------------------------------------

def compute_reference_forces(bodies, soft_eps_sq=1e-4, G=1.0):
    """
    Compute the gravitational force on each body using direct O(N^2) summation.
    Returns array of shape (N, 3) with [fx, fy, fz] per body.
    """
    N = bodies.shape[0]
    forces = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        rx, ry, rz, _, _, _, mi = bodies[i]
        for j in range(N):
            if i == j:
                continue
            xj, yj, zj, _, _, _, mj = bodies[j]
            dx = xj - rx
            dy = yj - ry
            dz = zj - rz
            r2 = dx * dx + dy * dy + dz * dz + soft_eps_sq
            inv_r3 = 1.0 / (r2 ** 1.5)
            f = G * mi * mj * inv_r3
            forces[i, 0] += f * dx
            forces[i, 1] += f * dy
            forces[i, 2] += f * dz
    return forces


# ---------------------------------------------------------------------------
# Distribution onto PE grid
# ---------------------------------------------------------------------------

def build_per_pe_tensors(bodies, grid_side, n_local):
    """
    Build the per-PE tensors that will be pushed via memcpy.

    Returns:
        local_particles : float32 array of shape (gs, gs, n_local * 7)
        ring_pos        : uint32 array of shape (gs, gs, 1)
        n_pred_fwd      : uint32 array (gs, gs, 1) — chunks expected on FWD
        n_pred_bwd      : uint32 array (gs, gs, 1) — chunks expected on BWD
        body_to_ringpos : list mapping global body index -> (ring_pos, slot)

    has_fwd / has_bwd are NOT host-pushed: they are comptime parameters
    set by layout.csl based on the PE position in the serpentine ring.
    """
    total_pe = grid_side * grid_side
    n_total = bodies.shape[0]
    assert n_total == total_pe * n_local, (
        f"N_TOTAL_BODIES ({n_total}) != grid_side^2 * N_LOCAL "
        f"({total_pe} * {n_local} = {total_pe * n_local})"
    )

    local_particles = np.zeros((grid_side, grid_side, n_local * 7), dtype=np.float32)
    ring_pos        = np.zeros((grid_side, grid_side, 1), dtype=np.uint32)
    n_pred_fwd      = np.zeros((grid_side, grid_side, 1), dtype=np.uint32)
    n_pred_bwd      = np.zeros((grid_side, grid_side, 1), dtype=np.uint32)

    body_to_ringpos = []   # body_to_ringpos[g] = (ring_pos, slot)

    L = total_pe   # ring length

    for rp in range(total_pe):
        pe_x, pe_y = ring_pos_to_pe(rp, grid_side)

        # Bodies for this PE: [rp * n_local, (rp+1) * n_local)
        start = rp * n_local
        end   = start + n_local
        chunk = bodies[start:end]   # shape (n_local, 7)

        # Flatten and store
        local_particles[pe_y, pe_x, :] = chunk.flatten()

        # Ring metadata
        ring_pos[pe_y, pe_x, 0] = rp

        # Number of predecessor chunks expected on each stream:
        #   FWD travels from PE 0 to PE L-1; PE at ring_pos rp drains
        #   chunks emitted by PE 0..rp-1 -> rp chunks.
        #   BWD is the mirror -> (L-1)-rp chunks.
        n_pred_fwd[pe_y, pe_x, 0] = rp
        n_pred_bwd[pe_y, pe_x, 0] = (L - 1) - rp

        for slot in range(n_local):
            body_to_ringpos.append((rp, slot))

    return (local_particles, ring_pos, n_pred_fwd, n_pred_bwd, body_to_ringpos)


# ---------------------------------------------------------------------------
# Memcpy helpers
# ---------------------------------------------------------------------------

def h2d_per_pe_f32(runner, sym, tensor_3d, grid_side, elems_per_pe):
    runner.memcpy_h2d(
        sym, tensor_3d.ravel(),
        0, 0, grid_side, grid_side, elems_per_pe,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
        order=MemcpyOrder.ROW_MAJOR,
    )


def h2d_per_pe_u32(runner, sym, tensor_3d, grid_side, elems_per_pe):
    runner.memcpy_h2d(
        sym, tensor_3d.ravel(),
        0, 0, grid_side, grid_side, elems_per_pe,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
        order=MemcpyOrder.ROW_MAJOR,
    )


def d2h_per_pe_f32(runner, sym, grid_side, elems_per_pe):
    out = np.zeros(grid_side * grid_side * elems_per_pe, dtype=np.float32)
    runner.memcpy_d2h(
        out, sym,
        0, 0, grid_side, grid_side, elems_per_pe,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
        order=MemcpyOrder.ROW_MAJOR,
    )
    return out.reshape(grid_side, grid_side, elems_per_pe)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",     required=True, help="compile output dir")
    parser.add_argument("--cmaddr",   help="IP:port for CS system")
    parser.add_argument("--bodies",   default="bodies.npy",
                        help="input bodies file (.npy)")
    parser.add_argument("--validate", action="store_true",
                        help="compute reference forces on host (slow)")
    parser.add_argument("--max-print", type=int, default=10,
                        help="max number of force outputs to print")
    args = parser.parse_args()

    # ---- Load compile-time params ----
    with open(f"{args.name}/out.json", encoding="utf-8") as f:
        compile_data = json.load(f)
    csl_params = compile_data.get("params", {})
    n_total    = int(csl_params.get("N_TOTAL_BODIES", 16384))
    n_local    = int(csl_params.get("N_LOCAL", 16))
    grid_side  = 32   # hardcoded in layout.csl
    total_pe   = grid_side * grid_side

    print("=" * 60)
    print(f"WSE-3 N-body launcher")
    print("=" * 60)
    print(f"  Fabric        : {grid_side} x {grid_side} = {total_pe} PE")
    print(f"  N_LOCAL       : {n_local}  (device slots per PE)")
    print(f"  Kernel slots  : {total_pe * n_local}  (max bodies the kernel can handle)")
    print(f"  Validation    : {'YES' if args.validate else 'NO'}")
    print()

    # ---- Load bodies ----
    if not os.path.exists(args.bodies):
        raise FileNotFoundError(
            f"{args.bodies} not found. Run 'python generate_bodies.py --n N' first."
        )
    bodies = np.load(args.bodies)
    n_real = bodies.shape[0]
    assert bodies.shape[1] == 7, f"bodies must have 7 columns, got {bodies.shape}"

    # Padding policy: if the loaded body count does not exactly fill the
    # grid_side*grid_side*N_LOCAL slots, pad with mass-zero "ghost"
    # particles. Ghost particles contribute zero force (F = G*m_i*m_j*...
    # is zero when either mass is zero), so the physics on the first
    # n_real bodies is preserved. The device sees a uniform N_LOCAL
    # particles per PE; the host trims results back to n_real at the end.
    n_needed = total_pe * n_local
    if n_real == n_needed:
        print(f"[1/5] Loaded {n_real} bodies from {args.bodies} (exact fit).")
    elif n_real < n_needed:
        n_pad = n_needed - n_real
        pad = np.zeros((n_pad, 7), dtype=np.float32)
        # Position must be valid but irrelevant; mass is 0.
        bodies = np.concatenate([bodies, pad], axis=0).astype(np.float32)
        print(f"[1/5] Loaded {n_real} bodies from {args.bodies}, "
              f"padded with {n_pad} ghost particles (m=0) to fill {n_needed} slots.")
    else:
        raise ValueError(
            f"bodies file has {n_real} particles but kernel only holds "
            f"{n_needed} slots ({total_pe} PE * {n_local} N_LOCAL). "
            f"Recompile with a larger N_LOCAL or use fewer particles."
        )

    n_total = bodies.shape[0]   # padded size, what the device sees

    # ---- Compute reference (only if --validate) ----
    ref_forces = None
    if args.validate:
        if n_real > 1024:
            print(f"  WARNING: --validate with N={n_real} will take a long time.")
        print(f"[2/5] Computing reference forces on host (O(N^2) sequential)...")
        t0 = time.time()
        # Reference only on the REAL bodies; ghost particles have m=0
        # and contribute zero force anyway, so they don't change the
        # result but would waste time.
        ref_forces = compute_reference_forces(bodies[:n_real])
        t1 = time.time()
        print(f"      reference wall-clock: {t1 - t0:.2f}s")
    else:
        print(f"[2/5] Skipped host reference (use --validate to enable).")

    # ---- Build per-PE tensors ----
    print(f"[3/5] Building per-PE tensors...")
    (local_particles, ring_pos, n_pred_fwd, n_pred_bwd, body_to_rp) = \
        build_per_pe_tensors(bodies, grid_side, n_local)

    # Sanity counts
    print(f"      Total chunks on FWD stream: {int(n_pred_fwd.sum())}")
    print(f"      Total chunks on BWD stream: {int(n_pred_bwd.sum())}")

    # ---- Load runtime ----
    print(f"[4/5] Loading kernel onto device...")
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True)
    runner.load()
    runner.run()

    sym = {
        "local_particles":    runner.get_id("ptr_local_particles"),
        "local_forces":       runner.get_id("ptr_local_forces"),
        "done":               runner.get_id("ptr_done"),
        "ring_pos":           runner.get_id("ptr_ring_pos"),
        "n_pred_fwd":         runner.get_id("ptr_n_pred_fwd"),
        "n_pred_bwd":         runner.get_id("ptr_n_pred_bwd"),
        "canarino":           runner.get_id("ptr_canarino"),
    }

    # ---- Push data ----
    print(f"[5/5] Pushing data to device...")
    t0 = time.time()
    h2d_per_pe_f32(runner, sym["local_particles"], local_particles, grid_side, n_local * 7)
    h2d_per_pe_u32(runner, sym["ring_pos"],   ring_pos,   grid_side, 1)
    h2d_per_pe_u32(runner, sym["n_pred_fwd"], n_pred_fwd, grid_side, 1)
    h2d_per_pe_u32(runner, sym["n_pred_bwd"], n_pred_bwd, grid_side, 1)
    t1 = time.time()
    print(f"      data push wall-clock : {t1 - t0:.2f}s")

    # ---- Launch ----
    print(f"\nLaunching compute_step...")
    t0 = time.time()
    try:
        runner.launch("compute_step", nonblock=False)
    except Exception as e:
        print(f"!!! CRASH DURING LAUNCH: {e}")
        runner.stop()
        return
    t1 = time.time()
    print(f"      compute wall-clock   : {t1 - t0:.3f}s")

    # ---- Pull forces ----
    print(f"\nRetrieving forces...")
    t0 = time.time()
    forces_3d = d2h_per_pe_f32(runner, sym["local_forces"], grid_side, n_local * 3)
    t1 = time.time()
    print(f"      data pull wall-clock : {t1 - t0:.2f}s")

    # Reshape: forces_3d[pe_y, pe_x, slot*3 .. slot*3+3]
    # Reorder back to global body index using body_to_rp
    global_forces = np.zeros((n_total, 3), dtype=np.float32)
    for g in range(n_total):
        rp, slot = body_to_rp[g]
        pe_x, pe_y = ring_pos_to_pe(rp, grid_side)
        global_forces[g] = forces_3d[pe_y, pe_x, slot*3:slot*3+3]

    # Trim padded ghost particles for display and validation.
    global_forces = global_forces[:n_real]

    # ---- Display / validate ----
    print(f"\n--- DEVICE RESULTS ---")
    n_nonzero = int(np.any(global_forces != 0.0, axis=1).sum())
    print(f"  Bodies with non-zero force: {n_nonzero}/{n_real}")

    n_show = min(args.max_print, n_real)
    print(f"\n  First {n_show} body forces:")
    for g in range(n_show):
        fx, fy, fz = global_forces[g]
        print(f"    body {g:5d}:  F = ({fx: .6e}, {fy: .6e}, {fz: .6e})")

    if ref_forces is not None:
        print(f"\n--- VALIDATION ---")
        diff = global_forces.astype(np.float64) - ref_forces
        rel_err = np.linalg.norm(diff, axis=1) / (np.linalg.norm(ref_forces, axis=1) + 1e-10)
        max_err = rel_err.max()
        avg_err = rel_err.mean()
        print(f"  Max relative error: {max_err:.4e}")
        print(f"  Avg relative error: {avg_err:.4e}")
        if max_err < 1e-3:
            print(f"  ✅ Validation PASSED (within tolerance)")
        else:
            print(f"  ❌ Validation FAILED (relative error too large)")
            # Print worst cases
            worst = np.argsort(rel_err)[-5:][::-1]
            for g in worst:
                print(f"    body {g}: device={global_forces[g]}, ref={ref_forces[g]}, "
                      f"rel_err={rel_err[g]:.4e}")

    runner.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
