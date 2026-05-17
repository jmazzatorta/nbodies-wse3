#!/usr/bin/env cs_python
"""
N-body host launcher (multistep leapfrog).

Workflow:
  1. Load bodies from bodies.npy.
  2. Compute ring topology.
  3. Distribute bodies across PEs in ring-pos order.
  4. Push everything to the device.
  5. Launch compute_step.  The device runs N_STEPS leapfrog iterations.
  6. Pull final particles (positions + velocities) and forces.
  7. Validate against reference.npy (produced by validate_multistep.py).
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
# Ring topology helpers (must match layout.csl serpentine convention)
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
# Reference O(N^2) gravity force (host-side, sequential)
# ---------------------------------------------------------------------------

def compute_reference_forces(bodies, soft_eps_sq=1e-4, G=1.0):
    """
    Compute the gravitational force on each body using direct O(N^2) summation.
    Returns array of shape (N, 3) with [fx, fy, fz] per body.

    Slow for large N -- intended for small validation runs (N<=512).
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
    parser.add_argument("--reference", default="reference.npy",
                        help="reference final state (shape (N,10)) from validate_multistep.py")
    parser.add_argument("--validate", action="store_true",
                        help="compare device final state with reference.npy")
    parser.add_argument("--max-print", type=int, default=10,
                        help="max number of body lines to print")
    args = parser.parse_args()

    # ---- Load compile-time params ----
    with open(f"{args.name}/out.json", encoding="utf-8") as f:
        compile_data = json.load(f)
    csl_params = compile_data.get("params", {})
    n_local    = int(csl_params.get("N_LOCAL", 1))
    n_steps    = int(csl_params.get("N_STEPS", 1))
    dt_num     = int(csl_params.get("DT_NUM", 1))
    dt_den     = int(csl_params.get("DT_DEN", 10000))
    dt         = dt_num / dt_den
    grid_side  = 32   # hardcoded in layout.csl
    total_pe   = grid_side * grid_side

    print("=" * 60)
    print(f"WSE-3 N-body launcher  (multistep leapfrog)")
    print("=" * 60)
    print(f"  Fabric        : {grid_side} x {grid_side} = {total_pe} PE")
    print(f"  N_LOCAL       : {n_local}  (device slots per PE)")
    print(f"  Kernel slots  : {total_pe * n_local}  (max bodies)")
    print(f"  N_STEPS       : {n_steps}")
    print(f"  DT            : {dt}")
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

    # Padding policy: fill grid_side*grid_side*N_LOCAL slots with mass-zero
    # ghost particles if needed. Ghosts contribute zero force and don't
    # affect physics on the first n_real bodies; trimmed on host at the end.
    n_needed = total_pe * n_local
    if n_real == n_needed:
        print(f"[1/4] Loaded {n_real} bodies from {args.bodies} (exact fit).")
    elif n_real < n_needed:
        n_pad = n_needed - n_real
        pad = np.zeros((n_pad, 7), dtype=np.float32)
        bodies = np.concatenate([bodies, pad], axis=0).astype(np.float32)
        print(f"[1/4] Loaded {n_real} bodies from {args.bodies}, "
              f"padded with {n_pad} ghosts (m=0) to fill {n_needed} slots.")
    else:
        raise ValueError(
            f"bodies file has {n_real} particles but kernel only holds "
            f"{n_needed} slots. Recompile with a larger N_LOCAL."
        )

    n_total = bodies.shape[0]   # padded size, what the device sees

    # ---- Load reference (if --validate) ----
    reference = None
    if args.validate:
        if not os.path.exists(args.reference):
            raise FileNotFoundError(
                f"{args.reference} not found. Run validate_multistep.py first:\n"
                f"  python3 validate_multistep.py --n_steps {n_steps} --dt {dt}"
            )
        reference = np.load(args.reference)
        if reference.shape != (n_real, 10):
            raise ValueError(
                f"{args.reference} has shape {reference.shape}, expected ({n_real}, 10). "
                f"Re-run validate_multistep.py with matching N_STEPS and DT."
            )
        print(f"      reference.npy loaded (shape {reference.shape})")

    # ---- Build per-PE tensors ----
    print(f"[2/4] Building per-PE tensors...")
    (local_particles, ring_pos, n_pred_fwd, n_pred_bwd, body_to_rp) = \
        build_per_pe_tensors(bodies, grid_side, n_local)

    # Sanity counts
    print(f"      Total chunks on FWD stream: {int(n_pred_fwd.sum())}")
    print(f"      Total chunks on BWD stream: {int(n_pred_bwd.sum())}")

    # ---- Load runtime ----
    print(f"[3/4] Loading kernel onto device...")
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
    print(f"[4/4] Pushing data to device...")
    t0 = time.time()
    h2d_per_pe_f32(runner, sym["local_particles"], local_particles, grid_side, n_local * 7)
    h2d_per_pe_u32(runner, sym["ring_pos"],   ring_pos,   grid_side, 1)
    h2d_per_pe_u32(runner, sym["n_pred_fwd"], n_pred_fwd, grid_side, 1)
    h2d_per_pe_u32(runner, sym["n_pred_bwd"], n_pred_bwd, grid_side, 1)
    t1 = time.time()
    print(f"      data push wall-clock : {t1 - t0:.2f}s")

    # ---- Launch ----
    print(f"\nLaunching compute_step ({n_steps} leapfrog steps)...")
    t0 = time.time()
    try:
        runner.launch("compute_step", nonblock=False)
    except Exception as e:
        print(f"!!! CRASH DURING LAUNCH: {e}")
        runner.stop()
        return
    t1 = time.time()
    print(f"      compute wall-clock   : {t1 - t0:.3f}s")

    # ---- Pull results: particles (x, v, m) AND forces ----
    print(f"\nRetrieving final particles + forces...")
    t0 = time.time()
    particles_3d = d2h_per_pe_f32(runner, sym["local_particles"], grid_side, n_local * 7)
    forces_3d    = d2h_per_pe_f32(runner, sym["local_forces"],    grid_side, n_local * 3)
    t1 = time.time()
    print(f"      data pull wall-clock : {t1 - t0:.2f}s")

    # Reorder back to global body order.
    global_particles = np.zeros((n_total, 7), dtype=np.float32)
    global_forces    = np.zeros((n_total, 3), dtype=np.float32)
    for g in range(n_total):
        rp, slot = body_to_rp[g]
        pe_x, pe_y = ring_pos_to_pe(rp, grid_side)
        global_particles[g] = particles_3d[pe_y, pe_x, slot*7:slot*7+7]
        global_forces[g]    = forces_3d[pe_y, pe_x, slot*3:slot*3+3]

    # Trim ghost particles.
    global_particles = global_particles[:n_real]
    global_forces    = global_forces[:n_real]

    # Split into components.
    dev_x = global_particles[:, 0:3]
    dev_v = global_particles[:, 3:6]
    dev_m = global_particles[:, 6]
    dev_f = global_forces

    # ---- Display ----
    print(f"\n--- DEVICE RESULTS (final state after {n_steps} steps) ---")
    n_show = min(args.max_print, n_real)
    print(f"  First {n_show} body states:")
    for g in range(n_show):
        x, y, z = dev_x[g]
        vx, vy, vz = dev_v[g]
        fx, fy, fz = dev_f[g]
        print(f"    body {g:5d}: x=({x: .4e},{y: .4e},{z: .4e}) "
              f"v=({vx: .4e},{vy: .4e},{vz: .4e})")

    # ---- Validate against reference.npy ----
    if reference is not None:
        ref_x = reference[:, 0:3]
        ref_v = reference[:, 3:6]
        ref_f = reference[:, 7:10]

        def rel_err(a, b):
            diff = a.astype(np.float64) - b.astype(np.float64)
            err = np.linalg.norm(diff, axis=1) / (np.linalg.norm(b, axis=1) + 1e-12)
            return err.max(), err.mean()

        print(f"\n--- VALIDATION (device vs reference.npy) ---")
        max_x, avg_x = rel_err(dev_x, ref_x)
        max_v, avg_v = rel_err(dev_v, ref_v)
        max_f, avg_f = rel_err(dev_f, ref_f)
        print(f"  Positions  : max rel err = {max_x:.4e},  avg = {avg_x:.4e}")
        print(f"  Velocities : max rel err = {max_v:.4e},  avg = {avg_v:.4e}")
        print(f"  Forces     : max rel err = {max_f:.4e},  avg = {avg_f:.4e}")

        tol = 1e-3   # f32 accumulation tolerance over N_STEPS
        ok = max(max_x, max_v, max_f) < tol
        if ok:
            print(f"  ✅ Validation PASSED (within {tol:.0e})")
        else:
            print(f"  ❌ Validation FAILED (tol={tol:.0e})")
            for name, dev, ref, err_arr in [
                ("positions",  dev_x, ref_x, np.linalg.norm(dev_x - ref_x, axis=1)),
                ("velocities", dev_v, ref_v, np.linalg.norm(dev_v - ref_v, axis=1)),
                ("forces",     dev_f, ref_f, np.linalg.norm(dev_f - ref_f, axis=1)),
            ]:
                worst = np.argsort(err_arr)[-3:][::-1]
                print(f"    Worst {name}:")
                for g in worst:
                    print(f"      body {g}: dev={dev[g]}, ref={ref[g]}, |err|={err_arr[g]:.4e}")

    runner.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
