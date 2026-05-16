#!/usr/bin/env python3
"""
High-fidelity leapfrog (KDK fused) N-body reference for validating WSE-3 results.

The leapfrog scheme is KDK fused with initial half-kick-backward and
final half-kick-forward (so that input and output velocities are both
at integer timesteps t=0 and t=N*dt):

    Initial:
        a_0 = forces(x_0) / m
        v_{-1/2} = v_0 - a_0 * dt/2

    Loop n = 0..N_STEPS-1:
        v_{n+1/2} = v_{n-1/2} + a_n * dt          # full kick
        x_{n+1}   = x_n + v_{n+1/2} * dt          # drift
        a_{n+1}   = forces(x_{n+1}) / m           # new accelerations

    Final:
        v_N = v_{N-1/2} + a_N * dt/2              # half kick forward
"""
import argparse
import time

import numpy as np

# Physics constants (must match constants.csl). These are exact
# float64 representations; the device sees them after compile-time
# f32 reinterpretation.
SOFT_EPS_SQ = 1.0e-4   # f64
G_GRAV      = 1.0      # f64


def compute_accelerations(x, m):
    """
    Compute O(N^2) Plummer-softened gravitational accelerations.

    Numerical strategy:
      - All arithmetic in f64.
      - Direct division by r^3 (not multiplication by 1/r^3).
      - Explicit pair loop; for each i, accumulate contributions from
        j = 0, 1, ..., N-1 (skipping j == i) using Kahan compensated
        summation per component.

    x : (N, 3) array of positions (will be promoted to f64 internally)
    m : (N,)  array of masses    (will be promoted to f64 internally)
    Returns:
        a : (N, 3) f64 array of accelerations
    """
    N = x.shape[0]
    x64 = np.asarray(x, dtype=np.float64)
    m64 = np.asarray(m, dtype=np.float64)
    a   = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        xi, yi, zi = x64[i, 0], x64[i, 1], x64[i, 2]

        # Kahan accumulators for the three components of a_i.
        ax, ay, az             = 0.0, 0.0, 0.0
        comp_x, comp_y, comp_z = 0.0, 0.0, 0.0   # compensation terms

        for j in range(N):
            if j == i:
                continue

            dx = x64[j, 0] - xi
            dy = x64[j, 1] - yi
            dz = x64[j, 2] - zi

            r2     = dx * dx + dy * dy + dz * dz
            r2_eps = r2 + SOFT_EPS_SQ
            r      = np.sqrt(r2_eps)
            r3     = r * r2_eps

            f_scalar = G_GRAV * m64[j] / r3

            # Contribution to a_i.
            contrib_x = f_scalar * dx
            contrib_y = f_scalar * dy
            contrib_z = f_scalar * dz

            # Kahan compensated summation for each component.
            yk = contrib_x - comp_x;  tk = ax + yk;  comp_x = (tk - ax) - yk;  ax = tk
            yk = contrib_y - comp_y;  tk = ay + yk;  comp_y = (tk - ay) - yk;  ay = tk
            yk = contrib_z - comp_z;  tk = az + yk;  comp_z = (tk - az) - yk;  az = tk

        a[i, 0] = ax
        a[i, 1] = ay
        a[i, 2] = az

    return a


def potential_energy(x, m):
    """
    Total Plummer-softened gravitational potential energy:
        U = -G * sum_{i<j} m_i * m_j / sqrt(r_ij^2 + eps^2)

    Computed in f64 with Kahan summation over the i<j pairs.
    """
    N = x.shape[0]
    x64 = np.asarray(x, dtype=np.float64)
    m64 = np.asarray(m, dtype=np.float64)

    U      = 0.0
    comp_U = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x64[j, 0] - x64[i, 0]
            dy = x64[j, 1] - x64[i, 1]
            dz = x64[j, 2] - x64[i, 2]
            r2_eps = dx*dx + dy*dy + dz*dz + SOFT_EPS_SQ
            r      = np.sqrt(r2_eps)
            term   = -G_GRAV * m64[i] * m64[j] / r

            # Kahan
            yk = term - comp_U
            tk = U + yk
            comp_U = (tk - U) - yk
            U = tk
    return U


def kinetic_energy(v, m):
    """
    Total kinetic energy: K = sum_i 0.5 * m_i * |v_i|^2
    Done in f64 with Kahan summation.
    """
    N = v.shape[0]
    v64 = np.asarray(v, dtype=np.float64)
    m64 = np.asarray(m, dtype=np.float64)

    K, comp_K = 0.0, 0.0
    for i in range(N):
        term = 0.5 * m64[i] * (v64[i, 0]**2 + v64[i, 1]**2 + v64[i, 2]**2)
        yk = term - comp_K
        tk = K + yk
        comp_K = (tk - K) - yk
        K = tk
    return K


def run_leapfrog(bodies, n_steps, dt, energy_track=False):
    """
    Run leapfrog KDK in f64 with initial backward and final forward
    half-kick. Returns final state, all in f64.

    bodies : (N, 7) array, [x, y, z, vx, vy, vz, m]  (will be promoted to f64)
    n_steps: number of full integration steps
    dt     : timestep size (will be promoted to f64)

    Returns:
        x_final : (N, 3) f64
        v_final : (N, 3) f64, recentered on the final timestep
        a_final : (N, 3) f64, accelerations at the final timestep
        energy_hist : list of (step, K, U, E) tuples, or [] if not tracking
    """
    dt      = np.float64(dt)
    half_dt = dt * 0.5

    x = np.asarray(bodies[:, 0:3], dtype=np.float64).copy()
    v = np.asarray(bodies[:, 3:6], dtype=np.float64).copy()
    m = np.asarray(bodies[:, 6],   dtype=np.float64).copy()

    # Initial accelerations a_0
    print(f"  [validate] Computing initial accelerations a_0 ...")
    t0 = time.time()
    a  = compute_accelerations(x, m)
    print(f"  [validate]   {time.time() - t0:.2f}s")

    # Backward half kick: v_{-1/2} = v_0 - a_0 * dt/2
    v = v - a * half_dt

    energy_hist = []
    if energy_track:
        v0 = np.asarray(bodies[:, 3:6], dtype=np.float64)
        K0 = kinetic_energy(v0, m)
        U0 = potential_energy(x, m)
        energy_hist.append((0, K0, U0, K0 + U0))
        print(f"  [validate] Step 0: K={K0:.10e}, U={U0:.10e}, E={K0+U0:.10e}")

    # ---- Leapfrog main loop ----
    for n in range(n_steps):
        # Full kick: v_{n+1/2} = v_{n-1/2} + a_n * dt
        v = v + a * dt
        # Drift: x_{n+1} = x_n + v_{n+1/2} * dt
        x = x + v * dt
        # Recompute accelerations: a_{n+1}
        t0 = time.time()
        a = compute_accelerations(x, m)
        t_eval = time.time() - t0

        if energy_track:
            # To get integer-step v at step n+1: v_{n+1} = v_{n+1/2} + a_{n+1} * dt/2
            v_int = v + a * half_dt
            K = kinetic_energy(v_int, m)
            U = potential_energy(x, m)
            energy_hist.append((n + 1, K, U, K + U))
            print(f"  [validate] Step {n+1}: K={K:.10e}, U={U:.10e}, E={K+U:.10e}   "
                  f"(eval {t_eval:.2f}s)")

    # Forward half kick: v_N = v_{N-1/2} + a_N * dt/2
    v = v + a * half_dt

    return x, v, a, energy_hist


def main():
    parser = argparse.ArgumentParser(
        description="High-fidelity leapfrog reference for WSE-3 N-body validation."
    )
    parser.add_argument("--bodies", default="bodies.npy",
                        help="Input particles file, shape (N, 7).")
    parser.add_argument("--n_steps", type=int, required=True,
                        help="Number of leapfrog integration steps.")
    parser.add_argument("--dt", type=float, required=True,
                        help="Timestep size.")
    parser.add_argument("--output", default="reference.npy",
                        help="Output file with final state (N, 10): "
                             "[x, y, z, vx, vy, vz, m, fx, fy, fz].")
    parser.add_argument("--energy-track", action="store_true",
                        help="Track total energy after every step (very expensive).")
    args = parser.parse_args()

    bodies = np.load(args.bodies)
    N = bodies.shape[0]
    print(f"[validate] Loaded {N} bodies from {args.bodies}")
    print(f"[validate] Integrating: N_STEPS={args.n_steps}, DT={args.dt}")
    print(f"[validate] Mode: HIGH-FIDELITY f64 with Kahan summation and direct division")

    t0 = time.time()
    x_final, v_final, a_final, energy_hist = run_leapfrog(
        bodies, args.n_steps, args.dt, energy_track=args.energy_track
    )
    t1 = time.time()
    print(f"[validate] Reference leapfrog completed in {t1 - t0:.2f}s")

    # Compute forces F = m * a (in f64).
    m64 = np.asarray(bodies[:, 6], dtype=np.float64)
    forces_f64 = m64[:, None] * a_final

    # Cast to f32 ONLY for the saved output (device exports f32).
    x_f32 = x_final.astype(np.float32)
    v_f32 = v_final.astype(np.float32)
    m_f32 = m64.astype(np.float32)
    f_f32 = forces_f64.astype(np.float32)

    output = np.concatenate(
        [x_f32, v_f32, m_f32[:, None], f_f32], axis=1
    ).astype(np.float32)
    np.save(args.output, output)
    print(f"[validate] Saved final state to {args.output} (shape {output.shape})")

    if args.energy_track:
        eh = np.array(energy_hist, dtype=np.float64)   # (n_steps+1, 4)
        np.save("energy_history.npy", eh)
        print(f"[validate] Saved energy history to energy_history.npy")
        E0 = eh[0, 3]
        EN = eh[-1, 3]
        rel_drift = (EN - E0) / abs(E0) if E0 != 0 else 0.0
        print(f"[validate] Energy drift: E0={E0:.10e}, EN={EN:.10e}, "
              f"relative drift = {rel_drift:+.4e}")


if __name__ == "__main__":
    main()
