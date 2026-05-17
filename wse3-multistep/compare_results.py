#!/usr/bin/env python3
"""
Confronta il device output (device_final.npy salvato da run.py) con la
reference Python (reference.npy salvato da validate_multistep.py).

Entrambi i file hanno la stessa struttura (N, 10):
    [x, y, z, vx, vy, vz, m, fx, fy, fz]

Stampa errori relativi su posizioni, velocità, forze.
"""
import argparse
import sys

import numpy as np


def rel_err_per_body(a, b):
    """Errore relativo per body (norma 2 della differenza / norma 2 del riferimento)."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    norm_b = np.linalg.norm(b.astype(np.float64), axis=1)
    norm_diff = np.linalg.norm(diff, axis=1)
    err = np.where(norm_b > 1e-12, norm_diff / np.where(norm_b > 1e-12, norm_b, 1.0), 0.0)
    return err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    default="device_final.npy")
    parser.add_argument("--reference", default="reference.npy")
    parser.add_argument("--tol",       type=float, default=1e-3,
                        help="Tolerance for pass/fail")
    parser.add_argument("--show-worst", type=int, default=5,
                        help="Show worst N bodies if validation fails")
    args = parser.parse_args()

    dev = np.load(args.device)
    ref = np.load(args.reference)

    if dev.shape != ref.shape:
        print(f"ERROR: shapes differ: device={dev.shape}, reference={ref.shape}")
        sys.exit(1)

    print(f"Comparing device ({args.device}) vs reference ({args.reference})")
    print(f"  shape: {dev.shape}")
    print()

    dev_x = dev[:, 0:3];  ref_x = ref[:, 0:3]
    dev_v = dev[:, 3:6];  ref_v = ref[:, 3:6]
    dev_f = dev[:, 7:10]; ref_f = ref[:, 7:10]

    err_x = rel_err_per_body(dev_x, ref_x)
    err_v = rel_err_per_body(dev_v, ref_v)
    err_f = rel_err_per_body(dev_f, ref_f)

    print(f"--- VALIDATION ---")
    print(f"  Positions   : max rel err = {err_x.max():.4e}   avg = {err_x.mean():.4e}")
    print(f"  Velocities  : max rel err = {err_v.max():.4e}   avg = {err_v.mean():.4e}")
    print(f"  Forces      : max rel err = {err_f.max():.4e}   avg = {err_f.mean():.4e}")

    max_overall = max(err_x.max(), err_v.max(), err_f.max())

    if max_overall < args.tol:
        print(f"\n  ✅ Validation PASSED (worst err {max_overall:.4e} < tol {args.tol:.0e})")
        sys.exit(0)
    else:
        print(f"\n  ❌ Validation FAILED (worst err {max_overall:.4e} >= tol {args.tol:.0e})")
        for name, dev_arr, ref_arr, err_arr in [
            ("positions",  dev_x, ref_x, err_x),
            ("velocities", dev_v, ref_v, err_v),
            ("forces",     dev_f, ref_f, err_f),
        ]:
            worst = np.argsort(err_arr)[-args.show_worst:][::-1]
            print(f"\n  Worst {name}:")
            for g in worst:
                print(f"    body {g}: dev={dev_arr[g]}, ref={ref_arr[g]}, |err|={err_arr[g]:.4e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
