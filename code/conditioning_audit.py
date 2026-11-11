#!/usr/bin/env python3
"""Audit the stable-subspace dimension, conditioning and residual at every production transform point.

Every Laplace variable used to build the reference table is audited: for each of
the 67 nodes lambda, the primary alias-corrected inversion samples
s_j = (A + 2 pi i j)/(2t), j = 0..N+K, at t = lambda and at the companion length
t = 3 lambda, with (A, N, K) = (12, 32, 32) at the orders m = 32, 48, 64, 96 and
the withheld order 128; the inversion check uses (14, 36, 36) at m = 64.  Lower
orders 8, 16, 24 are audited on the primary points for completeness.  For each
order the script records the minimum, median and maximum condition number of
E_+ Z_- in the orthonormal Schur basis, the largest invariant-subspace residual,
and the range of the stable dimension (which must equal m/2).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from build_reference_curve import INVERSION_CHECK, LAMBDA_NODES, PRIMARY
from wlc_pn_transfer import laplace_transform


def points(settings) -> list[complex]:
    out = []
    for lam in LAMBDA_NODES:
        for t in (float(lam), 3.0 * float(lam)):
            for j in range(settings.N + settings.K + 1):
                out.append((settings.A + 2.0 * math.pi * 1j * j) / (2.0 * t))
    return out


def audit(task: tuple[int, str]) -> dict[str, float | str]:
    m, which = task
    pts = points(PRIMARY if which == "primary" else INVERSION_CHECK)
    conds, res, dims = [], [], []
    for s in pts:
        _, d = laplace_transform(s, m, diagnostics=True)
        conds.append(d.incoming_condition_number)
        res.append(d.invariant_residual)
        dims.append(d.stable_dimension)
    return {
        "m": m,
        "setting": which,
        "n_points": len(pts),
        "condition_min": float(np.min(conds)),
        "condition_median": float(np.median(conds)),
        "condition_max": float(np.max(conds)),
        "residual_max": float(np.max(res)),
        "stable_dimension_min": int(np.min(dims)),
        "stable_dimension_max": int(np.max(dims)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "conditioning_audit_full.csv"))
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    tasks = [(m, "primary") for m in (8, 16, 24, 32, 48, 64, 96, 128)] + [(64, "check")]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(audit, tasks))
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("max condition number over all production points:", max(r["condition_max"] for r in rows))


if __name__ == "__main__":
    main()
