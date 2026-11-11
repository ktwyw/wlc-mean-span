#!/usr/bin/env python3
"""Exact test of the production inversion over the full range of lambda.

The telegraph (P1, m = 2) problem has the closed-form solution
F(t) = (2/sqrt 3)[e^{-t/2}(1+t) I0(t/2) + t e^{-t/2} I1(t/2) - 1]
and the analytic transform ``telegraph_p1_transform``.  This script inverts
(i) the analytic transform with the production series (A, N, K) = (12, 32, 32)
with the first alias removed, and (ii) the m = 2 transform of the production
solver through ``invert_alias_corrected``, at all 67 reference lengths.  The
error is compared with the predicted remaining alias
-e^{-2A}[F(9t) - F(5t)].
Output: validation/telegraph_inversion_test.csv
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv
from math import comb, exp, pi
from pathlib import Path

import mpmath as mp
import numpy as np

from build_reference_curve import LAMBDA_NODES, PRIMARY
from wlc_pn_transfer import invert_alias_corrected, telegraph_p1_transform

ROOT = Path(__file__).resolve().parents[1]


def exact(t: float) -> float:
    mp.mp.dps = 40
    x = mp.mpf(t) / 2
    e = mp.exp(-x)
    return float(2 / mp.sqrt(3) * (e * (1 + 2 * x) * mp.besseli(0, x) + 2 * x * e * mp.besseli(1, x) - 1))


def euler_series(transform, t: float, A: float, N: int, K: int) -> float:
    j = np.arange(N + K + 1)
    s = (A + 2j * pi * j) / (2 * t)
    b = (-1.0) ** j * np.array([transform(z).real for z in s])
    partial = np.cumsum(b) - 0.5 * b[0]
    w = np.array([comb(K, k) for k in range(K + 1)], float) / 2.0**K
    return float(exp(A / 2) / t * np.dot(w, partial[N:N + K + 1]))


def main() -> None:
    A, N, K = PRIMARY.A, PRIMARY.N, PRIMARY.K
    rows = []
    for t in LAMBDA_NODES:
        F = exact(t)
        analytic = euler_series(telegraph_p1_transform, t, A, N, K) - exp(-A) * euler_series(telegraph_p1_transform, 3 * t, A, N, K)
        solver = invert_alias_corrected(t, 2, PRIMARY)
        predicted = -exp(-2 * A) * (exact(9 * t) - exact(5 * t))
        rows.append({"lambda": repr(float(t)), "F_exact": repr(F),
                     "rel_error_analytic_transform": repr(analytic / F - 1),
                     "rel_error_solver_m2": repr(solver / F - 1),
                     "predicted_rel_residual_alias": repr(predicted / F)})
    out = ROOT / "validation" / "telegraph_inversion_test.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ea = np.array([float(r["rel_error_analytic_transform"]) for r in rows])
    es = np.array([float(r["rel_error_solver_m2"]) for r in rows])
    pr = np.array([float(r["predicted_rel_residual_alias"]) for r in rows])
    print(f"analytic: {ea.min():.3e} .. {ea.max():.3e}; solver m=2: {es.min():.3e} .. {es.max():.3e}; "
          f"predicted: {pr.min():.3e} .. {pr.max():.3e}; max |analytic-predicted| = {np.max(abs(ea-pr)):.2e}; "
          f"max |solver-predicted| = {np.max(abs(es-pr)):.2e}")


if __name__ == "__main__":
    main()
