#!/usr/bin/env python3
"""Milne displacement, small-s coefficient B, and sensitivity audits.

The stationary Milne problem contains a constant/linear Jordan sector near
zero.  The decaying subspace is separated with a declared threshold and the
threshold sensitivity is audited.  The next flexible-chain coefficient B is
obtained from positive-real-s transfer data.  A full sweep varies angular
order, fit window, polynomial degree, and the treatment of z_M.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eig, solve

from wlc_pn_transfer import angular_system, laplace_transform

Z_REPORTED = 0.71821164
B_REPORTED = 0.6468
B_REPORTED_HALF_RANGE = 0.0004
PRIMARY_S_MIN = 1.0e-4
PRIMARY_S_MAX = 3.0e-3
PRIMARY_N = 18
PRIMARY_DEGREE = 2


def _stationary_spectrum(m: int) -> tuple[np.ndarray, np.ndarray, object]:
    sys = angular_system(m)
    A0 = sys.R / sys.nodes[:, None]
    values, vectors = eig(A0, check_finite=False)
    return values, vectors, sys


def milne_length(m: int, zero_threshold: float = 1.0e-5) -> float:
    """Solve the stationary half-space matching problem at angular order m."""
    values, vectors, sys = _stationary_spectrum(m)
    mask = (np.real(values) < 0.0) & (np.abs(values) > zero_threshold)
    Z = vectors[:, mask]
    if Z.shape[1] != m // 2 - 1:
        raise RuntimeError(
            f"unexpected stationary stable dimension {Z.shape[1]} for m={m}, "
            f"threshold={zero_threshold:g}"
        )
    pos = sys.positive
    # At y=0: z_M-u + sum c_j g_j(u)=0 for u>0.
    matrix = np.column_stack([sys.h[pos], Z[pos, :]])
    rhs = sys.h[pos] * sys.nodes[pos]
    coefficient = solve(matrix, rhs, assume_a="gen", check_finite=False)
    return float(np.real(coefficient[0]))


def milne_threshold_diagnostics(m: int, threshold: float) -> dict[str, float | int]:
    values, vectors, sys = _stationary_spectrum(m)
    abs_values = np.abs(values)
    near = abs_values <= threshold
    mask = (np.real(values) < 0.0) & (abs_values > threshold)
    Z = vectors[:, mask]
    pos = sys.positive
    if Z.shape[1] != m // 2 - 1:
        z_value = float("nan")
    else:
        matrix = np.column_stack([sys.h[pos], Z[pos, :]])
        rhs = sys.h[pos] * sys.nodes[pos]
        coefficient = solve(matrix, rhs, assume_a="gen", check_finite=False)
        z_value = float(np.real(coefficient[0]))
    retained_abs = abs_values[mask]
    near_abs = abs_values[near]
    return {
        "m": m,
        "zero_threshold": threshold,
        "z_M": z_value,
        "near_zero_count": int(np.count_nonzero(near)),
        "near_zero_max_abs": float(np.max(near_abs)) if near_abs.size else float("nan"),
        "first_retained_decay_abs": float(np.min(retained_abs)) if retained_abs.size else float("nan"),
        "stationary_stable_dimension": int(Z.shape[1]),
    }


def b_estimates_for_grid(m: int, z_value: float, s_values: np.ndarray) -> np.ndarray:
    estimates = []
    for s in s_values:
        fhat = float(np.real(laplace_transform(complex(s), m, method="eig")))
        estimate = (
            fhat
            - 2.0 / (math.sqrt(3.0) * s**1.5)
            + 2.0 * z_value / s
        ) * math.sqrt(s) / math.sqrt(math.pi)
        estimates.append(estimate)
    return np.asarray(estimates, dtype=float)


def fit_intercept(s_values: np.ndarray, b_values: np.ndarray, degree: int) -> float:
    x = np.sqrt(s_values)
    polynomial = np.polyfit(x, b_values, degree)
    return float(polynomial[-1])


def primary_B(m: int, z_value: float) -> tuple[float, float]:
    s_values = np.logspace(math.log10(PRIMARY_S_MIN), math.log10(PRIMARY_S_MAX), PRIMARY_N)
    b_values = b_estimates_for_grid(m, z_value, s_values)
    central = fit_intercept(s_values, b_values, PRIMARY_DEGREE)
    linear = fit_intercept(s_values, b_values, 1)
    return central, abs(central - linear)


def sensitivity_sweep() -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    orders = (64, 96, 128, 160)
    windows = (
        (5.0e-5, 1.5e-3),
        (5.0e-5, 3.0e-3),
        (1.0e-4, 1.5e-3),
        (1.0e-4, 3.0e-3),
        (1.0e-4, 5.0e-3),
        (2.0e-4, 3.0e-3),
        (2.0e-4, 5.0e-3),
    )
    s_master = np.logspace(math.log10(3.0e-5), math.log10(5.0e-3), 61)
    rows: list[dict[str, float | int | str]] = []
    for m in orders:
        z_order = milne_length(m)
        # Evaluate the expensive transform only once per order.  Reconstruct
        # B(s) for either the per-order or continuum Milne displacement.
        fhat = np.array(
            [float(np.real(laplace_transform(complex(s), m, method="eig"))) for s in s_master],
            dtype=float,
        )
        for z_treatment, z_value in (("per_order", z_order), ("reported_continuum", Z_REPORTED)):
            b_master = (
                fhat
                - 2.0 / (math.sqrt(3.0) * s_master**1.5)
                + 2.0 * z_value / s_master
            ) * np.sqrt(s_master) / math.sqrt(math.pi)
            for degree in (1, 2, 3):
                for s_min, s_max in windows:
                    mask = (s_master >= s_min) & (s_master <= s_max)
                    value = fit_intercept(s_master[mask], b_master[mask], degree)
                    rows.append(
                        {
                            "m": m,
                            "z_treatment": z_treatment,
                            "z_value": z_value,
                            "polynomial_degree": degree,
                            "s_min": s_min,
                            "s_max": s_max,
                            "n_points": int(np.count_nonzero(mask)),
                            "B_intercept": value,
                            "deviation_from_reported": value - B_REPORTED,
                        }
                    )

    # Conservative, numerically stable subset used to audit the reported
    # parenthetical uncertainty.  Very small s is excluded because subtracting
    # the s^{-3/2} and s^{-1} terms becomes cancellation limited in double
    # precision at finite angular order.
    stable = [
        row
        for row in rows
        if int(row["m"]) >= 96
        and int(row["polynomial_degree"]) <= 2
        and float(row["s_min"]) >= 1.0e-4
        and float(row["s_max"]) <= 5.0e-3
    ]
    values = np.array([float(row["B_intercept"]) for row in stable])
    computed_half_range = float(np.max(np.abs(values - B_REPORTED)))
    summary = {
        "reported_B": B_REPORTED,
        "reported_half_range": B_REPORTED_HALF_RANGE,
        "stable_audit_min": float(np.min(values)),
        "stable_audit_max": float(np.max(values)),
        "stable_audit_median": float(np.median(values)),
        "stable_audit_computed_half_range": computed_half_range,
        "stable_audit_count": int(values.size),
        "primary_s_min": PRIMARY_S_MIN,
        "primary_s_max": PRIMARY_S_MAX,
        "primary_polynomial_degree": PRIMARY_DEGREE,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=(Path(__file__).resolve().parents[1] / "data"))
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    convergence_rows: list[dict[str, float | int]] = []
    for m in (8, 16, 32, 64, 96, 128, 160, 192, 256):
        z_value = milne_length(m)
        b_value = float("nan")
        degree_sensitivity = float("nan")
        if m in (32, 64, 96, 128, 160):
            b_value, degree_sensitivity = primary_B(m, z_value)
        convergence_rows.append(
            {
                "m": m,
                "z_M": z_value,
                "B": b_value,
                "B_fit_degree_sensitivity": degree_sensitivity,
            }
        )
        print(f"m={m:3d} z_M={z_value:.10f} B={b_value:.8f}")
    write_csv(out / "milne_and_B_convergence.csv", convergence_rows)

    sweep_rows, summary = sensitivity_sweep()
    write_csv(out / "B_sensitivity_sweep.csv", sweep_rows)
    with (out / "B_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    threshold_rows: list[dict[str, float | int]] = []
    for m in (64, 128, 256):
        for threshold in (1.0e-7, 3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4):
            threshold_rows.append(milne_threshold_diagnostics(m, threshold))
    write_csv(out / "milne_zero_mode_threshold.csv", threshold_rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote Milne/B audits to {out}")


if __name__ == "__main__":
    main()
