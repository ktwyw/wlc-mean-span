#!/usr/bin/env python3
"""Evaluate withheld high-angular-order spectral spot checks.

The primary continuum fit uses m={32,48,64,96}.  This audit evaluates later
orders at representative contour lengths without using them in the fit, and
(when three or more orders are requested) refits F_inf + c3 m^-3 + c4 m^-4 to
the high orders alone as an independent extrapolation.  The production solver
(ordered Schur) is used; see build_reference_curve.METHOD.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from wlc_pn_transfer import invert_alias_corrected
from build_reference_curve import METHOD, PRIMARY, continuum_fit

DEFAULT_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "wlc_mean_span_reference.csv"))
    parser.add_argument("--output", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "high_order_spot_checks.csv"))
    parser.add_argument("--m-values", type=int, nargs="+", default=[128, 192, 256])
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    args = parser.parse_args()

    with args.reference.resolve().open(newline="") as handle:
        reference = {float(r["lambda"]): float(r["F_continuum"]) for r in csv.DictReader(handle)}

    rows: list[dict[str, float]] = []
    for lam in args.lambdas:
        if lam not in reference:
            raise KeyError(f"lambda={lam:g} is not present in the reference table")
        row: dict[str, float] = {"lambda": lam, "F_continuum": reference[lam]}
        for m in args.m_values:
            value = invert_alias_corrected(lam, m, PRIMARY, method=METHOD)
            row[f"F_m{m}"] = value
            row[f"relative_difference_m{m}_pct"] = 100.0 * (value / reference[lam] - 1.0)
            print(f"lambda={lam:g} m={m} F={value:.12g}", flush=True)
        if len(args.m_values) >= 3:
            f_high, _ = continuum_fit(args.m_values, [row[f"F_m{m}"] for m in args.m_values], two_terms=True)
            row["F_high_order_extrapolation"] = f_high
            row["relative_difference_extrapolation_pct"] = 100.0 * (f_high / reference[lam] - 1.0)
            for m in args.m_values:
                row[f"scaled_residual_m{m}"] = (row[f"F_m{m}"] - reference[lam]) * m**3
            print(f"lambda={lam:g} high-order extrapolation={f_high:.12g} rel={row['relative_difference_extrapolation_pct']:.3e}%", flush=True)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
