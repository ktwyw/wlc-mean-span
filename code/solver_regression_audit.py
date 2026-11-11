#!/usr/bin/env python3
"""Cross-check equivalent finite-P_N half-space solver formulations.

For positive-real Laplace variables, compare the nodal transport eigensolve,
the ordered-Schur stable subspace, and the real symmetric moment pencil.  For
complex right-half-plane Bromwich samples, compare nodal eigensolve and Schur
forms.  These checks are regression diagnostics, not continuum-error bounds.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from wlc_pn_transfer import laplace_transform, laplace_transform_symmetric_pencil


def rel(a: complex, b: complex) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), 1.0e-300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "solver_regression_audit.csv"))
    args = parser.parse_args()
    rows: list[dict[str, float | int | str]] = []

    for m in (8, 16, 32, 64, 96, 128):
        for s in (0.01, 0.1, 1.0, 10.0):
            eig_value = laplace_transform(complex(s), m, method="eig")
            schur_value = laplace_transform(complex(s), m, method="schur")
            symmetric_value = laplace_transform_symmetric_pencil(s, m)
            rows.append(
                {
                    "m": m,
                    "sample_type": "positive_real",
                    "s_real": s,
                    "s_imag": 0.0,
                    "rel_eig_schur": rel(eig_value, schur_value),
                    "rel_eig_symmetric": rel(eig_value, symmetric_value),
                    "rel_schur_symmetric": rel(schur_value, symmetric_value),
                }
            )

    # Representative complex samples from the Euler-Bromwich contour.
    for m in (16, 32, 64, 96, 128):
        for lam in (0.01, 0.1, 1.0, 10.0, 100.0, 1.0e4, 1.0e5):
            for j in (0, 4, 12, 28, 64):
                s = (12.0 + 2.0 * math.pi * 1j * j) / (2.0 * lam)
                eig_value = laplace_transform(s, m, method="eig")
                schur_value = laplace_transform(s, m, method="schur")
                rows.append(
                    {
                        "m": m,
                        "sample_type": "complex_bromwich",
                        "s_real": float(s.real),
                        "s_imag": float(s.imag),
                        "rel_eig_schur": rel(eig_value, schur_value),
                        "rel_eig_symmetric": float("nan"),
                        "rel_schur_symmetric": float("nan"),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print("maximum eig-Schur relative discrepancy:", max(float(r["rel_eig_schur"]) for r in rows))
    real_rows = [r for r in rows if r["sample_type"] == "positive_real"]
    print("maximum eig-symmetric relative discrepancy:", max(float(r["rel_eig_symmetric"]) for r in real_rows))


if __name__ == "__main__":
    main()
