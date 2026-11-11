#!/usr/bin/env python3
"""Generate the complete continuum-extrapolated WLC mean-span reference table.

The script evaluates m=32,48,64,96 finite-P_N values at the 67 declared nodes
(ordered-Schur stable subspace; the eigendecomposition route is kept as a
regression check in solver_regression_audit.py), inverts each transform with
the alias-corrected Euler--Bromwich series (A=12, N=K=32), performs the stated
m^{-3}+m^{-4} continuum extrapolation, evaluates two alternative
extrapolations, and checks the inversion at m=64 with a second setting
(A=14, N=K=36, also alias-corrected).  A fifth order, m=128, is evaluated as a
check: it is not used in the extrapolation, but its distance from the
extrapolated value is one of the five components of the internal sensitivity.
The resulting CSV contains every finite-order value and the complete internal
sensitivity calculation.

Revision note (2026-10-29): the previous table used the uncorrected series with
A=22.  Its factor exp(A/2)~6e4 amplified floating-point noise in the transform
at small |s| from a few 1e-8 at lambda = 1e3 to nearly 1e-6 for lambda >= 5e4 (see
validation/inversion_noise_audit.csv); the alias-corrected A=12 series reduces
the prefactor by exp(5)~150.  The remaining alias is -exp(-2A)[F(9 lam)-F(5 lam)],
negative and below 4 exp(-2A) F < 1.6e-10 F in magnitude (exact telegraph test:
-1.5e-10..-3e-11 relative; validation/telegraph_inversion_test.csv).
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

# Prevent dense linear-algebra libraries from oversubscribing worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from wlc_pn_transfer import BromwichSettings, invert_alias_corrected

LAMBDA_NODES = np.array(
    [
        0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075,
        0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9,
        1.0, 1.25, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0,
        50.0, 60.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0,
        300.0, 400.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0,
        3000.0, 5000.0, 6000.0, 7500.0, 10000.0, 15000.0, 20000.0,
        30000.0, 50000.0, 75000.0, 100000.0,
    ],
    dtype=float,
)
M_VALUES = (32, 48, 64, 96)
M_WITHHELD = 128
PRIMARY = BromwichSettings(12.0, 32, 32)          # alias-corrected
METHOD = "schur"   # ordered Schur factorization for the stable subspace; "eig" is the regression check
INVERSION_CHECK = BromwichSettings(14.0, 36, 36)  # alias-corrected


def continuum_fit(m_values: Iterable[int], f_values: Iterable[float], two_terms: bool = True) -> tuple[float, np.ndarray]:
    m = np.asarray(list(m_values), dtype=float)
    f = np.asarray(list(f_values), dtype=float)
    if two_terms:
        X = np.column_stack([np.ones_like(m), m**-3, m**-4])
    else:
        X = np.column_stack([np.ones_like(m), m**-3])
    beta, *_ = np.linalg.lstsq(X, f, rcond=None)
    return float(beta[0]), beta


def evaluate_node(lam: float) -> dict[str, float]:
    vals = {m: invert_alias_corrected(lam, m, PRIMARY, method=METHOD) for m in M_VALUES}
    f128 = invert_alias_corrected(lam, M_WITHHELD, PRIMARY, method=METHOD)
    f_inf, _ = continuum_fit(M_VALUES, [vals[m] for m in M_VALUES], two_terms=True)
    f_alt3, _ = continuum_fit(M_VALUES[-3:], [vals[m] for m in M_VALUES[-3:]], two_terms=True)
    f_alt2, _ = continuum_fit(M_VALUES[-2:], [vals[m] for m in M_VALUES[-2:]], two_terms=False)
    f_inv64 = invert_alias_corrected(lam, 64, INVERSION_CHECK, method=METHOD)
    delta_inv64 = f_inv64 - vals[64]
    eps_abs = max(
        abs(f_inf - f_alt3),
        abs(f_inf - f_alt2),
        abs(vals[96] - f_inf),
        abs(delta_inv64),
        abs(f128 - f_inf),
    )
    return {
        "lambda": float(lam),
        **{f"F_m{m}": float(vals[m]) for m in M_VALUES},
        "F_continuum": float(f_inf),
        "F_alt_last3": float(f_alt3),
        "F_alt_last2": float(f_alt2),
        "F_m64_inversion_check": float(f_inv64),
        "delta_inversion_m64": float(delta_inv64),
        "epsilon_internal_abs": float(eps_abs),
        "epsilon_internal_rel": float(eps_abs / abs(f_inf)),
        "F_m128_check": float(f128),
        "delta_m128": float(f128 - f_inf),
    }


def _load_existing(path: Path) -> dict[float, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {float(r["lambda"]): {k: float(v) for k, v in r.items()} for r in csv.DictReader(f)}


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    rows = sorted(rows, key=lambda r: r["lambda"])
    fields = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "wlc_mean_span_reference.csv"))
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--fresh", action="store_true", help="ignore an existing partial output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = {} if args.fresh else _load_existing(output)
    pending = [float(x) for x in LAMBDA_NODES if float(x) not in existing]
    rows = list(existing.values())
    print(f"Existing nodes: {len(rows)}; pending: {len(pending)}", flush=True)

    if args.workers == 1:
        for i, lam in enumerate(pending, 1):
            row = evaluate_node(lam)
            rows.append(row)
            _write_rows(output, rows)
            print(f"[{i:02d}/{len(pending)}] lambda={lam:g} F={row['F_continuum']:.10g}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(evaluate_node, lam): lam for lam in pending}
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                _write_rows(output, rows)
                done += 1
                print(f"[{done:02d}/{len(pending)}] lambda={row['lambda']:g} F={row['F_continuum']:.10g}", flush=True)
    if rows:
        _write_rows(output, rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
