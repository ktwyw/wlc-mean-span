#!/usr/bin/env python3
"""Observed convergence order of the finite-volume solution and free-order extrapolation.

Reads the archived tensor grid (data/finite_volume_extended_tensor_grid.csv:
N_y = 200, 400, 800, 1600; N_u = 24, 32, 48, 64; lambda = 0.1, 1, 10) and
reports, for each lambda and N_u,

* the observed order p in N_y from three successive doublings,
  p = log2[(F_{N} - F_{N/2}) / (F_{2N} - F_{N})];
* a free-order (Aitken) spatial extrapolation from the three finest N_y,
  F_inf = F_3 - (F_3 - F_2)^2 / ((F_3 - F_2) - (F_2 - F_1));
* a second free-order variant that extrapolates the drift of the observed
  order by one doubling, p_3 = 2 p_2 - p_1, and sums the remaining
  increments at that order (extrapolation-model sensitivity);

then extrapolates the spatial limits in N_u^{-2} (linear fit on the three
largest N_u, as in the primary analysis) and compares with the spectral
reference values.  Writes data/finite_volume_order_analysis.csv.

The integer-power extrapolation used in the primary analysis (quadratic in
1/N_y) assumes first-order convergence; the observed orders at lambda = 1 and 10
are below one, which leaves a positive bias in that extrapolation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data" / "finite_volume_extended_tensor_grid.csv"
REF = ROOT / "data" / "wlc_mean_span_reference.csv"
OUT = ROOT / "data" / "finite_volume_order_analysis.csv"


def main() -> None:
    rows = list(csv.DictReader(GRID.open()))
    ref = {float(r["lambda"]): float(r["F_continuum"]) for r in csv.DictReader(REF.open())}
    out = []
    for lam in sorted({float(r["lambda"]) for r in rows}):
        nus = sorted({int(float(r["Nu"])) for r in rows if float(r["lambda"]) == lam})
        spatial = {}
        trend = {}
        orders = {}
        for nu in nus:
            sub = sorted((int(float(r["Ny"])), float(r["F_raw"])) for r in rows
                         if float(r["lambda"]) == lam and int(float(r["Nu"])) == nu)
            ny = np.array([a for a, _ in sub], float)
            f = np.array([b for _, b in sub], float)
            d = np.diff(f)
            orders[nu] = [float(np.log2(d[i] / d[i + 1])) for i in range(len(d) - 1)]
            f1, f2, f3 = f[-3:]
            spatial[nu] = f3 - (f3 - f2) ** 2 / ((f3 - f2) - (f2 - f1))
            p1, p2 = np.log2(d[-3] / d[-2]), np.log2(d[-2] / d[-1])
            r3 = 2.0 ** -(2.0 * p2 - p1)
            trend[nu] = f[-1] + d[-1] * r3 / (1.0 - r3)
        nu_arr = np.array(nus[-3:], float)
        sp = np.array([spatial[n] for n in nus[-3:]])
        coef = np.polyfit(nu_arr ** -2.0, sp, 1)
        f_inf = float(coef[1])
        f_trend = float(np.polyfit(nu_arr ** -2.0, np.array([trend[n] for n in nus[-3:]]), 1)[1])
        # angular sensitivity: change when the angular fit uses the three smallest N_u instead
        coef_lo = np.polyfit(np.array(nus[:3], float) ** -2.0, np.array([spatial[n] for n in nus[:3]]), 1)
        out.append({
            "lambda": lam,
            "observed_order_Ny_min": min(min(v) for v in orders.values()),
            "observed_order_Ny_max": max(max(v) for v in orders.values()),
            "observed_orders_at_Nu64": ";".join(f"{x:.3f}" for x in orders[nus[-1]]),
            "F_FV_free_order": f_inf,
            "F_spectral": ref[lam],
            "relative_difference_pct": 100.0 * (f_inf / ref[lam] - 1.0),
            "angular_fit_change_pct": 100.0 * (float(coef_lo[1]) - f_inf) / ref[lam],
            "F_FV_order_trend": f_trend,
            "relative_difference_order_trend_pct": 100.0 * (f_trend / ref[lam] - 1.0),
            "spatial_model_change_pct": 100.0 * (f_trend - f_inf) / ref[lam],
        })
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
