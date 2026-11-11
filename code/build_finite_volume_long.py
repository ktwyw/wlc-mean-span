#!/usr/bin/env python3
"""Finite-volume check of the spectral solution at lambda = 100.

The tensor grids used at lambda = 0.1, 1 and 10 are too costly at lambda = 100, where
the spatial domain is ten times longer.  The discretization error is taken to be
additive in the two grid parameters,

    F(Ny, Nu) = F + (spatial error)(Ny) + (angular error)(Nu),

which is tested on one off-axis grid.  The spatial series (Nu = 24, Ny = 3200 ... 51200)
is extrapolated with the observed order (Aitken, three finest grids) and, as a
sensitivity, with the integer-order model a/Ny + b/Ny^2; the angular series
(Ny = 1600, Nu = 24 ... 64) is extrapolated linearly in Nu^-2 over the three finest
grids (sensitivity: the two finest).  Output: data/finite_volume_long.csv (raw grid
values) and data/finite_volume_long_summary.json.  About one hour on two cores.
"""
from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from wlc_finite_volume import FVSettings, solve_fv

ROOT = Path(__file__).resolve().parents[1]
LAM = 100.0
NY_SERIES = (3200, 6400, 12800, 25600, 51200)      # at Nu = NU0
NU_SERIES = (24, 32, 48, 64)                        # at Ny = NY0
NU0, NY0 = 24, 1600
OFF_AXIS = (3200, 48)                               # additivity test
OUT = ROOT / "data" / "finite_volume_long.csv"


def evaluate(task):
    ny, nu = task
    return {"lambda": LAM, "Ny": ny, "Nu": nu, "Ymax": LAM, "F_raw": solve_fv(LAM, FVSettings(ny, nu, LAM))}


def read():
    if not OUT.exists():
        return {}
    return {(int(float(r["Ny"])), int(float(r["Nu"]))): float(r["F_raw"]) for r in csv.DictReader(OUT.open())}


def write(vals):
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lambda", "Ny", "Nu", "Ymax", "F_raw"])
        for (ny, nu), f in sorted(vals.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            w.writerow([repr(LAM), ny, nu, repr(LAM), repr(f)])


def main() -> None:
    vals = read()
    tasks = [(ny, NU0) for ny in NY_SERIES] + [(NY0, nu) for nu in NU_SERIES] + [OFF_AXIS, (OFF_AXIS[0], NU0)]
    tasks = sorted({t for t in tasks if t not in vals}, key=lambda t: -t[0] * t[1] ** 3)   # costliest first
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(evaluate, tasks):
            vals[(row["Ny"], row["Nu"])] = row["F_raw"]
            write(vals)
            print(f"Ny={row['Ny']} Nu={row['Nu']} F={row['F_raw']:.10f}", flush=True)

    with (ROOT / "data" / "wlc_mean_span_reference.csv").open() as fh:
        spectral = {float(r["lambda"]): float(r["F_continuum"]) for r in csv.DictReader(fh)}[LAM]

    # spatial extrapolation at Nu = NU0
    f = np.array([vals[(ny, NU0)] for ny in NY_SERIES])
    d = np.diff(f)
    orders = [float(np.log2(d[i] / d[i + 1])) for i in range(len(d) - 1)]
    aitken = float(f[-1] - d[-1] ** 2 / (d[-1] - d[-2]))                    # observed order, three finest
    x = 1.0 / np.array(NY_SERIES[-3:], float)
    integer = float(np.linalg.solve(np.vander(x, 3, increasing=True), f[-3:])[0])   # a/Ny + b/Ny^2
    # angular correction at Ny = NY0: F(Nu -> infinity) - F(NU0)
    g = np.array([vals[(NY0, nu)] for nu in NU_SERIES])
    z = np.array(NU_SERIES, float) ** -2
    ang3 = float(np.polyfit(z[-3:], g[-3:], 1)[1] - g[0])
    ang2 = float(np.polyfit(z[-2:], g[-2:], 1)[1] - g[0])
    # additivity: the angular increment 24 -> 48 on two spatial grids
    inc_axis = vals[(NY0, OFF_AXIS[1])] - vals[(NY0, NU0)]
    inc_off = vals[OFF_AXIS] - vals[(OFF_AXIS[0], NU0)]

    est_obs = aitken + ang3
    est_int = integer + ang3
    summary = {
        "lambda": LAM,
        "F_spectral": spectral,
        "spatial_series_Ny": list(NY_SERIES), "spatial_series_F_at_Nu24": [float(v) for v in f],
        "observed_spatial_orders": orders,
        "spatial_limit_observed_order": aitken, "spatial_limit_integer_order": integer,
        "angular_series_Nu": list(NU_SERIES), "angular_series_F_at_Ny1600": [float(v) for v in g],
        "angular_correction_three_finest": ang3, "angular_correction_two_finest": ang2,
        "angular_increment_24_to_48_at_Ny1600": float(inc_axis),
        "angular_increment_24_to_48_at_Ny3200": float(inc_off),
        "additivity_defect_rel": float((inc_off - inc_axis) / spectral),
        "F_fv_observed_order": est_obs, "rel_diff_observed_order": est_obs / spectral - 1.0,
        "F_fv_integer_order": est_int, "rel_diff_integer_order": est_int / spectral - 1.0,
        "angular_model_sensitivity_rel": float(abs(ang3 - ang2) / spectral),
    }
    (ROOT / "data" / "finite_volume_long_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
