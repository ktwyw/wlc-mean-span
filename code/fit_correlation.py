#!/usr/bin/env python3
"""Fit and audit the asymptotically constrained [3/4] rational correlation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

A_G = 4.0 / math.sqrt(3.0 * math.pi)
Z_M = 0.71821164
C2 = 37.0 / 768.0
B_CENTRAL = 0.6468
B_HALF_RANGE = 0.0004


def coefficients_from_bc(b: float, c: float, B: float = B_CENTRAL) -> dict[str, float]:
    d = b + 2.0 * C2
    f = c / (2.0 * A_G)
    e = b / (2.0 * A_G) + Z_M * c / (A_G * A_G)
    if f <= 0:
        raise ValueError("c must be positive")
    a = 2.0 * B * f + b * e / f - c * e * e / (f * f) + c * d / f
    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "B": B}


def rational(lam: np.ndarray, coeff: dict[str, float]) -> np.ndarray:
    w = np.sqrt(np.asarray(lam, dtype=float))
    a, b, c, d, e, f = (coeff[k] for k in ("a", "b", "c", "d", "e", "f"))
    return 0.5 * w**2 * (1 + a*w + b*w**2 + c*w**3) / (1 + a*w + d*w**2 + e*w**3 + f*w**4)


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    # Twenty-four same-solver holdout nodes spread across the entire interval;
    # endpoints remain in the calibration set.
    candidates = np.arange(1, n - 1)
    target = np.linspace(0, len(candidates) - 1, 24)
    hold = np.unique(candidates[np.rint(target).astype(int)])
    # Guarantee exactly 24 unique entries in the unlikely event of collisions.
    if len(hold) < 24:
        extras = [i for i in candidates if i not in set(hold)]
        hold = np.sort(np.concatenate([hold, extras[: 24 - len(hold)]]))
    hold = np.sort(hold[:24])
    cal = np.array([i for i in range(n) if i not in set(hold)], dtype=int)
    return cal, hold


def error_metrics(lam: np.ndarray, ref: np.ndarray, coeff: dict[str, float]) -> dict[str, float]:
    err = 100.0 * (rational(lam, coeff) / ref - 1.0)
    k = int(np.argmax(np.abs(err)))
    return {
        "n": int(len(lam)),
        "max_abs_rel_error_pct": float(abs(err[k])),
        "lambda_at_max": float(lam[k]),
        "signed_error_at_max_pct": float(err[k]),
        "rms_rel_error_pct": float(np.sqrt(np.mean(err**2))),
        "mean_abs_rel_error_pct": float(np.mean(np.abs(err))),
        "min_rel_error_pct": float(np.min(err)),
        "max_rel_error_pct": float(np.max(err)),
    }


def fit(lam: np.ndarray, ref: np.ndarray, B: float = B_CENTRAL) -> dict[str, float]:
    def objective(x: np.ndarray) -> float:
        b, c = x
        if b <= 0 or c <= 0:
            return 1e6
        try:
            coeff = coefficients_from_bc(float(b), float(c), B)
            den = 1 + coeff["a"]*np.sqrt(lam) + coeff["d"]*lam + coeff["e"]*lam**1.5 + coeff["f"]*lam**2
            if np.any(den <= 0):
                return 1e6
            return float(np.max(np.abs(rational(lam, coeff)/ref - 1.0)))
        except Exception:
            return 1e6

    de = differential_evolution(objective, bounds=[(0.2, 0.8), (0.08, 0.5)], tol=1e-10, polish=False, seed=20260926, workers=1)
    local = minimize(objective, de.x, method="Nelder-Mead", options={"xatol": 1e-13, "fatol": 1e-13, "maxiter": 20000})
    return coefficients_from_bc(float(local.x[0]), float(local.x[1]), B)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "wlc_mean_span_reference.csv"))
    p.add_argument("--out-dir", type=Path, default=(Path(__file__).resolve().parents[1] / "data"))
    args = p.parse_args()
    rows = list(csv.DictReader(args.reference.open()))
    lam = np.array([float(r["lambda"]) for r in rows])
    ref = np.array([float(r["F_continuum"]) for r in rows])
    cal_idx, hold_idx = split_indices(len(lam))

    coeff = fit(lam[cal_idx], ref[cal_idx], B_CENTRAL)
    coeff_lo = fit(lam[cal_idx], ref[cal_idx], B_CENTRAL - B_HALF_RANGE)
    coeff_hi = fit(lam[cal_idx], ref[cal_idx], B_CENTRAL + B_HALF_RANGE)

    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out / "rational_fit_coefficients.json").open("w") as f:
        json.dump({"central": coeff, "B_minus": coeff_lo, "B_plus": coeff_hi}, f, indent=2)

    with (out / "reference_nodes_with_split.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "F_continuum", "set", "F_rational", "relative_error_pct"])
        hold_set = set(hold_idx.tolist())
        fitvals = rational(lam, coeff)
        for i, (x, y, z) in enumerate(zip(lam, ref, fitvals)):
            w.writerow([x, y, "holdout" if i in hold_set else "calibration", z, 100*(z/y-1)])

    metrics = {
        "calibration": error_metrics(lam[cal_idx], ref[cal_idx], coeff),
        "holdout": error_metrics(lam[hold_idx], ref[hold_idx], coeff),
        "all": error_metrics(lam, ref, coeff),
        "crossover_0p1_10": error_metrics(lam[(lam>=0.1)&(lam<=10)], ref[(lam>=0.1)&(lam<=10)], coeff),
        "B_minus_all": error_metrics(lam, ref, coeff_lo),
        "B_plus_all": error_metrics(lam, ref, coeff_hi),
    }
    with (out / "rational_fit_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"coefficients": coeff, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
