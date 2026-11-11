#!/usr/bin/env python3
"""Evaluate the archived practical formula, the practical formula, on all 67 reference nodes.

The coefficients in data/rational_fit_coefficients.json were fitted once by
code/fit_correlation.py on 34 calibration nodes chosen from the 58 primary nodes
(0.01 <= lambda <= 5000); 24 interleaved primary nodes were withheld.  The nine
nodes above 5000 were added later and lie outside the calibration range.  This
script does not refit.  It labels every node as 'calibration', 'holdout'
(interleaved) or 'extrapolation', evaluates the formula with the archived
coefficients rounded to the eight decimals printed in the practical formula (the values a
reader uses) against the current reference table, and writes

* data/reference_nodes_with_split.csv  (67 rows)
* data/rational_fit_metrics.json       (per-set error metrics)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fit_correlation import B_CENTRAL, B_HALF_RANGE, error_metrics, fit, rational, split_indices

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    rows = list(csv.DictReader((DATA / "wlc_mean_span_reference.csv").open()))
    lam = np.array([float(r["lambda"]) for r in rows])
    ref = np.array([float(r["F_continuum"]) for r in rows])
    archived = json.load((DATA / "rational_fit_coefficients.json").open())["central"]
    # the printed coefficients of the practical formula: eight decimals
    coeff = {k: (round(v, 8) if k in "abcdef" else v) for k, v in archived.items()}
    primary = np.where(lam <= 5000.0)[0]
    cal_p, hold_p = split_indices(len(primary))
    labels = np.array(["extrapolation"] * len(lam), dtype=object)
    labels[primary[cal_p]] = "calibration"
    labels[primary[hold_p]] = "holdout"
    fitvals = rational(lam, coeff)
    with (DATA / "reference_nodes_with_split.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "F_continuum", "set", "F_rational", "relative_error_pct"])
        for x, y, s, z in zip(lam, ref, labels, fitvals):
            w.writerow([x, y, s, z, 100 * (z / y - 1)])
    sel = lambda name: labels == name
    withheld = (labels == "holdout") | (labels == "extrapolation")
    cross = (lam >= 0.1) & (lam <= 10)
    # Sensitivity of the fit to B within its stated range (refit on the same calibration nodes).
    cal = sel("calibration")
    coeff_lo = fit(lam[cal], ref[cal], B_CENTRAL - B_HALF_RANGE)
    coeff_hi = fit(lam[cal], ref[cal], B_CENTRAL + B_HALF_RANGE)
    metrics = {
        "note": "printed (8-decimal) coefficients evaluated on the current 67-node table; no refit",
        "full_precision_extrapolation_lambda_gt_5000": error_metrics(lam[sel("extrapolation")], ref[sel("extrapolation")], archived),
        "calibration": error_metrics(lam[cal], ref[cal], coeff),
        "holdout_interleaved": error_metrics(lam[sel("holdout")], ref[sel("holdout")], coeff),
        "extrapolation_lambda_gt_5000": error_metrics(lam[sel("extrapolation")], ref[sel("extrapolation")], coeff),
        "withheld_all": error_metrics(lam[withheld], ref[withheld], coeff),
        "all": error_metrics(lam, ref, coeff),
        "crossover_0p1_10": error_metrics(lam[cross], ref[cross], coeff),
        "B_minus_all": error_metrics(lam, ref, coeff_lo),
        "B_plus_all": error_metrics(lam, ref, coeff_hi),
    }
    (DATA / "rational_fit_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: (v if isinstance(v, str) else {kk: v[kk] for kk in ("n", "max_abs_rel_error_pct", "lambda_at_max")}) for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
