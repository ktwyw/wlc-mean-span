#!/usr/bin/env python3
"""Refresh the spectral reference values stored in derived comparison files.

The finite-volume comparison files store the spectral value next to each
finite-volume estimate.  After the reference table is recomputed, this script
replaces those columns (and the derived relative differences and ratios) with
the current table values, without rerunning the finite-volume solver.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    ref = {float(r["lambda"]): float(r["F_continuum"]) for r in csv.DictReader((DATA / "wlc_mean_span_reference.csv").open())}
    for name, fv_col, rel_col, sens_cols in (
        ("finite_volume_comparison.csv", "F_continuum_quad", "relative_difference_pct", ("FV_spatial_sensitivity_rel_pct", "FV_angular_sensitivity_rel_pct", "FV_combined_sensitivity_rel_pct")),
        ("finite_volume_extended_comparison.csv", "F_FV_extended", "relative_difference_pct", ("combined_sensitivity_rel_pct",)),
    ):
        path = DATA / name
        rows = list(csv.DictReader(path.open()))
        for r in rows:
            lam = float(r["lambda"]); old = float(r["F_spectral"]); new = ref[lam]
            r["F_spectral"] = repr(new)
            r[rel_col] = repr(100.0 * (float(r[fv_col]) / new - 1.0))
            for c in sens_cols:          # relative sensitivities were normalized by the spectral value
                r[c] = repr(float(r[c]) * old / new)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("refreshed", name)


if __name__ == "__main__":
    main()
