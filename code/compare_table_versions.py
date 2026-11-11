#!/usr/bin/env python3
"""Compare the three successive versions of the 67-node reference table.

* 2026-10-28: eigendecomposition route, uncorrected Euler--Bromwich series, A=22;
* 2026-10-29a: ordered-Schur route, A=22 (first post-review table);
* current: ordered-Schur route, alias-corrected series, A=12.

For each node the script reports the relative changes and, for lambda >= 100,
the relative deviation of each version from the long-chain expansion,
whose own remainder decreases as lambda^-2 (5.5e-6 at lambda=100) and is below
1e-9 for lambda >= 5000.  Writes validation/reference_table_versions.csv and
validation/inversion_noise_audit.csv.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "validation"
A, ZM, B = 4 / math.sqrt(3 * math.pi), 0.71821164, 0.64680


def main() -> None:
    eig = pd.read_csv(V / "previous_tables" / "reference_2026-10-28_eig_A22.csv")[["lambda", "F_continuum"]]
    s22 = pd.read_csv(V / "previous_tables" / "reference_2026-10-29a_schur_A22.csv")[["lambda", "F_continuum", "epsilon_internal_rel"]]
    cur = pd.read_csv(ROOT / "data" / "wlc_mean_span_reference.csv")[["lambda", "F_continuum", "epsilon_internal_rel"]]
    m = eig.rename(columns={"F_continuum": "F_eig_A22"}).merge(
        s22.rename(columns={"F_continuum": "F_schur_A22", "epsilon_internal_rel": "eps_rel_schur_A22"}), on="lambda").merge(
        cur.rename(columns={"F_continuum": "F_current", "epsilon_internal_rel": "eps_rel_current"}), on="lambda")
    m["rel_schur_A22_minus_eig_A22"] = m.F_schur_A22 / m.F_eig_A22 - 1
    m["rel_current_minus_schur_A22"] = m.F_current / m.F_schur_A22 - 1
    m.to_csv(V / "reference_table_versions.csv", index=False, float_format="%.12g")
    big = m[m["lambda"] >= 100].copy()
    e13 = A * big["lambda"] ** 0.5 - 2 * ZM + B / big["lambda"] ** 0.5
    big["dev_longchain_schur_A22"] = big.F_schur_A22 / e13 - 1
    big["dev_longchain_current"] = big.F_current / e13 - 1
    big[["lambda", "dev_longchain_schur_A22", "eps_rel_schur_A22", "dev_longchain_current", "eps_rel_current"]].to_csv(
        V / "inversion_noise_audit.csv", index=False, float_format="%.3e")
    print("max |Schur-eig| (A22):", m.rel_schur_A22_minus_eig_A22.abs().max())
    print("max |current - Schur A22| for lambda<=500:", m[m["lambda"] <= 500].rel_current_minus_schur_A22.abs().max())
    print("max |dev from the long-chain expansion|, lambda>=5000: A22 %.2e, current %.2e" % (
        big[big["lambda"] >= 5000].dev_longchain_schur_A22.abs().max(), big[big["lambda"] >= 5000].dev_longchain_current.abs().max()))


if __name__ == "__main__":
    main()
