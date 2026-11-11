#!/usr/bin/env python3
"""Long-chain check of the reference table against the long-chain expansion plus a fitted remainder.

z_M and B come from the stationary Milne problem, which involves no Laplace
inversion.  The regular matched expansion contains only half-integer powers
beyond the long-chain expansion (its transform is s^{-3/2}P1(s) + s^{-1}P2(s), and integer powers
of s do not contribute at large lambda), so the deviations F - F_as for
100 <= lambda <= 1500 are fitted with c2 lambda^{-3/2} + c4 lambda^{-5/2}; the
residual of the table against
F_as + remainder is then reported at every node, together with the internal
sensitivity.  Diagnostic refits are recorded: with an added lambda^{-1} term
(absent in the expansion), with an added lambda^{-1/2} term (a shift of B), with
lambda^{-2} in place of lambda^{-5/2}, and on 100 <= lambda <= 700 only (so that
the tested band 1e3..2e4 lies wholly outside the fit range).  The check is conditional on the long-chain expansion.
For lambda >= 1e4 the script also recomputes the m = 32-96 extrapolation with the
inversion parameter A = 10 and 14 (alias-corrected, N = K = 32; about 30 s on one core) to
show the setting dependence of F_inf where eps_int is not conservative.
Outputs: data/long_chain_remainder_check.csv and .json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from wlc_pn_transfer import BromwichSettings, invert_alias_corrected

ROOT = Path(__file__).resolve().parents[1]
Z_M, B = 0.71821164, 0.64680
FIT = (100.0, 1500.0)


def fit(lam, d, powers):
    A = np.array([lam ** p for p in powers]).T
    return np.linalg.lstsq(A, d, rcond=None)[0]


def f_inf(lam: float, A: float) -> float:
    ms = [32, 48, 64, 96]
    f = [invert_alias_corrected(lam, m, BromwichSettings(A, 32, 32)) for m in ms]
    M = np.array([[1.0, m ** -3.0, m ** -4.0] for m in ms])
    return float(np.linalg.lstsq(M, np.array(f), rcond=None)[0][0])


def resid_max(lam, d, F, m, band, powers):
    c = fit(lam[m], d[m], powers)
    r = (d - sum(ci * lam ** p for ci, p in zip(c, powers))) / F
    return float(np.max(abs(r[band])))


def main() -> None:
    rows = list(csv.DictReader((ROOT / "data" / "wlc_mean_span_reference.csv").open()))
    lam = np.array([float(r["lambda"]) for r in rows])
    F = np.array([float(r["F_continuum"]) for r in rows])
    eps = np.array([float(r["epsilon_internal_rel"]) for r in rows])
    F13 = 4 * np.sqrt(lam / (3 * np.pi)) - 2 * Z_M + B / np.sqrt(lam)
    d = F - F13
    m = (lam >= FIT[0]) & (lam <= FIT[1])
    BASIS = [-1.5, -2.5]
    c2, c4 = fit(lam[m], d[m], BASIS)
    c1 = fit(lam[m], d[m], [-1.0] + BASIS)[0]
    dB = fit(lam[m], d[m], [-0.5] + BASIS)[0]
    resid = (d - c2 * lam ** -1.5 - c4 * lam ** -2.5) / F
    band = (lam >= 1e3) & (lam <= 2e4)
    m700 = (lam >= 100.0) & (lam <= 700.0)
    A_dev = {}
    for i in np.nonzero(lam >= 1e4)[0]:
        A_dev[i] = [f_inf(float(lam[i]), A) / F[i] - 1 for A in (10.0, 14.0)]
    out = ROOT / "data" / "long_chain_remainder_check.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lambda", "rel_dev_from_longchain", "rel_residual_longchain_plus_remainder", "epsilon_internal_rel",
                    "residual_over_epsilon", "rel_change_F_inf_A10", "rel_change_F_inf_A14"])
        for i in np.nonzero(lam >= 100)[0]:
            a10, a14 = (repr(float(v)) for v in A_dev[i]) if i in A_dev else ("", "")
            w.writerow([repr(float(lam[i])), repr(float(d[i] / F[i])), repr(float(resid[i])), repr(float(eps[i])),
                        repr(float(resid[i] / eps[i])), a10, a14])
    summary = {
        "fit_range": FIT, "basis": BASIS, "c2_lambda_m3_2": float(c2), "c4_lambda_m5_2": float(c4),
        "refit_lambda_m1_coefficient": float(c1), "refit_B_shift": float(dB),
        "max_abs_rel_residual_1e3_to_2e4": float(np.max(abs(resid[band]))),
        "max_abs_rel_residual_over_eps_1e3_to_2e4": float(np.max(abs(resid[band] / eps[band]))),
        "max_abs_rel_residual_1e3_to_2e4_with_lambda_m1": resid_max(lam, d, F, m, band, [-1.0] + BASIS),
        "max_abs_rel_residual_1e3_to_2e4_with_lambda_m1_2": resid_max(lam, d, F, m, band, [-0.5] + BASIS),
        "max_abs_rel_residual_1e3_to_2e4_with_lambda_m2_instead": resid_max(lam, d, F, m, band, [-1.5, -2.0]),
        "max_abs_rel_residual_1e3_to_2e4_fit_100_to_700": resid_max(lam, d, F, m700, band, BASIS),
        "c2_fit_100_to_700": float(fit(lam[m700], d[m700], BASIS)[0]),
        "max_abs_A_change_over_eps_lambda_ge_1e4": float(max(max(abs(v) for v in A_dev[i]) / eps[i] for i in A_dev)),
        "rel_residual_above_2e4": {repr(float(l)): float(r) for l, r in zip(lam[lam > 2e4], resid[lam > 2e4])},
        "residual_over_eps_above_2e4": {repr(float(l)): float(r / e) for l, r, e in zip(lam[lam > 2e4], resid[lam > 2e4], eps[lam > 2e4])},
        "note": "Conditional on the long-chain expansion; z_M and B from the stationary Milne problem (no Laplace inversion).",
    }
    (ROOT / "data" / "long_chain_remainder_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
