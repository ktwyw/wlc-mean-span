#!/usr/bin/env python3
"""Direct test of the second-order matching constant a2 (supplementary material).

The supplement obtains a2 = z_M^2 - 4/45 - I from a reciprocity argument, without
solving the second-order boundary problem.  Here that problem is solved directly in
the nodal P_N discretization, independently of milne_analysis.py:

    first order   Psi = y - u + z_M + (decaying modes),          Psi(0, u>0) = 0
    second order  Phi = y^2/2 - y u + z_M (y - u) + a2 + (u^2 - 1/3)/3
                        + (decaying modes),                      Phi(0, u>0) = 0

and the moment identities M1 = 2/3, M2 = 2 z_M/3, I = M3/2 - 1/5 and the two routes
to B are compared.  Output: data/second_order_matching_check.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDERS = (32, 64, 128)


def solve(m: int) -> dict:
    u, w = np.polynomial.legendre.leggauss(m)
    ell = np.arange(m)
    phi = np.array([np.polynomial.legendre.legval(u, np.eye(m)[i]) * np.sqrt((2 * i + 1) / 2) for i in ell])
    lap = phi.T @ np.diag(-ell * (ell + 1) / 2) @ (phi * w)        # angular diffusion on nodal values
    ev, vec = np.linalg.eig(np.diag(1 / u) @ lap)                  # d g/dy = D_u^{-1} L g
    dec = [i for i in range(m) if ev[i].real < -1e-3]              # decaying modes (the zero mode is double)
    assert len(dec) == m // 2 - 1
    mu, modes = ev[dec].real, vec[:, dec].real
    pos, neg = u > 0, u < 0
    system = np.column_stack([np.ones(pos.sum()), modes[pos]])
    first = np.linalg.solve(system, u[pos])                        # unknowns: z_M and mode amplitudes
    z, c = first[0], first[1:]
    h, v, wn = (-u + z + modes @ c)[neg], -u[neg], w[neg]          # outgoing trace h(v) = Psi(0, -v)
    mom = [float(np.sum(wn * v**n * h)) for n in range(4)]
    layer = float(np.sum(c * (0.5 * (w @ modes)) / (-mu)))         # I = int <Psi - y + u - z_M> dy
    phi2 = (u**2 - 1 / 3) / 3
    a2 = float(np.linalg.solve(system, (z * u - phi2)[pos])[0])    # direct second-order solve
    a2_rec = z * z - 4 / 45 - layer                                # reciprocity
    b_mom = np.sqrt(3 / np.pi) * (2 * z * z - 2 * mom[3] + 17 / 45)
    b_dir = (2 * np.sqrt(3) * (a2 - layer) - 11 / (15 * np.sqrt(3))) / np.sqrt(np.pi)
    return {"m": m, "z_M": z, "M1_minus_2/3": mom[1] - 2 / 3, "M2_minus_2zM/3": mom[2] - 2 * z / 3,
            "I": layer, "I_minus_(M3/2-1/5)": layer - (mom[3] / 2 - 0.2),
            "a2_direct": a2, "a2_reciprocity": a2_rec, "a2_difference": a2 - a2_rec,
            "B_moment_formula": b_mom, "B_direct": b_dir}


def main() -> None:
    rows = [solve(m) for m in ORDERS]
    out = ROOT / "data" / "second_order_matching_check.csv"
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (repr(val) if isinstance(val, float) else val) for k, val in r.items()})
    for r in rows:
        print(r)
    assert max(abs(r["a2_difference"]) for r in rows) < 1e-10
    assert max(abs(r["M1_minus_2/3"]) + abs(r["M2_minus_2zM/3"]) + abs(r["I_minus_(M3/2-1/5)"]) for r in rows) < 1e-10


if __name__ == "__main__":
    main()
