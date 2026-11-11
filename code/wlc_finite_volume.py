#!/usr/bin/env python3
"""Conservative finite-volume validation solver for the WLC half-space PDE.

The method is intentionally independent of the Legendre transfer solver.  It
uses a cell-centered grid in target distance y and tangent projection u,
conservative angular diffusion, first-order upwind transport, and sparse
matrix-exponential propagation of an augmented affine system.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class FVSettings:
    Ny: int
    Nu: int
    Ymax: float

    def validate(self) -> None:
        if self.Ny < 2 or self.Nu < 4:
            raise ValueError("Ny>=2 and Nu>=4 required")
        if self.Ymax <= 0:
            raise ValueError("Ymax must be positive")


def build_augmented_generator(settings: FVSettings) -> tuple[csc_matrix, float, float, np.ndarray, np.ndarray]:
    settings.validate()
    Ny, Nu, Ymax = settings.Ny, settings.Nu, settings.Ymax
    dy = Ymax / Ny
    du = 2.0 / Nu
    y = (np.arange(Ny) + 0.5) * dy
    u = -1.0 + (np.arange(Nu) + 0.5) * du
    n = Ny * Nu

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    forcing = np.zeros(n, dtype=float)

    def idx(i: int, j: int) -> int:
        return i * Nu + j

    # Angular diffusion: 1/2 d_u[(1-u^2)d_u q].
    faces = -1.0 + np.arange(Nu + 1) * du
    a_face = 0.5 * (1.0 - faces**2) / (du * du)
    # Round tiny endpoint values to zero.
    a_face[0] = 0.0
    a_face[-1] = 0.0

    for i in range(Ny):
        for j in range(Nu):
            k = idx(i, j)
            diag = 0.0
            if j > 0:
                a = a_face[j]
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(a)
                diag -= a
            if j < Nu - 1:
                a = a_face[j + 1]
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(a)
                diag -= a

            uj = u[j]
            if uj > 0:
                # -u q_y with backward/upwind difference.
                diag -= uj / dy
                if i > 0:
                    rows.append(k); cols.append(idx(i - 1, j)); vals.append(uj / dy)
                else:
                    # Incoming half-range boundary q(0,u>0,t)=1.
                    forcing[k] += uj / dy
            else:
                # -u q_y with forward/upwind difference.
                diag += uj / dy  # uj<0
                if i < Ny - 1:
                    rows.append(k); cols.append(idx(i + 1, j)); vals.append(-uj / dy)
                # At Ymax, incoming q=0, so no forcing term.

            rows.append(k); cols.append(k); vals.append(diag)

    # Augment dq/dt=Aq+b with a constant state c'=0, c(0)=1.
    for k, bk in enumerate(forcing):
        if bk != 0.0:
            rows.append(k); cols.append(n); vals.append(float(bk))
    rows.append(n); cols.append(n); vals.append(0.0)
    Aaug = coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1)).tocsc()
    return Aaug, dy, du, y, u


def solve_fv(lam: float, settings: FVSettings) -> float:
    """Return the finite-volume mean-span value F(lambda)."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    Aaug, dy, du, _, _ = build_augmented_generator(settings)
    n = settings.Ny * settings.Nu
    x0 = np.zeros(n + 1, dtype=float)
    x0[-1] = 1.0
    scaled = Aaug * lam
    # Supplying the exact trace avoids recomputing the sparse trace inside
    # expm_multiply.  SciPy controls the matrix-exponential action approximation internally;
    # there is no user-specified tolerance in this implementation.
    trace_scaled = float(np.sum(scaled.diagonal()))
    xt = expm_multiply(scaled, x0, traceA=trace_scaled)
    q = xt[:n].reshape(settings.Ny, settings.Nu)
    # F=integral dy du q, because the factor two for the range cancels the
    # one-half angular average.
    return float(dy * du * np.sum(q))


def default_ymax(lam: float) -> float:
    # Inextensibility gives exact compact support q(y,u;lambda)=0 for y>lambda.
    # The equality y=lambda has zero weight after isotropic averaging, so Ymax=lambda
    # is the natural finite domain and introduces no tail truncation in the integral.
    return float(lam)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lambda", dest="lam", type=float, required=True)
    p.add_argument("--Ny", type=int, required=True)
    p.add_argument("--Nu", type=int, required=True)
    p.add_argument("--Ymax", type=float, default=None)
    args = p.parse_args()
    ymax = default_ymax(args.lam) if args.Ymax is None else args.Ymax
    val = solve_fv(args.lam, FVSettings(args.Ny, args.Nu, ymax))
    print(f"lambda={args.lam:g} Ny={args.Ny} Nu={args.Nu} Ymax={ymax:g} F={val:.12g}")


if __name__ == "__main__":
    main()
