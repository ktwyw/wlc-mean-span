#!/usr/bin/env python3
"""Finite-P_N half-range transfer solver for the projected WLC mean span.

This module implements the numerical method described in the manuscript:

    q_lambda + u q_y = L_u q,
    L_u = 1/2 d_u[(1-u^2)d_u],

with incoming boundary data q(0+,u>0;lambda)=1.  In Laplace space, the
Legendre-Galerkin angular truncation is solved by an ordered complex Schur
factorization of the nodal transport matrix.  An Euler-accelerated
right-half-plane Bromwich series then inverts the transform.

The finite-order transform is exact for the stated angular truncation and
half-range collocation.  Continuum values require angular-order extrapolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, exp, pi
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig, eigh, schur, solve
from scipy.special import eval_legendre, roots_legendre

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BromwichSettings:
    """Euler-accelerated Fourier/Bromwich inversion settings."""

    A: float = 22.0
    N: int = 28
    K: int = 28

    def validate(self) -> None:
        if self.A <= 0:
            raise ValueError("A must be positive")
        if self.N < 1 or self.K < 0:
            raise ValueError("N must be positive and K nonnegative")


@dataclass(frozen=True)
class AngularSystem:
    """Nodal representation of an even finite-P_N angular system."""

    m: int
    nodes: FloatArray
    weights: FloatArray
    V: FloatArray
    Lambda: FloatArray
    R: FloatArray
    h: FloatArray
    positive: NDArray[np.int_]


@dataclass(frozen=True)
class TransformDiagnostics:
    stable_dimension: int
    incoming_condition_number: float
    invariant_residual: float


@lru_cache(maxsize=32)
def angular_system(m: int) -> AngularSystem:
    """Construct the orthonormal Legendre/Gauss nodal system."""

    if m <= 0 or m % 2:
        raise ValueError("m must be a positive even integer")
    u, w = roots_legendre(m)
    V = np.empty((m, m), dtype=float)
    for ell in range(m):
        phi = np.sqrt((2 * ell + 1) / 2.0) * eval_legendre(ell, u)
        V[ell, :] = np.sqrt(w) * phi
    # Orthogonality is exact in exact arithmetic because the Gauss rule
    # integrates products through degree 2m-2.
    lam = np.array([ell * (ell + 1) / 2.0 for ell in range(m)], dtype=float)
    Lambda = np.diag(lam)
    R = -V.T @ Lambda @ V
    h = np.sqrt(w)
    positive = np.flatnonzero(u > 0)
    return AngularSystem(m, u, w, V, Lambda, R, h, positive)


def _ordered_stable_schur(A: ComplexArray) -> tuple[ComplexArray, ComplexArray, int]:
    """Return ordered complex Schur factors with Re(eigenvalue)<0 first."""

    T, Z, sdim = schur(A, output="complex", sort=lambda z: np.real(z) < 0.0)
    return T, Z, int(sdim)


def laplace_transform(
    s: complex,
    m: int,
    *,
    diagnostics: bool = False,
    method: str = "schur",
) -> complex | tuple[complex, TransformDiagnostics]:
    r"""Evaluate the finite-order Laplace transform \hat F_m(s).

    Parameters
    ----------
    s:
        Laplace variable with positive real part.
    m:
        Even number of angular moments/nodes.
    diagnostics:
        Return stable-subspace dimension, incoming-matrix condition number,
        and invariant-subspace residual.
    """

    if np.real(s) <= 0:
        raise ValueError("laplace_transform requires Re(s)>0")
    sys = angular_system(m)
    # Nodal system: g_y = A g, A = U^{-1}(R-sI).  Dividing rows by u
    # applies U^{-1} without forming a dense inverse.
    A = (sys.R.astype(complex) - s * np.eye(m, dtype=complex)) / sys.nodes[:, None]
    expected = m // 2
    if method == "schur" or diagnostics:
        T, Z, sdim = _ordered_stable_schur(A)
        Zs = Z[:, :sdim]
    elif method == "eig":
        eigvals, eigvecs = eig(A, check_finite=False)
        mask = np.real(eigvals) < 0.0
        sdim = int(np.count_nonzero(mask))
        Zs = eigvecs[:, mask]
        T = None
    else:
        raise ValueError("method must be 'eig' or 'schur'")
    if sdim != expected:
        raise RuntimeError(
            f"stable dimension {sdim} differs from expected {expected} "
            f"for m={m}, s={s!r}"
        )
    incoming = Zs[sys.positive, :]
    rhs = sys.h[sys.positive] / s
    coeff = solve(incoming, rhs, assume_a="gen", check_finite=False)
    g0 = Zs @ coeff
    # Integral_0^infinity exp(Ay)g0 dy = -A^{-1}g0.
    integ = -solve(A, g0, assume_a="gen", check_finite=False)
    value = np.dot(sys.h, integ)

    if not diagnostics:
        return complex(value)

    cond = float(np.linalg.cond(incoming))
    As = A @ Zs
    if T is not None:
        residual = float(np.linalg.norm(As - Zs @ T[:sdim, :sdim]) / max(1.0, np.linalg.norm(As)))
    else:
        residual = float("nan")
    return complex(value), TransformDiagnostics(sdim, cond, residual)



def laplace_transform_symmetric_pencil(
    s: float,
    m: int,
    *,
    diagnostics: bool = False,
) -> complex | tuple[complex, TransformDiagnostics]:
    r"""Evaluate ``\hat F_m(s)`` through the real symmetric moment pencil.

    This formulation is available only for positive real ``s``.  With
    ``C = Lambda + s I`` and ``K = C^{-1/2} J C^{-1/2}``, the positive
    eigenvalues of ``K`` are the spatial decay lengths.  The incoming
    half-range values are imposed in the same weighted Gauss-node variables
    used by :func:`laplace_transform`.
    """

    if not np.isreal(s) or float(np.real(s)) <= 0.0:
        raise ValueError("symmetric-pencil evaluation requires positive real s")
    s_real = float(np.real(s))
    sys = angular_system(m)
    # Moment-space streaming matrix J = V diag(u) V^T.
    J = sys.V @ np.diag(sys.nodes) @ sys.V.T
    cdiag = np.diag(sys.Lambda) + s_real
    invsqrt = 1.0 / np.sqrt(cdiag)
    K = (invsqrt[:, None] * J) * invsqrt[None, :]
    alpha, Z = eigh(K, check_finite=False)
    mask = alpha > 0.0
    alpha_pos = alpha[mask]
    Zpos = Z[:, mask]
    expected = m // 2
    if alpha_pos.size != expected:
        raise RuntimeError(
            f"positive decay-length count {alpha_pos.size} differs from expected "
            f"{expected} for m={m}, s={s_real:g}"
        )
    # Moment eigenvectors and weighted nodal values.
    Q = invsqrt[:, None] * Zpos
    nodal_modes = sys.V.T @ Q
    incoming = nodal_modes[sys.positive, :]
    rhs = sys.h[sys.positive] / s_real
    coeff = solve(incoming, rhs, assume_a="gen", check_finite=False)
    value = np.sqrt(2.0) * np.dot(alpha_pos * Q[0, :], coeff)

    if not diagnostics:
        return complex(value)
    cond = float(np.linalg.cond(incoming))
    # Symmetric residual K Z = Z diag(alpha).
    residual = float(
        np.linalg.norm(K @ Zpos - Zpos * alpha_pos[None, :])
        / max(1.0, np.linalg.norm(K @ Zpos))
    )
    return complex(value), TransformDiagnostics(expected, cond, residual)


def invert_euler_bromwich(
    lam: float,
    m: int,
    settings: BromwichSettings = BromwichSettings(),
    *,
    method: str = "schur",
) -> float:
    r"""Invert \hat F_m(s) by an Euler-accelerated Bromwich series."""

    settings.validate()
    if lam <= 0:
        raise ValueError("lam must be positive")
    A_E, N, K = settings.A, settings.N, settings.K
    b = np.empty(N + K + 1, dtype=float)
    for j in range(N + K + 1):
        s = (A_E + 2.0 * pi * 1j * j) / (2.0 * lam)
        b[j] = ((-1.0) ** j) * np.real(laplace_transform(s, m, method=method))
    # T_r = b0/2 + sum_{j=1}^r b_j.
    partial = np.cumsum(b) - 0.5 * b[0]
    weights = np.array([comb(K, k) for k in range(K + 1)], dtype=float) / (2.0**K)
    return float(exp(A_E / 2.0) / lam * np.dot(weights, partial[N : N + K + 1]))


def invert_alias_corrected(
    lam: float,
    m: int,
    settings: BromwichSettings,
    *,
    method: str = "schur",
) -> float:
    r"""Euler--Bromwich inversion with the first Poisson alias removed.

    The ideal Fourier series with parameter A equals
    ``F_A(t) = sum_{k>=0} exp(-kA) F((2k+1)t)``.  The combination
    ``F_A(t) - exp(-A) F_A(3t)`` therefore cancels the first alias, leaving
    ``O(exp(-2A))``.  This permits a small A, which reduces the factor
    ``exp(A/2)`` that multiplies floating-point noise in the transform; the
    production setting is A=12.  The remaining alias,
    ``-exp(-2A)[F(9t) - F(5t)] + O(exp(-3A))``, is negative and below
    ``4 exp(-2A) F(t)`` (about 1.6e-10 F) by subadditivity.
    """

    first = invert_euler_bromwich(lam, m, settings, method=method)
    companion = invert_euler_bromwich(3.0 * lam, m, settings, method=method)
    return float(first - exp(-settings.A) * companion)


def finite_order_curve(
    lambdas: Sequence[float] | NDArray[np.float64],
    m: int,
    settings: BromwichSettings = BromwichSettings(),
    *,
    method: str = "schur",
) -> FloatArray:
    """Evaluate F_m(lambda) at a sequence of positive contour lengths."""

    arr = np.asarray(lambdas, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("all lambda values must be positive")
    return np.array([invert_euler_bromwich(float(x), m, settings, method=method) for x in arr], dtype=float)


def telegraph_p1_transform(s: complex) -> complex:
    r"""Independent analytic m=2 transform for c=1/sqrt(3), decay rate a=1.

    F_hat(s) = 2*c/s * [sqrt(1+1/s)-1].
    The rationalized expression below avoids subtraction at large s and
    does not call the numerical half-range solver being tested.
    """
    z=complex(s)
    if not np.isfinite(z) or z.real<=0:
        raise ValueError('The analytic P1 transform requires finite Re(s)>0.')
    root=np.sqrt(1.0+1.0/z)
    return complex((2.0/np.sqrt(3.0))/z/z/(root+1.0))


def conditioning_audit(
    m_values: Iterable[int],
    s_values: Iterable[complex],
) -> list[dict[str, float]]:
    """Audit incoming stable-subspace conditioning and residuals."""

    rows: list[dict[str, float]] = []
    for m in m_values:
        conds: list[float] = []
        residuals: list[float] = []
        dims: list[int] = []
        for s in s_values:
            _, d = laplace_transform(s, m, diagnostics=True)
            conds.append(d.incoming_condition_number)
            residuals.append(d.invariant_residual)
            dims.append(d.stable_dimension)
        rows.append(
            {
                "m": float(m),
                "condition_min": float(np.min(conds)),
                "condition_median": float(np.median(conds)),
                "condition_max": float(np.max(conds)),
                "residual_max": float(np.max(residuals)),
                "stable_dimension_min": float(np.min(dims)),
                "stable_dimension_max": float(np.max(dims)),
            }
        )
    return rows


if __name__ == "__main__":
    # Lightweight smoke test against selected published values.
    for lam in (0.1, 1.0, 10.0):
        vals = [invert_euler_bromwich(lam, m) for m in (16, 24, 32, 48, 64)]
        print(f"lambda={lam:g}: " + ", ".join(f"m={m}:{v:.10g}" for m, v in zip((16,24,32,48,64), vals)))
