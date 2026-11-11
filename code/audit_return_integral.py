#!/usr/bin/env python3
"""Independent normalization tests for the random-acceleration return integral.

The input is the first-return kernel printed in the manuscript.  This script
checks its speed-weighted double integral directly, the intermediate eta
integral, the rational half-line and full-line representations, and the
partial-fraction identity.  It does not fit or infer the stiff coefficient
from the spectral reference curve.  Quadrature errors are numerical error
estimates, not certified interval bounds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import scipy
from scipy.integrate import quad
from scipy.special import erf
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
J_EXACT = 37.0 / 192.0
C2_EXACT = 37.0 / 768.0


def return_kernel(v: float, v0: float, t: float = 1.0) -> float:
    """First-return density for velocity generator d_v^2; v,v0 >= 0."""
    if v < 0.0 or v0 < 0.0 or t <= 0.0:
        raise ValueError("Require v >= 0, v0 >= 0 and t > 0")
    return (
        math.sqrt(3.0) / (2.0 * math.pi) * v / t**2
        * math.exp(-(v*v - v*v0 + v0*v0) / t)
        * float(erf(math.sqrt(3.0 * v * v0 / t)))
    )


def direct_kernel_integral(t: float = 1.0) -> tuple[float, float, float]:
    """Nested quadrature on both positive half-lines, without eta reduction."""
    inner_errors: list[float] = []

    def inner(v0: float) -> float:
        value, error = quad(
            lambda v: return_kernel(v, v0, t),
            0.0, math.inf, epsabs=2e-12, epsrel=2e-12, limit=180,
        )
        inner_errors.append(v0 * error)
        return v0 * value

    value, outer_error = quad(inner, 0.0, math.inf, epsabs=2e-11,
                              epsrel=2e-11, limit=180)
    return value, outer_error, max(inner_errors, default=0.0)


def rational_integrand(r: float) -> float:
    return (3.0 + 4.0*r*r) / ((1.0 + r*r)**2 * (1.0 + 4.0*r*r)**2)


def run_audit() -> tuple[list[dict[str, object]], dict[str, object]]:
    r = sp.symbols("r", real=True)
    rational = (3 + 4*r**2) / ((1+r**2)**2 * (1+4*r**2)**2)
    partial = (-sp.Rational(16, 27)/(1+4*r**2)
               + sp.Rational(32, 9)/(1+4*r**2)**2
               + sp.Rational(4, 27)/(1+r**2)
               - sp.Rational(1, 9)/(1+r**2)**2)
    if sp.cancel(rational - partial) != 0:
        raise AssertionError("Partial-fraction identity failed")
    # Exact integration of each rational component using the elementary
    # integrals pi/sqrt(a), pi/(2 sqrt(a)) on the full real line.
    full_exact = sp.simplify(-sp.Rational(16,27)*sp.pi/2
                             + sp.Rational(32,9)*sp.pi/4
                             + sp.Rational(4,27)*sp.pi
                             - sp.Rational(1,9)*sp.pi/2)
    assert full_exact == 37*sp.pi/54
    assert sp.simplify(sp.Rational(9,32)*full_exact/sp.pi) == sp.Rational(37,192)

    results: list[dict[str, object]] = []
    for t in (0.25, 1.0, 4.0):
        value, outer, max_inner = direct_kernel_integral(t)
        results.append({"representation": "direct_return_kernel", "t": t,
                        "value": value, "target": J_EXACT,
                        "absolute_difference": abs(value-J_EXACT),
                        "quadrature_error_estimate": outer,
                        "max_sampled_weighted_inner_error": max_inner})

    eta_value, eta_error = quad(
        lambda eta: eta**1.5 * (eta*eta+eta+1.0)
        / ((eta*eta-eta+1.0)**2 * (eta+1.0)**3),
        0, math.inf, epsabs=2e-13, epsrel=2e-13, limit=180,
    )
    half, half_error = quad(rational_integrand, 0.0, math.inf,
                            epsabs=2e-13, epsrel=2e-13, limit=180)
    full, full_error = quad(rational_integrand, -math.inf, math.inf,
                            epsabs=3e-13, epsrel=3e-13, limit=180)
    for label, integral, error, prefactor in (
        ("eta_integral", eta_value, eta_error, 9.0/(8.0*math.pi)),
        ("r_half_line", half, half_error, 9.0/(16.0*math.pi)),
        ("r_full_line_corrected", full, full_error, 9.0/(32.0*math.pi)),
    ):
        value = prefactor*integral
        results.append({"representation": label, "t": 1.0, "value": value,
                        "target": J_EXACT, "absolute_difference": abs(value-J_EXACT),
                        "quadrature_error_estimate": abs(prefactor)*error,
                        "max_sampled_weighted_inner_error": 0.0})
    # Both normalizations must produce the same WLC stiff coefficient.
    assert abs(9*half/(16*math.pi)/4 - C2_EXACT) < 1e-12
    assert abs(9*full/(32*math.pi)/4 - C2_EXACT) < 1e-12
    if max(float(row["absolute_difference"]) for row in results) > 2e-9:
        raise AssertionError("Return-integral normalization exceeds tolerance")
    # A regression guard: the old full-line prefactor gives twice J.
    incorrect_full = 9.0 * full / (16.0*math.pi)
    assert abs(incorrect_full - 2.0*J_EXACT) < 1e-12
    summary = {
        "status": "PASS", "J_exact": "37/192", "J_decimal": J_EXACT,
        "stiff_coefficient_exact": "37/768", "stiff_coefficient_decimal": C2_EXACT,
        "full_rational_integral_exact": "37*pi/54",
        "half_rational_integral_exact": "37*pi/108",
        "correct_half_line_prefactor": "9/(16*pi)",
        "correct_full_line_prefactor": "9/(32*pi)",
        "incorrect_old_full_line_result": incorrect_full,
        "maximum_absolute_J_difference": max(float(row["absolute_difference"]) for row in results),
        "partial_fraction_identity": "symbolically exact",
        "scipy_version": scipy.__version__, "sympy_version": sp.__version__,
        "interpretation": "Normalization audit; does not establish the full matched-asymptotic remainder.",
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    results, summary = run_audit()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "return_integral_audit.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    (args.out_dir / "return_integral_audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
