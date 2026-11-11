#!/usr/bin/env python3
"""Symbolic/analytic checks for the added theory; no trajectory simulation.

These tests check algebra and normalizations, not existence/remainder theorems.
They complement the explicit assumptions and proofs in the SI.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import sympy as sp
from scipy.integrate import quad
from scipy.special import gamma

ROOT=Path(__file__).resolve().parents[1]
u,y,z,a2,I,M3,k=sp.symbols('u y z a2 I M3 k',real=True)
L=lambda f: sp.diff((1-u**2)*sp.diff(f,u),u)/2
avg=lambda f: sp.integrate(sp.expand(f),(u,-1,1))/2
T=lambda f: u*sp.diff(f,y)-L(f)
checks={}
def zero(name,expr):
    simplified=sp.simplify(expr)
    checks[name]=str(simplified)
    if simplified!=0:raise AssertionError((name,simplified))

phi2=(u**2-sp.Rational(1,3))/3
phi3=(u**3-5*u)/18
zero('slow_order_k', L(u)+u)
zero('slow_order_k2', L(phi2)+u*u-sp.Rational(1,3))
zero('slow_order_k3', L(phi3)+u*phi2-u/3)
zero('s4_coefficient',avg(u*phi3)+sp.Rational(11,135))
zero('mean_phi2',avg(phi2))
zero('mean_phi3',avg(phi3))
Psi_far=y-u+z
Phi_far=y*y/2-y*u+z*(y-u)+a2+phi2
zero('stationary_far_field',T(Psi_far))
zero('second_order_inhomogeneous_equation',T(Phi_far)+sp.Rational(1,3))
reciprocal=avg(u*Phi_far*Psi_far.subs(u,-u))
zero('reciprocal_far_polynomial',reciprocal-(-y*y/6-z*y/3+(a2-z*z)/3+sp.Rational(4,135)))
zero('reciprocal_integration_constant',(reciprocal+y*y/6+z*y/3+I/3).subs(a2,z*z-sp.Rational(4,45)-I))
Bnorm=2*(z*z-sp.Rational(4,45)-2*I)-sp.Rational(11,45)
zero('B_moment_formula',Bnorm.subs(I,M3/2-sp.Rational(1,5))-(2*z*z-2*M3+sp.Rational(17,45)))
V=sp.symbols('V')
zero('B_variance_formula',(2*z*z-2*M3+sp.Rational(17,45)).subs(M3,sp.Rational(2,3)*(V+z*z))-(sp.Rational(2,3)*z*z+sp.Rational(17,45)-sp.Rational(4,3)*V))
zero('positive_lower_bound',2*z*z-sp.Rational(4,3)*z+sp.Rational(17,45)-(2*(z-sp.Rational(1,3))**2+sp.Rational(7,45)))
J=sp.Rational(37,192)
zero('three_dimensional_grazing_coefficient',J/2*sp.Rational(1,2)-sp.Rational(37,768))
length,p=sp.symbols('L p',positive=True);F=sp.Function('F')
X=p*F(length/p)
zero('stiffness_first_derivative',sp.diff(X,p)-(F(length/p)-(length/p)*sp.Subs(sp.diff(F(y),y),y,length/p)))
zero('stiffness_second_derivative',sp.diff(X,p,2)-length**2/p**3*sp.Subs(sp.diff(F(y),y,2),y,length/p))
# Independently integrate stationary angular distributions in D=2,...,8.
dim_rows=[]
for D in range(2,9):
    pi0=gamma(D/2)/(math.sqrt(math.pi)*gamma((D-1)/2))
    mu=gamma(D/2)/(math.sqrt(math.pi)*gamma((D+1)/2))
    # Angular theta avoids an integrable endpoint singularity at D=2.
    norm=quad(lambda th:pi0*math.sin(th)**(D-2),0,math.pi,epsabs=2e-13)[0]
    num_mu=2*quad(lambda th:pi0*math.cos(th)*math.sin(th)**(D-2),0,math.pi/2,epsabs=2e-13)[0]
    a0=2/(D-1)
    assert abs(norm-1)<1e-12 and abs(num_mu-mu)<1e-12
    assert abs(pi0*a0/mu-1)<1e-12
    dim_rows.append({'D':D,'pi0':float(pi0),'a0':a0,'mu':float(mu),
                     'absolute_c2':float(J/2*pi0*a0),'relative_c2':float(J/2*pi0*a0/mu)})
import csv
with (ROOT/'data/grazing_dimension_coefficients.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(dim_rows[0]));w.writeheader();w.writerows(dim_rows)
report={'symbolic_residuals':checks,'number_symbolic_checks':len(checks),'dimensions_checked':dim_rows,
        'status':'All listed algebraic and normalization checks passed.',
        'not_proved_by_code':'Uniform first-passage convergence, half-space existence/uniqueness, and asymptotic remainder estimates; see explicit SI assumptions.'}
(ROOT/'data/theory_symbolic_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
