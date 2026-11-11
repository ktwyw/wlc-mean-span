#!/usr/bin/env python3
"""Symbolic cross-checks for the midpoint bound and clock-variance formula."""
from pathlib import Path
import json
import sympy as s
ROOT=Path(__file__).resolve().parents[1]
t,r=s.symbols('t r',positive=True)
m2=2*(t-1+s.exp(-t));m4=s.Rational(4,27)*(45*t*t-156*t+214-54*(t+4)*s.exp(-t)+2*s.exp(-3*t))
g=m2**s.Rational(3,2)/(2*s.sqrt(m4))
three=s.series(g.subs(t,t/2)+g/2,t,0,4).removeO().simplify()
assert s.simplify(three-(t/2-t*t/16+t**3/576))==0
V=s.integrate(2*(t-r)*s.Rational(4,45)*s.exp(-3*r),(r,0,t)).simplify()
assert s.simplify(V-s.Rational(8,405)*(3*t-1+s.exp(-3*t)))==0
print('Midpoint lower expansion:',three)
print('Clock variance:',V)
out={'status':'PASS','midpoint_moment_lower_series':str(three),'clock_variance_from_integrated_covariance':str(V),
     'scope':'Symbolic algebra only; not a verification of the stochastic limit interchanges.'}
(ROOT/'validation/analytic_limits_symbolic.json').write_text(json.dumps(out,indent=2)+'\n')
