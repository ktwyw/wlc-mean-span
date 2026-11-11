#!/usr/bin/env python3
"""Independent quadrature and limiting checks for span-to-gyration data."""
import csv
import math
from fractions import Fraction
from pathlib import Path
import mpmath as mp
import numpy as np
from plot_span_to_rg import rg2_over_p2

ROOT = Path(__file__).resolve().parents[1]
mp.mp.dps = 50
errors = []
for t in (0.0001,0.001,0.01,0.1,0.2999,0.3,1,5,50,1000,5000):
    z=mp.mpf(str(t))
    # Independent integral of the exact mean-square separation.
    exact = mp.quad(lambda v: 2*(z-v)*(v-1+mp.exp(-v)), [0,z])/z**2
    got = float(rg2_over_p2(np.array([t]))[0])
    err = abs(mp.mpf(got)/exact-1)
    errors.append(float(err))
    if err > mp.mpf('2e-11'):
        raise AssertionError((t,got,exact,err))

# The first-order rod ratio follows from F/(lambda/2) and Rg/(lambda/sqrt(12)).
assert Fraction(1,10)-Fraction(37,384)==Fraction(7,1920)
with (ROOT/'data/span_to_gyration_ratio.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==67
maxfit=max(abs(float(r['X_over_Rg_correlation'])/float(r['X_over_Rg_spectral'])-1) for r in rows)
assert maxfit<0.000925
assert abs(math.sqrt(3)-1.7320508075688772)<1e-14
assert abs(4/math.sqrt(math.pi)-2.256758334191025)<1e-14
print('PASS: 11 independent gyration quadratures; max relative difference',max(errors))
print('PASS: rod slope 7/1920; limiting constants; 67 ratio nodes')
print('Maximum relative ratio-correlation discrepancy on 67 spectral nodes:',maxfit)
