#!/usr/bin/env python3
"""Alias cancellation for Fourier/Bromwich inversions, with exact m=2 audit.

The continuous Fourier series is F_A(t)=sum_{k>=0} exp(-kA)F((2k+1)t).
Thus F_A(t)-exp(-A)F_A(3t) removes its first alias. Euler truncation,
finite angular order, eigensolver error, and floating-point error remain.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,math
from pathlib import Path
import mpmath as mp
from wlc_pn_transfer import BromwichSettings,invert_euler_bromwich
ROOT=Path(__file__).resolve().parents[1]
def invert_corrected(t,m,A=16.,n=32,k=32,companion_m=None):
    o=BromwichSettings(A,n,k)
    first=invert_euler_bromwich(t,m,o)
    comp=invert_euler_bromwich(3*t,m if companion_m is None else companion_m,o)
    return first-math.exp(-A)*comp,first,comp

def telegraph_exact(t):
    mp.mp.dps=60; x=mp.mpf(str(t))/2;c=1/mp.sqrt(3)
    return 2*c*(mp.exp(-x)*mp.besseli(0,x)+2*x*mp.exp(-x)*(mp.besseli(0,x)+mp.besseli(1,x))-1)
def main():
    out=[]
    for t in (.0125,.025,.1,1.):
      exact=telegraph_exact(t)
      for A in (16.,18.,20.,22.,24.):
        v,raw,c=invert_corrected(t,2,A)
        out.append({'lambda':t,'A':A,'raw':raw,'corrected':v,'exact':float(exact),'raw_abs_error':abs(float(raw-exact)),
                    'corrected_abs_error':abs(float(v-exact)),'theoretical_continuum_alias_bound':4*t/math.sqrt(3)*math.exp(-2*A)/(1-math.exp(-A))**2})
    with (ROOT/'data/alias_correction_exact_audit.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    for z in out:
      if z['A']==16:print(z)
if __name__=='__main__':main()
