#!/usr/bin/env python3
"""Quadrature of McKean's first-return kernel for unit-variance Brownian velocity."""
from __future__ import annotations
import csv,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import erf
ROOT=Path(__file__).resolve().parents[1]
def probability(xi):
    v0=math.sqrt(2)*xi
    def inner(t):
        def K(v):
            return math.sqrt(3)/(2*math.pi)*v/t**2*math.exp(-(v*v-v*v0+v0*v0)/t)*erf(math.sqrt(3*v*v0/t))
        return quad(K,0,math.inf,epsabs=2e-10,epsrel=2e-9,limit=120)[0]
    return quad(inner,0,1,epsabs=1e-9,epsrel=1e-8,limit=120)
def main():
    xis=sorted(set(np.linspace(.1,2.1,41).tolist()+[.3,.7,1.2,2.]))
    rows=[]
    for xi in xis:
        v,e=probability(xi);rows.append({'xi':xi,'return_probability':v,'outer_quadrature_error_estimate':e})
    with (ROOT/'data/grazing_kernel_probabilities.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Saved',len(rows),'integrated-Brownian return probabilities.')
if __name__=='__main__':main()
