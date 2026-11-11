#!/usr/bin/env python3
"""Check the WLC tangent-based martingale clock variance with heat-kernel paths."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import csv,math
from pathlib import Path
import numpy as np
from wlc_path_simulation import sample_uniform_sphere
from sphere_heat_kernel import heat_inverse,heat_step
from wlc_moments import clock_variance
ROOT=Path(__file__).resolve().parents[1]
def discrete_variance(t,n):
    h=t/n;k=np.arange(1,n,dtype=float)
    return (4/45)*h*h*(n-.5+2*np.sum((n-k)*np.exp(-3*k*h))+.5*np.exp(-3*t))
def main():
    rows=[]
    for lam,steps,N,seed in [(1.,100,50000,9901100),(10.,500,50000,9910500)]:
        h=lam/steps;rng=np.random.default_rng(seed);grid,cdf,diag=heat_inverse(h);t=sample_uniform_sphere(rng,N);q=np.zeros_like(t)
        for _ in range(steps):
            u=heat_step(t,grid,cdf,rng);q+=h*(2-t*t-u*u)/2;t=u
        v=np.mean((q-2*lam/3)**2,axis=1);mean=float(v.mean());se=float(v.std(ddof=1)/math.sqrt(N));discrete=float(discrete_variance(lam,steps));continuum=float(clock_variance(lam))
        rows.append({'lambda':lam,'steps':steps,'n':N,'seed':seed,'variance_MC':mean,'SE':se,'variance_discrete':discrete,'variance_continuum':continuum,'z_vs_discrete':(mean-discrete)/se,'kernel_moment_max_error':diag['max_moment_error']})
    with (ROOT/'data/martingale_clock_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(rows)
if __name__=='__main__':main()
