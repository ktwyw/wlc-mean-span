#!/usr/bin/env python3
"""Seeded polygonal-chain validation for the ideal WLC mean projected span.

The tangent transition is von Mises-Fisher with its concentration chosen so
that the first tangent correlation equals exp(-h).  This is an independent
finite-step model and converges to rotational Brownian motion as h -> 0; it is
not the exact spherical heat kernel at finite h.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def mean_cosine(kappa: float) -> float:
    if kappa < 1e-4:
        return kappa/3.0 - kappa**3/45.0
    return 1.0 / math.tanh(kappa) - 1.0 / kappa


def kappa_for_step(h: float) -> float:
    target = math.exp(-h)
    return brentq(lambda k: mean_cosine(k) - target, 1e-8, max(20.0, 4.0/h))


def sample_uniform_sphere(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n,3))
    x /= np.linalg.norm(x, axis=1)[:,None]
    return x


def sample_vmf_about(t: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    n = len(t)
    xi = rng.random(n)
    e2 = math.exp(-2.0*kappa) if kappa < 350 else 0.0
    mu = 1.0 + np.log(xi + (1.0-xi)*e2)/kappa
    mu = np.clip(mu, -1.0, 1.0)
    g = rng.normal(size=(n,3))
    g -= np.sum(g*t,axis=1)[:,None]*t
    norm = np.linalg.norm(g,axis=1)
    bad = norm < 1e-12
    while np.any(bad):
        gb = rng.normal(size=(np.count_nonzero(bad),3))
        tb = t[bad]
        gb -= np.sum(gb*tb,axis=1)[:,None]*tb
        g[bad] = gb
        norm[bad] = np.linalg.norm(gb,axis=1)
        bad = norm < 1e-12
    g /= norm[:,None]
    return mu[:,None]*t + np.sqrt(np.maximum(0.0,1.0-mu*mu))[:,None]*g


def simulate_batch(lam: float, h: float, n_paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = sample_uniform_sphere(rng,n_paths)
    pos = np.zeros((n_paths,3))
    pmin = pos.copy(); pmax = pos.copy()
    remaining = lam
    step_index = 0
    while remaining > 1e-15:
        dt = min(h, remaining)
        kappa = kappa_for_step(dt)
        t2 = sample_vmf_about(t,kappa,rng)
        delta_t = t2-t
        # Turning point of quadratic position interpolation for each coordinate.
        with np.errstate(divide='ignore',invalid='ignore'):
            frac = -t/delta_t
        valid = (frac>0.0)&(frac<1.0)&np.isfinite(frac)
        internal = pos.copy()
        # Evaluate only valid coordinate-wise turning points to avoid inf*0 warnings.
        internal[valid] = (pos + dt*(t*np.where(valid,frac,0.0)
                            + 0.5*delta_t*np.where(valid,frac*frac,0.0)))[valid]
        pmin = np.minimum(pmin, internal)
        pmax = np.maximum(pmax, internal)
        pos = pos + 0.5*dt*(t+t2)
        pmin = np.minimum(pmin,pos); pmax=np.maximum(pmax,pos)
        t=t2; remaining-=dt; step_index+=1
    spans=pmax-pmin
    return np.mean(spans,axis=1)


def simulate(lam: float, h: float, paths: int, seed: int, batch_size: int=20000) -> tuple[float,float]:
    vals=[]; done=0; batch=0
    while done<paths:
        n=min(batch_size,paths-done)
        vals.append(simulate_batch(lam,h,n,seed+batch*1000003))
        done+=n; batch+=1
    x=np.concatenate(vals)
    return float(np.mean(x)), float(np.std(x,ddof=1)/math.sqrt(len(x)))


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--lambda',dest='lam',type=float,required=True)
    p.add_argument('--step',type=float,required=True)
    p.add_argument('--paths',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args()
    mean,sem=simulate(args.lam,args.step,args.paths,args.seed)
    row={'lambda':args.lam,'delta_tau':args.step,'paths':args.paths,'seed':args.seed,'F_MC':mean,'SEM':sem}
    print(row)
    if args.output:
        exists=args.output.exists(); args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row.keys()))
            if not exists:w.writeheader()
            w.writerow(row)

if __name__=='__main__':
    main()
