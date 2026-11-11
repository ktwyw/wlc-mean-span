#!/usr/bin/env python3
"""Coupled-step path validation with numerical inversion of S2 heat kernel.

Rotational transitions target the exact WLC heat kernel up to finite Legendre
series and inverse-CDF interpolation. Position uses piecewise-linear tangent;
continuous within-segment extrema are included. Four resolutions share the
same fine tangent paths, allowing statistically precise paired differences.
Default cases: lambda = 0.1, 1, 10, 100 (data/heat_path_coupled.csv).  With
--long: lambda = 1e3 and 1e4 (data/heat_path_long.csv), the long-chain check of
the supplementary material (--long-extra adds three further batches of 20 000 chains
at 1e4, data/heat_path_long_extra.csv; pool_long_paths.py pools the four); for long chains the trapezoidal position rule inflates the diffusivity
by (h/2)coth(h/2) = 1 + h^2/12 + ...; the finest steps are chosen so that the
resulting bias, h^2/24, is below 3e-4, and the finest-grid values are reported.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,json,math
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sphere_heat_kernel import heat_inverse,heat_step
from wlc_path_simulation import sample_uniform_sphere
from wlc_moments import radial_moments
ROOT=Path(__file__).resolve().parents[1]
class Tracker:
    def __init__(self,t,steps,span):
        self.t=t.copy();self.pos=np.zeros_like(t);self.lo=self.pos.copy();self.hi=self.pos.copy();self.mid=None
        self.steps=steps;self.stride=span
    def update(self,end,step,h):
        u0=self.t;du=end-u0;x=self.pos
        with np.errstate(divide='ignore',invalid='ignore'): frac=-u0/du
        mask=(frac>0)&(frac<1)&np.isfinite(frac);f=np.where(mask,frac,0.)
        xm=x+h*(u0*f+du*f*f/2);xn=x+h*(u0+end)/2
        self.hi=np.maximum(self.hi,np.maximum(xn,np.where(mask,xm,-np.inf)))
        self.lo=np.minimum(self.lo,np.minimum(xn,np.where(mask,xm,np.inf)))
        self.pos=xn;self.t=end.copy()
        if step==self.steps//2:self.mid=xn.copy()
    def finish(self):
        assert self.mid is not None
        span=self.hi-self.lo
        three=(np.abs(self.mid)+np.abs(self.pos-self.mid)+np.abs(self.pos))/2
        assert np.min(span-three)>-1e-10
        return span.mean(axis=1),three.mean(axis=1),np.abs(self.pos).mean(axis=1),np.sum(self.pos**2,axis=1)
def stats(x):return float(np.mean(x)),float(np.std(x,ddof=1)/math.sqrt(len(x)))
def job(case):
    lam,steps,n,seed,grid=case;h=lam/steps;rng=np.random.default_rng(seed)
    u,cdf,diag=heat_inverse(h,grid);t=sample_uniform_sphere(rng,n);strides=(8,4,2,1)
    trackers={k:Tracker(t,steps,k) for k in strides}
    for step in range(1,steps+1):
        t=heat_step(t,u,cdf,rng)
        for k in strides:
            if step%k==0:trackers[k].update(t,step,h*k)
    samples={k:trackers[k].finish() for k in strides};fine=samples[1][0];rows=[]
    for k in strides:
        span,three,end,r2=samples[k];sm,se=stats(span);tm,te=stats(three);em,ee=stats(end);r2m,r2se=stats(r2);dm,de=stats(span-fine)
        rows.append(dict(lam=lam,steps=steps//k,dt=h*k,n=n,seed=seed,span=sm,span_se=se,
            three_point=tm,three_point_se=te,endpoint_abs=em,endpoint_abs_se=ee,R2=r2m,R2_se=r2se,
            R2_exact=float(radial_moments(np.array([lam]))[0][0]),
            pairwise_minus_fine=dm,pairwise_se=de))
    folder=ROOT/'data/heat_path_samples';folder.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(folder/f'heat_l{lam:g}_n{steps}_N{n}_seed{seed}.npz',
                        **{f'{key}_stride{k}':arr for k,pack in samples.items() for key,arr in zip(('span','three','endpoint_abs','R2'),pack)},
                        lam=lam,steps=steps,n=n,seed=seed)
    return rows,diag

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--paths',type=int,default=30000)
    p.add_argument('--grid',type=int,default=24001);p.add_argument('--pilot',action='store_true')
    p.add_argument('--long',action='store_true');p.add_argument('--long-extra',action='store_true');a=p.parse_args()
    if a.pilot: cases=[(1.,256,2000,773256,a.grid)]
    elif a.long_extra: cases=[(10000.,131072,20000,8161310+i,a.grid) for i in (1,2,3)]      # three more batches at 1e4
    elif a.long: cases=[(1000.,16384,40000,8151638,a.grid),(10000.,131072,20000,8161310,a.grid)]
    else: cases=[(.1,512,a.paths,8110512,a.grid),(1.,1024,a.paths,8121024,a.grid),
                 (10.,2048,a.paths,8132048,a.grid),(100.,8192,a.paths,8148192,a.grid)]
    rows=[];diags=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(job,c):c for c in cases}
        for f in as_completed(futures):
            rr,d=f.result();rows+=rr;diags.append(d)
            print('lambda',rr[0]['lam'],'finest span',rr[-1]['span'],'SE',rr[-1]['span_se'],flush=True)
    rows.sort(key=lambda r:(r['lam'],r['seed'],r['steps']));diags.sort(key=lambda r:r['h'])
    prefix='heat_path_pilot' if a.pilot else ('heat_path_long_extra' if a.long_extra else ('heat_path_long' if a.long else 'heat_path_coupled'))
    with (ROOT/'data'/f'{prefix}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/'validation'/f'{prefix}_kernel.json').write_text(json.dumps(diags,indent=2)+'\n')
    print('wrote',prefix,flush=True)
if __name__=='__main__':main()
