#!/usr/bin/env python3
"""Independent path-level test of the flux-weighted first-return relation.

Draw the outward normal speed with density 2v, simulate the finite-step 3D
angular model, and record first positive returns. The linear interpolation of
crossings is exact for the discretized straight-bond path. The result is not
a rigorous proof of the continuum identity and has finite-step bias.
"""
from __future__ import annotations
import argparse,csv,math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from wlc_path_simulation import kappa_for_step
from projected_path_simulation import step_vmf

ROOT=Path(__file__).resolve().parents[1]

def job(arg):
    lam,steps,n,seed=arg
    rng=np.random.default_rng(seed); h=lam/steps; k=kappa_for_step(h)
    v=np.sqrt(rng.random(n));phi=2*np.pi*rng.random(n); tr=np.sqrt(1-v*v)
    t=np.column_stack((-v,tr*np.cos(phi),tr*np.sin(phi)))
    x=np.zeros(n); inds=np.arange(n); ret=np.full(n,lam)
    for i in range(steps):
        old=x.copy();x+=h*t[:,0]
        hit=x>=0
        if hit.any():
            ret[inds[hit]]=h*(i-old[hit]/(x[hit]-old[hit]))
            keep=~hit;x=x[keep];t=t[keep];inds=inds[keep]
        if not len(inds) or i+1==steps:break
        t=step_vmf(t,k,rng)
    val=ret/2
    out=ROOT/'data'/'return_samples';out.mkdir(parents=True,exist_ok=True)
    filename=f'return_l{lam:g}_n{steps}_seed{seed}.npz'
    np.savez_compressed(out/filename,truncated_return=ret,lam=lam,steps=steps,n=n,seed=seed)
    return dict(lam=lam,steps=steps,dt=h,n=n,seed=seed,mean_half_truncated_return=float(val.mean()),
                se=float(val.std(ddof=1)/math.sqrt(n)),returned_fraction=float(np.mean(ret<lam)),sample_file=filename)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--campaign',choices=['full','refined','short'],default='full');a=p.parse_args()
    tasks=[]
    if a.campaign=='full':
        for j,lam in enumerate((.1,1.,3.,10.,30.,100.)):
            fine=max(1024,int(math.ceil(lam/.025/256))*256)
            for steps in (fine//2,fine):tasks.append((lam,steps,50000,1400000+j*10000+steps))
    elif a.campaign=='refined':
        tasks=[(30.,5120,60000,1815120),(100.,8192,60000,1908192),
               (100.,16384,60000,1916384)]
    else:
        tasks=[(.01,1024,120000,2501024),(.025,1024,120000,2511024),
               (.05,1024,120000,2521024)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows=[]
        for row in ex.map(job,tasks):print(row,flush=True);rows.append(row)
    name={'full':'flux_return_campaign.csv','refined':'flux_return_refined.csv',
          'short':'flux_return_short.csv'}[a.campaign]
    with (ROOT/'data'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
