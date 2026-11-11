#!/usr/bin/env python3
"""Couple the WLC grazing-return event to integrated Brownian velocity.

The WLC tangent targets the sphere heat-kernel transition (finite series and
inverse grid). Brownian normal increments are generated from the same radial
uniform and azimuth after rotation into the projected-tangent frame. This is
an exact standard-normal marginal, but both integrated contours are then
approximated by a piecewise-linear velocity with sampled turning points.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,math
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sphere_heat_kernel import heat_inverse
ROOT=Path(__file__).resolve().parents[1]
def updated_position(x,u,v,h):
    d=v-u
    with np.errstate(divide='ignore',invalid='ignore'):f=-u/d
    valid=(f>0)&(f<1)&np.isfinite(f);f=np.where(valid,f,0.)
    xm=x+h*(u*f+d*f*f/2);xn=x+h*(u+v)/2
    return xn,np.minimum(xn,np.where(valid,xm,np.inf))<0

def run(j):
    lam,steps,n,seed,grid=j; xis=np.array([.3,.7,1.2,2.]);m=len(xis);N=n*m;h=lam/steps;rng=np.random.default_rng(seed)
    q,cdf,diag=heat_inverse(h,grid);initial=np.repeat(xis,n);u0=np.sqrt(lam)*initial;phi=2*np.pi*rng.random(N);rad=np.sqrt(1-u0*u0)
    tangent=np.column_stack((u0,rad*np.cos(phi),rad*np.sin(phi)));W=initial.copy();x=np.zeros(N);z=x.copy();hit=np.zeros(N,bool);hitB=np.zeros(N,bool)
    for _ in range(steps):
      uni=np.clip(rng.random(N),np.finfo(float).tiny,1.);ang=2*np.pi*rng.random(N);c=np.interp(uni,cdf,q)
      norm=np.hypot(tangent[:,0],tangent[:,1]);e1=np.empty_like(tangent);good=norm>1e-14
      e1[good,0]=-tangent[good,1]/norm[good];e1[good,1]=tangent[good,0]/norm[good];e1[good,2]=0.;e1[~good]=(1.,0.,0.)
      e2=np.cross(tangent,e1);perp=np.cos(ang)[:,None]*e1+np.sin(ang)[:,None]*e2
      un=tangent[:,0];denom=np.sqrt(np.maximum(1-un*un,np.finfo(float).tiny))
      eta=np.sqrt(-2*np.log(uni))*perp[:,0]/denom
      new=c[:,None]*tangent+np.sqrt(np.maximum(0.,1-c*c))[:,None]*perp
      v=new[:,0]/np.sqrt(lam);w=W+eta/math.sqrt(steps)
      x,cross=updated_position(x,un/np.sqrt(lam),v,1/steps);z,crossB=updated_position(z,W,w,1/steps)
      hit|=cross;hitB|=crossB;tangent=new;W=w
    out=[]
    for j,xi in enumerate(xis):
      a=hit[j*n:(j+1)*n].astype(float);b=hitB[j*n:(j+1)*n].astype(float);diff=a-b
      out.append({'lambda':lam,'xi':float(xi),'steps':steps,'n':n,'seed':seed,'return_WLC':float(a.mean()),'SE_WLC':float(a.std(ddof=1)/math.sqrt(n)),
          'return_BM':float(b.mean()),'SE_BM':float(b.std(ddof=1)/math.sqrt(n)),'paired_difference':float(diff.mean()),
          'paired_SE':float(diff.std(ddof=1)/math.sqrt(n)),'kernel_moment_max_error':diag['max_moment_error']})
    folder=ROOT/'data/grazing_coupled_samples';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'grazing_coupled_l{lam:g}_n{steps}_N{n}_seed{seed}.npz',
       WLC=hit.reshape(m,n),Brownian=hitB.reshape(m,n),xis=xis,lam=lam,steps=steps,n=n,seed=seed)
    return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--paths',type=int,default=30000);p.add_argument('--steps',type=int,default=512);a=p.parse_args()
 cases=[(lam,a.steps,a.paths,7200000+i*10000+a.steps,24001) for i,lam in enumerate((.002,.01,.05))];rows=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  f={pool.submit(run,j):j for j in cases}
  for future in as_completed(f):
   rr=future.result();rows+=rr;print('lambda',rr[0]['lambda'],'max absolute paired difference',max(abs(v['paired_difference']) for v in rr),flush=True)
 rows.sort(key=lambda v:(v['lambda'],v['xi']))
 with (ROOT/'data/grazing_coupled_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Wrote coupled grazing-return records.',flush=True)
if __name__=='__main__':main()
