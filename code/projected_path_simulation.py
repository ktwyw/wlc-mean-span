#!/usr/bin/env python3
"""Seeded 2D/3D WLC polygon paths for mean-span and endpoint diagnostics.

The 3D vMF transition matches the tangent correlation but is not the exact
spherical heat kernel at finite step; position uses straight segments. The
2D angular increment is exact at steps, but the position integration remains
discrete. Only extrema, total variation and endpoint are retained.
"""
from __future__ import annotations
import argparse,csv,math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from wlc_path_simulation import kappa_for_step,sample_uniform_sphere
ROOT=Path(__file__).resolve().parents[1]

def step_vmf(t,kappa,rng):
    n=len(t);z=np.clip(rng.random(n),np.finfo(float).tiny,1.)
    e2=math.exp(-2*kappa) if kappa<350 else 0.
    c=np.clip(1+np.log(z+(1-z)*e2)/kappa,-1.,1.)
    phi=2*np.pi*rng.random(n);norm=np.hypot(t[:,0],t[:,1]);e1=np.empty_like(t);safe=norm>1e-14
    e1[safe,0]=-t[safe,1]/norm[safe];e1[safe,1]=t[safe,0]/norm[safe];e1[safe,2]=0.;e1[~safe]=(1.,0.,0.)
    e2v=np.cross(t,e1);s=np.sqrt(np.maximum(0.,1-c*c))
    return c[:,None]*t+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2v)

def generate(lam,steps,n,dimension,seed):
    rng=np.random.default_rng(seed);h=lam/steps
    if dimension==2:
        angle=2*np.pi*rng.random(n);tangent=np.column_stack((np.cos(angle),np.sin(angle)))
    elif dimension==3:
        tangent=sample_uniform_sphere(rng,n);kappa=kappa_for_step(h)
    else:raise ValueError('dimension must be 2 or 3')
    pos=np.zeros((n,dimension));lo=pos.copy();hi=pos.copy();tv=pos.copy()
    for i in range(steps):
        pos+=h*tangent;tv+=h*np.abs(tangent);lo=np.minimum(lo,pos);hi=np.maximum(hi,pos)
        if i+1==steps:break
        if dimension==2:
            a=np.sqrt(2*h)*rng.standard_normal(n);co=np.cos(a);si=np.sin(a);x=tangent[:,0];y=tangent[:,1]
            tangent=np.column_stack((co*x-si*y,si*x+co*y))
        else:tangent=step_vmf(tangent,kappa,rng)
    span=hi-lo;gap=tv-span
    if np.min(gap)<-5e-11*max(1.,lam):raise ArithmeticError('Range exceeds total variation')
    return span,np.maximum(gap,0.),pos

def job(arg):
    kind,d,lam,steps,n,seed,filename=arg;span,gap,end=generate(lam,steps,n,d,seed)
    out=ROOT/'data/span_samples';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/filename,span=span,gap=gap,endpoint=end,lam=lam,steps=steps,n=n,dimension=d,seed=seed)
    mu=2/math.pi if d==2 else .5;sample=gap.mean(axis=1)/(mu*lam**2)
    return dict(kind=kind,dimension=d,lam=lam,steps=steps,dt=lam/steps,n=n,seed=seed,
       mean_span=float(span.mean()),mean_span_se=float(span.mean(axis=1).std(ddof=1)/math.sqrt(n)),
       grazing_ratio=float(sample.mean()),grazing_ratio_se=float(sample.std(ddof=1)/math.sqrt(n)),sample_file=filename)

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',choices=['grazing','endpoint'],required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();jobs=[]
    if a.campaign=='grazing':
        for d in (2,3):
            for j,lam in enumerate((.01,.025,.05,.1)):
                for steps in (256,1024):
                    seed=910000+d*100000+j*10000+steps
                    filename=f'grazing_d{d}_l{lam:g}_n{steps}_seed{seed}.npz'
                    jobs.append(('grazing',d,lam,steps,40000,seed,filename))
    else:
        for j,lam in enumerate((.005,.01,.02,.05,.1)):
            n=160000 if lam<=.01 else 80000;steps=512;seed=2100000+j*10000+steps
            filename=f'stiff_d3_l{lam:g}_n{steps}_seed{seed}.npz'
            jobs.append(('endpoint',3,lam,steps,n,seed,filename))
        jobs.append(('endpoint',3,.1,2048,60000,3142048,'stiff_d3_l0.1_n2048_seed3142048.npz'))
        jobs.extend([('endpoint',3,30.,5120,40000,4135120,'endpoint_l30_n5120_seed4135120.npz'),
                     ('endpoint',3,100.,16384,30000,4216384,'endpoint_l100_n16384_seed4216384.npz')])
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results=[]
        for r in ex.map(job,jobs):print(r,flush=True);results.append(r)
    name='grazing_simulation.csv' if a.campaign=='grazing' else 'endpoint_campaign.csv'
    with (ROOT/'data'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
if __name__=='__main__':main()
