#!/usr/bin/env python3
"""Numerical inversion of the S^2 rotational heat-kernel transition law.

The Legendre series targets the exact heat kernel at grid times. Series
truncation, monotone projection, and interpolation introduce numerical error;
this is not an exact sampling algorithm. See the returned diagnostics.
"""
from __future__ import annotations
import math
import numpy as np

def heat_inverse(h, grid=24001, tol=1e-15, qmax=44.):
    if not (0<h<=1): raise ValueError('0<h<=1 required')
    qmax=min(qmax,2/h)
    u=1-h*np.linspace(qmax,0,grid)
    if u[0]>-1: u=np.r_[-1.,u]
    u=np.clip(u,-1,1)
    p0=np.ones_like(u);p1=u.copy();cdf=(u+1)/2
    lmax=math.ceil((math.sqrt(1+8*math.log(1/tol)/h)-1)/2)+3
    for ell in range(1,lmax+1):
        p2=((2*ell+1)*u*p1-ell*p0)/(ell+1)
        cdf+=0.5*math.exp(-0.5*ell*(ell+1)*h)*(p2-p0)
        p0,p1=p1,p2
    raw=cdf.copy();cdf=np.maximum.accumulate(np.clip(cdf,0,1));cdf[0]=0.;cdf[-1]=1.
    # Integrate the numerical CDF's induced quantile law. These are deterministic
    # quadrature diagnostics of interpolation+truncation, not certified bounds.
    moments=[]
    p0=np.ones_like(u);p1=u.copy()
    for ell in range(1,6):
        if ell==1:p=p1
        else:
            p=((2*ell-1)*u*p1-(ell-1)*p0)/ell
            p0,p1=p1,p
        numerical=np.sum((p[:-1]+p[1:])*np.diff(cdf)/2)
        exact=math.exp(-ell*(ell+1)*h/2)
        moments.append({'ell':ell,'computed':float(numerical),'exact':exact,'absolute_difference':abs(float(numerical)-exact)})
    diagnostic={'h':h,'grid':len(u),'lmax':lmax,'qmax':qmax,'max_monotone_clip':float(np.max(np.abs(cdf-raw))),
                'max_moment_error':max(x['absolute_difference'] for x in moments),'moments':moments,
                'qualification':'finite series and numerical inverse CDF, no exact-sampling or continuum-path claim'}
    return u,cdf,diagnostic

def heat_step(t,u,cdf,rng):
    n=len(t);cosine=np.interp(rng.random(n),cdf,u);az=2*np.pi*rng.random(n)
    norm=np.hypot(t[:,0],t[:,1]);e1=np.empty_like(t);good=norm>1e-14
    e1[good,0]=-t[good,1]/norm[good];e1[good,1]=t[good,0]/norm[good];e1[good,2]=0.;e1[~good]=(1.,0.,0.)
    e2=np.cross(t,e1);sin=np.sqrt(np.maximum(0.,1-cosine*cosine))
    return cosine[:,None]*t+sin[:,None]*(np.cos(az)[:,None]*e1+np.sin(az)[:,None]*e2)
