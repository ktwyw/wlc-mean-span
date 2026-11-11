#!/usr/bin/env python3
"""Validate the standard-normal marginal used for coupled grazing paths."""
from pathlib import Path
import json,math
import numpy as np
from wlc_path_simulation import sample_uniform_sphere
ROOT=Path(__file__).resolve().parents[1];rng=np.random.default_rng(848120);n=250000
u=sample_uniform_sphere(rng,n);rad=np.sqrt(-2*np.log(rng.random(n)));angle=2*np.pi*rng.random(n)
norm=np.hypot(u[:,0],u[:,1]);e1=np.empty_like(u);mask=norm>1e-14
e1[mask,0]=-u[mask,1]/norm[mask];e1[mask,1]=u[mask,0]/norm[mask];e1[mask,2]=0.;e1[~mask]=(1.,0.,0.)
e2=np.cross(u,e1);perp=np.cos(angle)[:,None]*e1+np.sin(angle)[:,None]*e2
eta=rad*perp[:,0]/np.sqrt(np.maximum(1-u[:,0]**2,np.finfo(float).tiny))
rows=[]
for lo,hi in ((-1,-.5),(-.5,0),(0,.5),(.5,1)):
 a=eta[(u[:,0]>=lo)&(u[:,0]<hi)];rows.append({'range':[lo,hi],'n':len(a),'mean':float(a.mean()),'second_moment':float(np.mean(a*a)),'fourth_moment':float(np.mean(a**4))})
summary={'seed':848120,'n':n,'mean':float(eta.mean()),'second_moment':float(np.mean(eta**2)),'fourth_moment':float(np.mean(eta**4)),
         'conditional_bins':rows,'interpretation':'Sampling check of the coupling construction; the exact normal marginal follows from radial/azimuthal Gaussian symmetry.'}
(ROOT/'validation/coupled_gaussian_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
