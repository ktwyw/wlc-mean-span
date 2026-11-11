#!/usr/bin/env python3
"""High-order, first-alias-cancelled stiff-chain evaluation.

Companion 3t transforms use m=96 for cost; representative higher-order
companions quantify the resulting correction sensitivity. Neither angular
extrapolation nor all inversion errors have a certified continuum bound.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,json,math
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from wlc_pn_transfer import BromwichSettings,invert_euler_bromwich
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data'
def job(j):
    t,m,A,role=j;v=invert_euler_bromwich(t,m,BromwichSettings(A,32,32))
    return {'lambda':t,'m':m,'A':A,'role':role,'F_raw':v}
def fit(ms,vals,powers):
    m=np.asarray(ms,float); X=np.column_stack([np.ones(len(m))]+[m**-p for p in powers]);return float(np.linalg.lstsq(X,vals,rcond=None)[0][0])
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    nodes=(.0125,.025,.05,.075,.1,.2);orders=(192,256,320,384,512);jobs=[]
    for t in nodes:
      for m in orders:jobs.append((t,m,16.,'primary'))
      jobs.append((3*t,96,16.,'companion'))
    for t in (.0125,.025,.1):jobs.append((3*t,384,16.,'companion_check'))
    for t in (.0125,.05,.1):
      jobs.extend([(t,384,18.,'inversion_check'),(3*t,96,18.,'companion_check_A18')])
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
      fs={ex.submit(job,j):j for j in jobs}
      for i,f in enumerate(as_completed(fs),1):
        row=f.result();rows.append(row);print(f"{i}/{len(jobs)} {row['role']} lambda={row['lambda']:g} m={row['m']} A={row['A']:g} F={row['F_raw']:.13g}",flush=True)
    rows.sort(key=lambda z:(z['lambda'],z['role'],z['m'],z['A']))
    with (D/'stiff_alias_raw.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=[]
    for t in nodes:
      match=lambda role,lam:[z for z in rows if z['role']==role and abs(z['lambda']-lam)<1e-14]
      comp=match('companion',3*t)[0]['F_raw'];pvals={z['m']:z['F_raw']-math.exp(-16)*comp for z in match('primary',t)}
      base=fit(orders,[pvals[m] for m in orders],(3,4));variations=[
          fit((256,320,384,512),[pvals[m] for m in (256,320,384,512)],(3,4)),
          fit((320,384,512),[pvals[m] for m in (320,384,512)],(3,4)),
          fit((256,320,384,512),[pvals[m] for m in (256,320,384,512)],(3,)),
          fit(orders,[pvals[m] for m in orders],(3,4,5))]
      check=match('companion_check',3*t);comp_err=math.exp(-16)*abs(check[0]['F_raw']-comp) if check else 0.
      alt=match('inversion_check',t)
      if alt:
        c18=match('companion_check_A18',3*t)[0]['F_raw'];iv=abs((alt[0]['F_raw']-math.exp(-18)*c18)-pvals[384])
      else:iv=0.
      sens=max([abs(base-v) for v in variations]+[iv,comp_err])
      out.append({'lambda':t,'F_extrap':base,'extrapolation_sensitivity':sens,'cubic_residual':(base-t/2+(37/768)*t*t)/t**3,
                  'cubic_sensitivity':sens/t**3,'inversion_check_abs':iv,'companion_order_check_abs':comp_err,
                  'F_m512_corrected':pvals[512],**{f'variant_{i}':v for i,v in enumerate(variations)}})
    with (D/'stiff_alias_summary.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    models=[]
    for lo in (.0125,.025,.05):
      sel=[r for r in out if r['lambda']>=lo];x=np.array([z['lambda'] for z in sel]);y=np.array([z['cubic_residual'] for z in sel])
      for name,X in [('linear',np.column_stack((np.ones(len(x)),x))),('quadratic',np.column_stack((np.ones(len(x)),x,x*x))),('sqrt_plus_linear',np.column_stack((np.ones(len(x)),np.sqrt(x),x)))]:
        b=np.linalg.lstsq(X,y,rcond=None)[0];models.append({'lo':lo,'model':name,'C3':float(b[0]),'rms_residual':float(np.sqrt(np.mean((X@b-y)**2))),'coefficients':b.tolist()})
    (D/'stiff_alias_model_fits.json').write_text(json.dumps({'fits':models,'qualification':'Model sensitivity is exploratory; no proof of pure cubic expansion or certified continuum error.'},indent=2)+'\n')
    print('Alias-corrected stiff diagnostics complete.',flush=True)
if __name__=='__main__':main()
