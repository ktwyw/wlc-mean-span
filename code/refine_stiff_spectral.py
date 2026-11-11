#!/usr/bin/env python3
"""Refine short-chain finite-angular-order span data and extrapolation diagnostics.

The m^-3/m^-4 extrapolation is empirical, not a proved convergence rate. The
third-order coefficient extracted from a fit is exploratory and is not an
analytical proof that such an expansion exists.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from wlc_pn_transfer import BromwichSettings,invert_euler_bromwich
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'

def task(job):
    lam,m,alt=job
    opt=BromwichSettings(24,32,32) if alt else BromwichSettings(22,28,28)
    f=invert_euler_bromwich(lam,m,opt,method='eig')
    return {'lambda':lam,'m':m,'inversion':'check' if alt else 'primary','F_m':f}

def extrap(ms,fs,powers):
    x=np.column_stack([np.ones(len(ms))]+[np.asarray(ms,dtype=float)**(-k) for k in powers])
    return np.linalg.lstsq(x,np.asarray(fs),rcond=None)[0][0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    nodes=(.025,.05,.075,.1,.15,.2,.3,.5);orders=(96,128,160,192,256,320)
    jobs=[(x,m,False) for x in nodes for m in orders]+[(x,192,True) for x in (.05,.1,.2,.5)]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(task,j):j for j in jobs}
        for count,f in enumerate(as_completed(fs),1):
            row=f.result();rows.append(row)
            print(f"{count}/{len(jobs)}: lambda={row['lambda']:g} m={row['m']} {row['inversion']} F={row['F_m']:.12g}",flush=True)
    rows.sort(key=lambda r:(r['lambda'],r['m'],r['inversion']))
    with (DATA/'stiff_refinement_orders.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=[]
    for x in nodes:
        p={r['m']:r['F_m'] for r in rows if r['lambda']==x and r['inversion']=='primary'}
        v0=extrap([128,160,192,256,320],[p[k] for k in (128,160,192,256,320)],(3,4))
        v1=extrap([160,192,256,320],[p[k] for k in (160,192,256,320)],(3,4))
        v2=extrap([192,256,320],[p[k] for k in (192,256,320)],(3,4))
        v3=extrap([256,320],[p[k] for k in (256,320)],(3,))
        alt=[r['F_m'] for r in rows if r['lambda']==x and r['m']==192 and r['inversion']=='check']
        delta=abs(alt[0]-p[192]) if alt else 0.
        diag=max(abs(v0-v1),abs(v0-v2),abs(v0-v3),delta)
        out.append({'lambda':x,'F_extrap':v0,'F_last4':v1,'F_last3':v2,'F_last2':v3,
                    'F_m320':p[320],'inversion_difference_m192':delta,'extrapolation_sensitivity':diag,
                    'cubic_residual':(v0-x/2+(37/768)*x*x)/x**3,
                    'cubic_sensitivity':diag/x**3})
    with (DATA/'stiff_refinement_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    vals=[]
    for lo,hi in ((.05,.2),(.075,.3),(.1,.5)):
        selected=[r for r in out if lo<=r['lambda']<=hi];x=np.array([r['lambda'] for r in selected]);y=np.array([r['cubic_residual'] for r in selected])
        for degree in (1,2):
            if len(x)<=degree:continue
            co=np.polynomial.polynomial.polyfit(x,y,degree)
            vals.append({'range':[lo,hi],'degree':degree,'coefficients':co.tolist(),'cubic_extrapolated':float(co[0])})
    (DATA/'stiff_cubic_fit_diagnostic.json').write_text(json.dumps({'fits':vals,'status':'Exploratory extrapolation assuming an analytic next term; not a proof or certified error.'},indent=2)+'\n')
    print('Wrote stiff refinement data and exploratory fit diagnostics.',flush=True)

if __name__=='__main__':main()
