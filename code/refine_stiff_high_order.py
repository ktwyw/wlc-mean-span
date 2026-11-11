#!/usr/bin/env python3
"""Higher-angular-order stiff-span diagnostics and model-sensitivity fits."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from wlc_pn_transfer import BromwichSettings,invert_euler_bromwich
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data'
def job(arg):
 lam,m,check=arg; b=BromwichSettings(24,32,32) if check else BromwichSettings(22,28,28)
 return {'lambda':lam,'m':m,'inversion':'check' if check else 'primary','F_m':invert_euler_bromwich(lam,m,b)}
def fit(ms,vals,powers):
 m=np.asarray(ms,float);A=np.column_stack([np.ones(len(m))]+[m**-p for p in powers]);return float(np.linalg.lstsq(A,vals,rcond=None)[0][0])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 nodes=(.0125,.025,.05,.075,.1,.2);order=(96,128,160,192,256,320,384,512)
 with (D/'stiff_refinement_orders.csv').open() as f:arch=list(csv.DictReader(f))
 rows=[{'lambda':float(r['lambda']),'m':int(r['m']),'inversion':r['inversion'],'F_m':float(r['F_m'])}
       for r in arch if float(r['lambda']) in nodes and int(r['m']) in order]
 have={(r['lambda'],r['m'],r['inversion']) for r in rows};jobs=[]
 for lam in nodes:
  for m in order:
   if (lam,m,'primary') not in have:jobs.append((lam,m,False))
 for lam in (.0125,.025,.1):jobs.append((lam,384,True))
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs={ex.submit(job,j):j for j in jobs}
  for i,f in enumerate(as_completed(fs),1):
   r=f.result();rows.append(r);print(f"{i}/{len(jobs)} lambda={r['lambda']} m={r['m']} {r['inversion']} F={r['F_m']:.14g}",flush=True)
 rows.sort(key=lambda r:(r['lambda'],r['m'],r['inversion']))
 with (D/'stiff_high_order_values.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 out=[]
 for lam in nodes:
  p={r['m']:r['F_m'] for r in rows if r['lambda']==lam and r['inversion']=='primary'}
  primary=fit([160,192,256,320,384,512],[p[j] for j in [160,192,256,320,384,512]],(3,4))
  variants={
   'last5_m3_m4':fit([192,256,320,384,512],[p[j] for j in [192,256,320,384,512]],(3,4)),
   'last4_m3_m4':fit([256,320,384,512],[p[j] for j in [256,320,384,512]],(3,4)),
   'last3_m3_m4':fit([320,384,512],[p[j] for j in [320,384,512]],(3,4)),
   'last3_m3':fit([320,384,512],[p[j] for j in [320,384,512]],(3,)),
   'last5_m3_m4_m5':fit([192,256,320,384,512],[p[j] for j in [192,256,320,384,512]],(3,4,5)),
  }
  alt=[r['F_m']-p[r['m']] for r in rows if r['lambda']==lam and r['inversion']=='check']
  sensitivity=max([abs(primary-v) for v in variants.values()]+[abs(x) for x in alt]+[0.])
  out.append({'lambda':lam,'F_extrap':primary,'extrapolation_sensitivity':sensitivity,
              'cubic_residual':(primary-lam/2+(37/768)*lam**2)/lam**3,
              'cubic_sensitivity':sensitivity/lam**3,'inversion_max_difference':max([abs(x) for x in alt]+[0.]),**variants})
 with (D/'stiff_high_order_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 models=[]
 for lo in (.0125,.025,.05):
  sel=[r for r in out if r['lambda']>=lo];x=np.array([r['lambda'] for r in sel]);y=np.array([r['cubic_residual'] for r in sel])
  for name,A in [('linear',np.column_stack((np.ones(len(x)),x))),('quadratic',np.column_stack((np.ones(len(x)),x,x*x))),('sqrt_plus_linear',np.column_stack((np.ones(len(x)),np.sqrt(x),x)))]:
   beta=np.linalg.lstsq(A,y,rcond=None)[0];models.append({'lo':lo,'model':name,'C3':float(beta[0]),'rms_residual':float(np.sqrt(np.mean((A@beta-y)**2))),'coefficients':beta.tolist()})
 (D/'stiff_high_order_model_fits.json').write_text(json.dumps({'fits':models,'qualification':'Exploratory fits; compare model and window dependence. No proof of a pure cubic term or certified errors.'},indent=2)+'\n')
 print('High-order results complete.',flush=True)
if __name__=='__main__':main()
