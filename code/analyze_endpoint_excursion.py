#!/usr/bin/env python3
"""Analyze E[range-|projected endpoint|] and test exact radial moments."""
import csv,math
from pathlib import Path
import numpy as np
from wlc_moments import radial_moments
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';S=D/'span_samples'

def main():
    rows=[]
    for f in sorted(S.glob('stiff_d3_*.npz'))+sorted(S.glob('endpoint_*.npz')):
        with np.load(f) as a:
            if 'endpoint' not in a:continue
            lam=float(a['lam']);steps=int(a['steps']);w=a['span'];r=a['endpoint'];n=len(r)
        gap=(w-np.abs(r)).mean(axis=1)/lam**2
        rv=np.sum(r*r,axis=1);m2,m4=radial_moments(lam)
        rows.append({'lambda':lam,'steps':steps,'n':n,'seed_file':f.name,
          'excursion_over_lambda2':float(gap.mean()),'excursion_se':float(gap.std(ddof=1)/math.sqrt(n)),
          'sample_R2':float(rv.mean()),'R2_se':float(rv.std(ddof=1)/math.sqrt(n)),'exact_R2':float(m2),
          'sample_R4':float((rv*rv).mean()),'R4_se':float((rv*rv).std(ddof=1)/math.sqrt(n)),'exact_R4':float(m4)})
    rows.sort(key=lambda z:(z['lambda'],z['steps']))
    with (D/'endpoint_excursion_diagnostic.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote',len(rows),'endpoint-excursion and radial-moment checks.')
    for r in rows:
        print(r['lambda'],r['steps'],f"gap={r['excursion_over_lambda2']:.6g}+/-{r['excursion_se']:.2g}",
              f"R2 z={(r['sample_R2']-r['exact_R2'])/r['R2_se']:.2f}",
              f"R4 z={(r['sample_R4']-r['exact_R4'])/r['R4_se']:.2f}")
if __name__=='__main__':main()
