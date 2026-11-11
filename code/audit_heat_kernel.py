#!/usr/bin/env python3
"""Grid refinement of inverse-CDF spherical heat-kernel quadrature."""
import csv,json
from pathlib import Path
from sphere_heat_kernel import heat_inverse
ROOT=Path(__file__).resolve().parents[1]
configs=[.1/512,1/1024,10/2048,100/8192];rows=[]
for h in configs:
 for grid in (12001,24001,48001):
  _,_,d=heat_inverse(h,grid)
  rows.append({'h':h,'grid':grid,'lmax':d['lmax'],'max_legendre_moment_difference':d['max_moment_error'],'max_cdf_monotone_adjustment':d['max_monotone_clip']})
with (ROOT/'data/heat_kernel_grid_audit.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
