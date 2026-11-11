#!/usr/bin/env python3
"""Recompute span where the leading rod and Gaussian asymptotes intersect."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json, math
from pathlib import Path
import numpy as np
from wlc_pn_transfer import BromwichSettings,invert_euler_bromwich
ROOT=Path(__file__).resolve().parents[1]
t=64/(3*math.pi); orders=(32,48,64,96)
y=np.array([invert_euler_bromwich(t,m,BromwichSettings(22,28,28)) for m in orders])
A=np.array([[1,m**-3,m**-4] for m in orders]);f=float(np.linalg.lstsq(A,y,rcond=None)[0][0])
record={'lambda_star':t,'angular_orders':orders,'F_orders':y.tolist(),'F_extrapolated':f,
        'intersection_leading_value':t/2,'relative_excess_percent':100*(t/(2*f)-1),
        'qualification':'The angular-order extrapolation is empirical and is not a certified enclosure.'}
(ROOT/'validation/crossover_intersection.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
