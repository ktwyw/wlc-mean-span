#!/usr/bin/env python3
"""Plot mean projected span / RMS radius of gyration for the ideal 3D WLC.

The points are the 67-node spectral reference values. The line
uses the previously calibrated rational correlation; it is not a new spectral
calculation. The exact gyration formula follows by integrating the WLC tangent
correlation. A short-length series prevents subtractive cancellation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from figure_io import save_figure

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'

def rg2_over_p2(lam):
    """Exact <Rg,c^2>/p^2, with a cancellation-safe small-lambda series."""
    x = np.asarray(lam, dtype=float)
    if np.any(x <= 0):
        raise ValueError('L/p must be positive')
    result = np.empty_like(x)
    short = x < 0.3
    z = x[short]
    # 2 sum_{n>=2} (-1)^n x^n/(n+2)!, through n=9.
    result[short] = z*z*(1/12 + z*(-1/60 + z*(1/360 + z*(-1/2520 +
                    z*(1/20160 + z*(-1/181440 + z*(1/1814400 - z/19958400)))))))
    z = x[~short]
    result[~short] = z/3 - 1 + 2/z + 2*np.expm1(-z)/(z*z)
    return result

def rational_span(lam, c):
    w = np.sqrt(lam)
    return 0.5*w*w*(1+c['a']*w+c['b']*w*w+c['c']*w**3)/(
        1+c['a']*w+c['d']*w*w+c['e']*w**3+c['f']*w**4)

def main():
    with (DATA/'wlc_mean_span_reference.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r['lambda']) for r in rows])
    F = np.array([float(r['F_continuum']) for r in rows])
    sensitivity = np.array([float(r['epsilon_internal_abs']) for r in rows])
    with (DATA/'rational_fit_coefficients.json').open() as f:
        c = json.load(f)['central']
    g = np.sqrt(rg2_over_p2(x))
    Fr = rational_span(x, c)
    with (DATA/'span_to_gyration_ratio.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['L_over_p','X_over_p_spectral','Rg_over_p_rms_exact',
                    'X_over_Rg_spectral','X_over_Rg_correlation',
                    'ratio_internal_sensitivity_abs','ratio_internal_sensitivity_rel'])
        w.writerows(zip(x, F, g, F/g, Fr/g, sensitivity/g, sensitivity/F))

    # The ratio is plotted in main-article Fig. 2(b) by plot_main_figures.py.
    print('Wrote 67-node span-to-gyration source table.')
    for lam in (0.01, 0.1, 1., 10., 100., 1000., 5000.):
        k = np.where(np.isclose(x,lam,rtol=0,atol=1e-12))[0]
        if len(k):
            i=k[0]; print(f'L/p={lam:g}: X/Rg={F[i]/g[i]:.8f}')
    print(f'Limits: rod={math.sqrt(3):.8f}, coil={4/math.sqrt(math.pi):.8f}')

if __name__ == '__main__':
    main()
