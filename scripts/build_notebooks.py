#!/usr/bin/env python3
"""Build and execute one explanatory notebook per main-article figure.

Each notebook reads only archived files in ``data/`` (and the fitted
coefficients in ``data/rational_fit_coefficients.json``), reproduces the
quantities plotted, and renders the figure with the same functions used for
the article (``code/plot_main_figures.py``).  Run from the package root:

    python scripts/build_notebooks.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks'
NB.mkdir(exist_ok=True)

HEADER = '''import sys, json, math
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import plot_main_figures as pmf
pd.set_option('display.precision', 8)'''

LICENSE_NOTE = ('*Code cells: MIT licence (`LICENSE`). Explanatory text and the figure: CC BY 4.0 '
                '(`DATA_LICENSE.md`). Part of the code-and-data repository for "Mean span dimension '
                'of an ideal wormlike chain" (Y. Wang).*')

def nb(title, cells):
    n = nbf.v4.new_notebook()
    n.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    n.cells.append(nbf.v4.new_markdown_cell(f'# {title}\n\n{LICENSE_NOTE}'))
    n.cells.append(nbf.v4.new_code_cell(HEADER))
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == 'md' else nbf.v4.new_code_cell(src))
    return n

# ----------------------------------------------------------------- Fig. 1
fig1 = nb('Fig. 1: Projected span and its growth at records', [
('md', '''## What the figure shows

Fig. 1 is a concept figure. Panel (a) shows one conformation of an ideal wormlike chain (WLC) with
$L/p=6$, projected onto a plane that contains the measurement axis $\\hat{\\mathbf e}$. The span
$X_c$ is the distance between the two extreme planes normal to $\\hat{\\mathbf e}$; it is compared
with the projected end-to-end distance $|\\mathbf R\\cdot\\hat{\\mathbf e}|$. Panel (b) plots the
projected position $Z(s)=\\mathbf r(s)\\cdot\\hat{\\mathbf e}$ along the contour together with its
running maximum and minimum; the span grows only on the *record* segments, where $Z$ exceeds its
previous maximum or falls below its previous minimum. This is the pathwise content of the
first-return identity of the article.

**The conformation is illustrative only.** It carries no quantitative claim and is not used
anywhere else in the paper.'''),
('md', '''## How the conformation is generated

The tangent $\\mathbf t(s)$ performs rotational diffusion on the unit sphere with
$\\langle\\mathbf t(s)\\cdot\\mathbf t(s')\\rangle=e^{-|s-s'|/p}$. In reduced units ($p=1$) this is
simulated by small Euler steps: a Gaussian increment of variance $h$ in the tangent plane, followed by
renormalization to unit length. Positions follow by trapezoidal integration of the tangent. The
random seed is chosen deterministically by `choose_chain()`, which scans a fixed range of seeds for a
conformation whose span clearly exceeds the projected end-to-end distance and whose extrema lie in
the interior of the contour, so that the figure illustrates the point it is meant to make.'''),
('code', '''seed = pmf.choose_chain()
s, r = pmf.illustrative_chain(seed=seed)
z = r[:, 2]
print('seed =', seed)
print('span X_c         =', z.max() - z.min())
print('|R.e|            =', abs(z[-1] - z[0]))
print('ratio span/|R.e| =', (z.max() - z.min()) / abs(z[-1] - z[0]))'''),
('md', '''## Records

The thick segments in panel (b) are the contour positions at which $Z$ equals its running maximum
$M(s)$ or running minimum $m(s)$. Their total length is the part of the contour that contributes to
the span.'''),
('code', '''M = np.maximum.accumulate(z); m = np.minimum.accumulate(z)
rec = (np.abs(z - M) < 1e-12) | (np.abs(z - m) < 1e-12)
print('fraction of contour on record segments:', rec.mean().round(3))'''),
('md', '''## Panel (c): the same trace read backward from a record

The point $s^*$ (diamond in panel b) lies on a growing record of the running minimum. Read backward
from $s^*$, the height of the trace above the record level, $Z(s^*-\\sigma)-Z(s^*)$, starts with slope
$v=|U(s^*)|$ and stays positive for all $0<\\sigma\\le s^*$: the reversed contour has not returned to the
record plane. The first-return identity states that the growth rate of the mean span is half the probability of this
event, with the starting speed weighted by $v$ (density $2v$ on $0<v<1$).'''),
('code', '''z_star_i = np.argmin(np.abs(s - 4.582))
grad = np.gradient(z, s)
back = z[: z_star_i + 1][::-1] - z[z_star_i]
print('s* =', s[z_star_i], '  v =', -grad[z_star_i], '  min height above the record for t>0:', back[1:].min())'''),
('md', '''## Panel (d): the Milne boundary layer

$\\langle\\Psi\\rangle_u(y)=\\tfrac12\\int_{-1}^1\\Psi(y,u)\\,du$ is the angle-averaged solution of the stationary
Milne problem at angular order $m=128$ (`code/build_boundary_moments.py`). Its far field is
$\\langle y-u+z_M\\rangle_u=y+z_M$, which extrapolates to zero at $y=-z_M$, a distance $z_Mp$ beyond the
record plane. Near the record the profile lies below the far field because the tangents there are not
equilibrated; at the plane it equals $M_0/2$.'''),
('code', '''from build_boundary_moments import stationary_milne
sol = stationary_milne(128)
psi0 = 0.5*(sol.evaluate([0.0]) @ sol.weights)[0]
print('z_M(m=128) =', sol.z, '   <Psi>_u at the record =', psi0, '   far field there =', sol.z)'''),
('md', '## Render the figure'),
('code', '''fig, meta = pmf.figure1()
fig'''),
])

# ----------------------------------------------------------------- Fig. 2
fig2 = nb('Fig. 2: Mean span through the rod-to-coil crossover', [
('md', '''## What the figure shows

Panel (a) shows the compensated mean span $X/\\sqrt{Lp}=F(\\lambda)/\\sqrt\\lambda$ at the 67
nodes of the spectral first-return solution (tabulated in full in the supplementary material), together with the two-term
short-chain expansion, $F=\\lambda/2-37\\lambda^2/768$, and the three-term long-chain
expansion, $F=4\\sqrt{\\lambda/3\\pi}-2z_M+B/\\sqrt\\lambda$ with $z_M=0.71821164$ and
$B=0.6468$. Panel (b) divides the same spans by the exact root-mean-square radius of gyration of the
WLC, the radius-of-gyration formula, and compares the ratio with its short-chain expansion and the conditional
long-chain expansion.'''),
('md', '''## Source data

`data/wlc_mean_span_reference.csv` holds, for each node $\\lambda$, the finite-order values
$F_{32},F_{48},F_{64},F_{96}$, the continuum extrapolation `F_continuum`, two alternative
extrapolations, the inversion check, the check order `F_m128_check`, and the internal sensitivity. All
67 rows were produced by `code/build_reference_curve.py`. Only `lambda` and `F_continuum` are
plotted; the sensitivities are smaller than the symbols.'''),
('code', '''ref = pd.read_csv(ROOT / 'data' / 'wlc_mean_span_reference.csv')
print(len(ref), 'nodes;  max relative sensitivity =', ref.epsilon_internal_rel.max())
ref[['lambda', 'F_m32', 'F_m96', 'F_continuum', 'epsilon_internal_abs']].iloc[[0, 20, 44, 57, 66]]'''),
('md', '''## Quantities derived in the notebook

The expansions are drawn only where their error is at most about 1 %: the short-chain
expansion for $\\lambda\\le1$ and the long-chain expansion for $\\lambda\\ge3$ in panel (a); for the ratio in
panel (b) the corresponding ranges are $\\lambda\\le8$ and $\\lambda\\ge30$.

The radius of gyration uses the closed form $g(\\lambda)=R_g^2/p^2=\\lambda/3-1+2/\\lambda-2(1-e^{-\\lambda})/\\lambda^2$,
with a power series below $\\lambda=0.3$ to avoid cancellation. The expansions are evaluated from
their printed coefficients; nothing is fitted here. The table below reproduces the accuracy statements
of the article.'''),
('code', '''lam = ref['lambda'].to_numpy(); F = ref.F_continuum.to_numpy()
Rg = np.sqrt(pmf.rg2_over_p2(lam))
short = pmf.short_expansion(lam); longc = pmf.long_expansion(lam)
tab = pd.DataFrame({'lambda': lam, 'F': F, 'X/Rg': F/Rg,
                    'short-chain error (%)': 100*np.abs(short/F-1),
                    'long-chain error (%)': 100*np.abs(longc/F-1)})
tab[tab['lambda'].isin([0.1, 0.3, 0.5, 1, 5, 10, 100, 5000, 1e5])].round(6)'''),
('code', '''print('rod limit  sqrt(3)     =', math.sqrt(3))
print('coil limit 4/sqrt(pi)  =', 4/math.sqrt(math.pi))
print('X/Rg at 1e5            =', (F/Rg)[-1], ' (%.2f%% below coil value)' % (100*(1-(F/Rg)[-1]/(4/math.sqrt(math.pi)))))'''),
('md', '## Render the figure'),
('code', '''fig = pmf.figure2()
fig'''),
])

# ----------------------------------------------------------------- Fig. 3
fig3 = nb('Fig. 3: Asymptotic coefficients', [
('md', '''## What the figure shows

Four tests of the asymptotic coefficients derived in the article.

* **(a)** The short-chain deficit $(\\lambda/2-F)/\\lambda^2$ from refined spectral solutions
  (angular orders up to 512 with first-alias-corrected inversion, `data/stiff_alias_summary.csv`)
  approaches $37/768$ as $\\lambda\\to0$.
* **(b)** The excess of the span over the mean projected end-to-end distance,
  $[X-\\langle|\\mathbf R\\cdot\\hat{\\mathbf e}|\\rangle]/(p\\lambda^2)$, from polygonal path sampling
  with von Mises–Fisher tangent steps (`data/endpoint_excursion_diagnostic.csv`), compared with the
  predicted $9/256$. Bars are two sampling standard errors.
* **(c)** The deficit of the spectral span relative to the leading Gaussian law,
  $4\\sqrt{\\lambda/3\\pi}-F$, plotted against $\\lambda^{-1/2}$ for $\\lambda\\ge5$; the intercept is
  $2z_M$ and the slope is $-B$.
* **(d)** The wall offset of the depletion layer, $2R_g/\\sqrt\\pi-\\delta$ with $\\delta=X/2$, from the same
  spectral values; it approaches $z_M$ with the slope $-(B/2+\\sqrt{3/\\pi})$ (the wall-offset expansion of the article).'''),
('code', '''stiff = pd.read_csv(ROOT / 'data' / 'stiff_alias_summary.csv')
stiff['deficit/lambda^2'] = (stiff['lambda']/2 - stiff.F_extrap) / stiff['lambda']**2
print('37/768 =', 37/768)
stiff[['lambda', 'F_extrap', 'extrapolation_sensitivity', 'deficit/lambda^2']]'''),
('code', '''ex = pd.read_csv(ROOT / 'data' / 'endpoint_excursion_diagnostic.csv')
ex = ex[ex['lambda'] <= 0.1].sort_values(['lambda', 'steps']).groupby('lambda').tail(1)
print('9/256 =', 9/256)
ex[['lambda', 'steps', 'n', 'excursion_over_lambda2', 'excursion_se']]'''),
('code', '''ref = pd.read_csv(ROOT / 'data' / 'wlc_mean_span_reference.csv')
big = ref[ref['lambda'] >= 5]
u = 1/np.sqrt(big['lambda']); off = pmf.A_GAUSS*np.sqrt(big['lambda']) - big.F_continuum
print('2 z_M =', 2*pmf.Z_M, '   B =', pmf.B_COEF)
# the line in panel (c) is NOT a fit to these points; it uses z_M (transport theory) and B (small-s analysis)
print('max |offset - (2z_M - B/sqrt(lambda))| for lambda>=5 :', np.max(np.abs(off - (2*pmf.Z_M - pmf.B_COEF*u))))'''),
('code', '''g = pmf.rg2_over_p2(big['lambda'].to_numpy())
wall = 2*np.sqrt(g)/np.sqrt(np.pi) - big.F_continuum.to_numpy()/2
pd.DataFrame({'lambda': big['lambda'], 'wall offset / p': wall,
              'z_M - (B/2 + sqrt(3/pi))/sqrt(lambda)': pmf.Z_M - (pmf.B_COEF/2 + np.sqrt(3/np.pi))*u}).iloc[::6]'''),
('md', '## Render the figure'),
('code', '''fig = pmf.figure3()
fig'''),
])

# ----------------------------------------------------------------- Fig. 4
fig4 = nb('Fig. 4: The practical formula', [
('md', '''## What the figure shows

the practical formula of the article is a rational function of $w=\\sqrt\\lambda$ that reproduces the rod limit,
the coefficient $37/768$, the Gaussian amplitude, the offset $-2z_M$ and the coefficient $B$ by
construction; its two remaining shape parameters were fitted by minimax to 34 of the spectral nodes
(`code/fit_correlation.py`). Panel (a) draws the practical formula through all 67 spectral values; panel (b) shows
its relative error at every node and distinguishes the 34 calibration nodes (filled), the 24 withheld
nodes interleaved with them (open circles, a test of smoothness) and the nine withheld nodes above
$\\lambda=5000$ (diamonds), which lie outside the calibration range and are the extrapolation test.'''),
('code', '''coef = json.load(open(ROOT / 'data' / 'rational_fit_coefficients.json'))['central']
coef'''),
('code', '''ref = pd.read_csv(ROOT / 'data' / 'wlc_mean_span_reference.csv')
lam = ref['lambda'].to_numpy(); F = ref.F_continuum.to_numpy()
err = 100*(pmf.rational(lam)/F - 1)
split = pd.read_csv(ROOT / 'data' / 'reference_nodes_with_split.csv').set_index('lambda')['set']
sets = pd.Series(lam).map(split).fillna('withheld (lambda > 5000)')
summary = pd.DataFrame({'lambda': lam, 'set': sets.values, 'error (%)': err})
print('max |error| overall      : %.4f %%' % np.abs(err).max())
for name, grp in summary.groupby('set'):
    print('max |error| %-16s: %.4f %% (%d nodes)' % (name, grp['error (%)'].abs().max(), len(grp)))'''),
('md', '''The constraints are visible in the coefficients: $d_R-b_R=37/384$ enforces the short-chain
coefficient, and the large-$w$ expansion of the practical formula reproduces the three long-chain terms.'''),
('code', '''print('d_R - b_R =', coef['d'] - coef['b'], '   37/384 =', 37/384)
w = np.sqrt(np.array([1e6, 1e8]))
print('F_R/sqrt(lambda) at lambda=1e6, 1e8:', pmf.rational(w**2)/w, '  (4/sqrt(3 pi) =', pmf.A_GAUSS, ')')'''),
('md', '## Render the figure'),
('code', '''fig = pmf.figure4()
fig'''),
])

for name, n in [('fig1', fig1), ('fig2', fig2), ('fig3', fig3), ('fig4', fig4)]:
    path = NB / f'{name}.ipynb'
    nbf.write(n, path)
    subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                    '--inplace', '--ExecutePreprocessor.timeout=600', str(path)],
                   cwd=NB, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('built and executed', path.name)
