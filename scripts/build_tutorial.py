#!/usr/bin/env python3
"""Build and execute the tutorial notebooks in ``tutorial/``.

The tutorial follows the article topic by topic.  Each chapter states the
result, derives it (symbolically with SymPy where that is possible), verifies it
numerically against archived data or a live computation, and records what is
exact, derived or conditional.  Run from the package root:

    python scripts/build_tutorial.py            # build and execute all chapters
    python scripts/build_tutorial.py 05 06      # only the listed chapters
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TUT = ROOT / 'tutorial'
TUT.mkdir(exist_ok=True)

HEADER = '''import sys, json, math, time
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'tutorial' else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np, pandas as pd, sympy as sp
import matplotlib.pyplot as plt
pd.set_option('display.precision', 8)
plt.rcParams.update({'figure.dpi': 110, 'font.size': 9})
REF = pd.read_csv(ROOT / 'data' / 'wlc_mean_span_reference.csv')
LAM, F = REF['lambda'].to_numpy(), REF['F_continuum'].to_numpy()
Z_M, B_COEF, A_GAUSS = 0.71821164, 0.64680, 4/math.sqrt(3*math.pi)
def check(name, value, expected, tol):
    ok = abs(value - expected) <= tol
    print(('PASS' if ok else 'FAIL') + f'  {name}: {value:.10g}  (expected {expected:.10g}, tol {tol:g})')
    return ok'''

NOTE = ('*Code cells: MIT licence (`LICENSE`). Text and figures: CC BY 4.0 (`DATA_LICENSE.md`). '
        'Part of the code-and-data repository for "Mean span dimension of an ideal wormlike chain" (Y. Wang, 2026).*')


def nb(title, cells, header=True):
    n = nbf.v4.new_notebook()
    n.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    n.cells.append(nbf.v4.new_markdown_cell(f'# {title}\n\n{NOTE}'))
    if header:
        n.cells.append(nbf.v4.new_code_cell(HEADER))
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == 'md' else nbf.v4.new_code_cell(src))
    return n


CH = {}

# =============================================================== 00 overview
CH['00_overview'] = nb('Chapter 0: How to read this tutorial', [
('md', r'''## Purpose

This tutorial is a walk-through and verification of the article *Mean span dimension of an ideal
wormlike chain*. Every chapter follows one topic of the
article, derives its results in more detail than the article can, and then checks them: symbolic
identities with SymPy, numerical claims against the archived data in `data/`, and a few things by live
computation with the same code that produced the data. The last chapter recomputes the quantitative
statements of the article (values to their printed rounding, bounds one-sidedly) and prints a pass/fail list.

| Chapter | Topic | What is verified |
|---|---|---|
| 01 Model and span | Model and definitions | definitions; a direct simulation of the span; Fig. 1 |
| 02 First-return identity | The first-return identity | both derivations step by step; records at chain ends and flux-weighted returns by simulation; concavity; $F\le\lambda/2$ |
| 03 Radius of gyration and depletion | Span, radius of gyration and depletion | $g(\lambda)$ by symbolic integration; ratio limits; $7/1920$; the identity $\delta=X/2$ on sampled chains; the wall offset |
| 04 Short chains | Short chains | the reduction of the return-kernel integral to $37/192$ step by step; the dominated-convergence envelope; $37/768$; $9/256$; dimension independence |
| 05 Long chains | Long chains | the matching to first and second order step by step; $z_M$ computed live; the moment identities; the reciprocity value of $a_2$ against a direct solution; $B$; the telegraph process |
| 06 Spectral solver | Numerical solution | the method built step by step; the transfer formula; the alias series and its bound; convergence in angular order; a live reference value |
| 07 Independent checks | Independent tests of the solution | finite-volume and path-sampling checks; Fig. S1 |
| 08 Practical formula | The practical formula | constraints, sign audit and error of the practical formula; Fig. 4 |
| 09 Verification summary | all | the quoted numbers recomputed, bounds tested one-sidedly |

`tutorial/README.md` has an index from every result of the article to the chapter and section that
derives or checks it.

The four figure notebooks in `notebooks/` are the production versions of Figs. 1-4; Fig. S1 of the
supplementary material is drawn in chapter 7.

## Status labels

The article distinguishes three kinds of result, and this tutorial keeps the same labels:

* **exact** -- identities and inequalities that hold without approximation;
* **derived** -- limits obtained with an explicit justification of the limit interchange
  (the coefficient $37/768$, the leading Gaussian law);
* **conditional** -- terms that rest on a regular boundary-layer matching and a controlled
  transform inversion that are not proved (the offset $-2z_M$, the coefficient $B$, and the
  long-chain expansion of $X/R_g$, and the wall offset of the depletion layer).

Numerical sensitivities quoted anywhere are diagnostics, not certified error bounds.

## Notation

$L$ contour length, $p$ persistence length, $\lambda=L/p$, $X$ mean span (mean span dimension), $F=X/p$,
$Z$ reduced projected position, $U$ projected tangent ($-1\le U\le1$), $R_g$ root-mean-square
radius of gyration. Every notebook starts by loading the 67-node reference table
`data/wlc_mean_span_reference.csv` into `LAM` and `F`.'''),
], header=False)

# =============================================================== 01 model
CH['01_model_and_span'] = nb('Chapter 1: The wormlike chain and its projected span', [
('md', r'''## The model

The ideal three-dimensional wormlike chain (WLC) is a continuous curve $\mathbf r(s)$, $0\le s\le L$,
with unit tangent $\mathbf t(s)$ and bending energy
$$\frac{\mathcal H}{k_BT}=\frac p2\int_0^L\Big|\frac{\partial\mathbf t}{\partial s}\Big|^2\,ds ,$$
so that $\langle\mathbf t(s)\cdot\mathbf t(s')\rangle=e^{-|s-s'|/p}$. Both ends are free and the initial
tangent is isotropic. The tangent performs Brownian motion on the unit sphere with generator
$\tfrac12\Delta_{S^2}$ in units of $p$.

For a fixed unit vector $\hat{\mathbf e}$, the **projected span** of one conformation and the **mean span** are
$$X_c=\max_s\mathbf r(s)\cdot\hat{\mathbf e}-\min_s\mathbf r(s)\cdot\hat{\mathbf e},\qquad X=\langle X_c\rangle=p\,F(\lambda),\qquad\lambda=L/p .$$
By isotropy $X$ does not depend on $\hat{\mathbf e}$. In reduced units the projected position
$Z=\mathbf r\cdot\hat{\mathbf e}/p$ and projected tangent $U=\mathbf t\cdot\hat{\mathbf e}$ obey
$$dZ=U\,d\tau,\qquad dU=-U\,d\tau+\sqrt{1-U^2}\,dW_\tau ,$$
with $U$ uniform on $[-1,1]$ in equilibrium. The pair $(Z,U)$ is Markovian; $Z$ alone is not.'''),
('md', r'''## A direct simulation

The cleanest check of everything that follows is a direct simulation of chains. The sampler in
`code/sphere_heat_kernel.py` draws each tangent step from a numerically inverted series for the
spherical heat kernel (the exact transition law of the tangent over a step $h$, up to truncation and
interpolation), so the only remaining approximation is the straight-segment interpolation of the
position between nodes. We simulate 4000 chains at $\lambda=1$ with step $h=0.01$ and compare the mean
span, averaged over the three coordinate axes, with the reference value.'''),
('code', r'''from sphere_heat_kernel import heat_inverse, heat_step
def simulate_spans(lam, h, N, seed=1):
    u, cdf, diag = heat_inverse(h)
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(N, 3)); t /= np.linalg.norm(t, axis=1)[:, None]
    pos = np.zeros((N, 3)); mx = pos.copy(); mn = pos.copy()
    for _ in range(int(round(lam / h))):
        tn = heat_step(t, u, cdf, rng)
        pos = pos + 0.5 * (t + tn) * h; t = tn
        mx = np.maximum(mx, pos); mn = np.minimum(mn, pos)
    span = (mx - mn).mean(axis=1)             # average over the three axes within each chain
    endpoint = np.abs(pos).mean(axis=1)       # |R . e| averaged over axes
    return span, endpoint, diag
span, endpoint, diag = simulate_spans(1.0, 0.01, 4000)
F_ref = float(REF.loc[REF['lambda'] == 1.0, 'F_continuum'].iloc[0])
print(f'simulated mean span  F = {span.mean():.5f} +/- {span.std()/np.sqrt(len(span)):.5f}  (1 SE)')
print(f'reference value      F = {F_ref:.6f}')
print(f'mean |R.e|/p           = {endpoint.mean():.5f}   (the span exceeds it by {100*(span.mean()/endpoint.mean()-1):.1f} %)')
print('heat-kernel moment error of the sampler:', diag['max_moment_error'])'''),
('md', r'''The span always exceeds the projected end-to-end distance, by about 8 % at $\lambda=1$. The two
quantities separate because the chain can fold back and extend beyond either end. (Chapter 4 derives the
leading excess, $9\lambda^2/256$ for short chains.)

## Rod and coil limits

For a rigid rod the projected length is $L|\cos\theta|$ with $\theta$ uniform on the sphere, so
$X=L\langle|\cos\theta|\rangle=L/2$. For a Gaussian coil the projected position is Brownian motion and
$X=4R_g/\sqrt\pi$. Neither limit describes a chain of intermediate length, which is the subject of the
article:'''),
('code', r'''for lam in (0.01, 1.0, 6.790, 100.0):
    i = np.argmin(np.abs(LAM - lam)); l = LAM[i]
    print(f'lambda={l:8.3g}  F={F[i]:10.5f}   rod L/2={l/2:10.5f}   Gaussian 4 sqrt(lambda/3pi)={A_GAUSS*np.sqrt(l):10.5f}')
lam_cross = 64/(3*math.pi)
print(f'\nthe two limits cross at lambda = 64/(3 pi) = {lam_cross:.4f}, where both give {lam_cross/2:.4f}; the actual value there is', np.interp(lam_cross, LAM, F).round(4))'''),
('md', r'''## Figure 1

`notebooks/fig1.ipynb` draws one illustrative conformation and shows that the span grows only on the
*record* segments of the projected position, where $Z$ exceeds its running maximum or falls below its
running minimum. That observation is the starting point of Chapter 2.'''),
])

# =============================================================== 02 first-return identity
CH['02_first_return_identity'] = nb('Chapter 2: The first-return identity', [
('md', r'''## Statement

Let $v$ be drawn from the flux-weighted density $2v$ on $0<v<1$, start a projected contour on a plane
with outward speed $v$, and let $T$ be the reduced length at which it first returns to the plane. Then
$$F'(\lambda)=\tfrac12\Pr(T>\lambda),\qquad F(\lambda)=\tfrac12\,\mathbb E\min(T,\lambda).$$
**Status: exact.**

## Derivation 1: from the kinetic equation

Let $M_\lambda=\max_{\tau\le\lambda}Z_\tau$. By the symmetry $(Z,U)\to(-Z,-U)$ of the stationary
ensemble, $F=2\,\mathbb E M_\lambda$. For $y>0$ let $q(y,u;\lambda)=\Pr\{M_\lambda>y\mid U_0=u\}$. Because
$(Z,U)$ is Markovian, $q$ obeys the backward equation
$$\partial_\lambda q+u\,\partial_yq=\mathcal L_uq,\qquad \mathcal L_u=\tfrac12\partial_u(1-u^2)\partial_u,$$
with $q(0^+,u)=1$ for $u>0$ (a chain pointing at the plane reaches it at once) and $q\to0$ as $y\to\infty$.
Integrate over $y>0$ and $u\in[-1,1]$. The angular operator is conservative, $\int\mathcal L_uq\,du=0$.
The streaming term gives a boundary flux at $y=0$:
$\int_{-1}^1u\,q(0^+,u)\,du=\int_0^1u\,du-\int_0^1v\,q(0^+,-v)\,dv=\tfrac12-\int_0^1v\,q(0^+,-v)\,dv$.
Since $F=2\int\!\!\int q\,du\,dy\cdot\tfrac12$ (the factor $\tfrac12$ is the uniform angular average),
$$F'(\lambda)=\tfrac12-\int_0^1v\,q(0^+,-v;\lambda)\,dv .$$
For fixed $v>0$ the outgoing trace $q(0^+,-v;\lambda)$ is the probability that a contour started on the plane
with outward speed $v$ has returned by $\lambda$, i.e. the distribution function of $T_v$. Weighting with
$2v$ gives $\Pr(T>\lambda)=1-2\int_0^1v\,q(0^+,-v)\,dv=2F'(\lambda)$.

## Derivation 2: pathwise, without the kinetic equation

The span of a growing chain increases only while the endpoint is at its running maximum or minimum, at
rate $|U|$. Reversing the contour direction on $[0,\lambda]$ turns "the endpoint is a record" into "a
contour started on a plane, moving away from it, has not yet returned". Stationarity, time reversibility
and sign symmetry of $U$ are all that is used, together with boundedness of $U$; the initial speed is
weighted by $2v\varpi(v)/\mu$ with $\varpi$ the stationary density and $\mu=\mathbb E|U|$. For the WLC,
$\varpi=\tfrac12$ and $\mu=\tfrac12$. No independence of successive excursions is assumed.

## Numerical test

We estimate $\Pr(T>\lambda)$ by simulating flux-weighted returns with the heat-kernel sampler, and compare
with $2F'(\lambda)$ obtained by differentiating the practical formula of the article (which matches the
reference curve to 0.1 %).'''),
('code', r'''from sphere_heat_kernel import heat_inverse, heat_step
import plot_main_figures as pmf
def survival(lams, h=0.01, N=20000, seed=7):
    """Flux-weighted first-return survival Pr(T > lambda) by direct simulation."""
    u, cdf, _ = heat_inverse(h)
    rng = np.random.default_rng(seed)
    v = np.sqrt(rng.random(N))                       # density 2v on (0,1)
    phi = 2*np.pi*rng.random(N)
    t = np.column_stack([np.sqrt(1-v*v)*np.cos(phi), np.sqrt(1-v*v)*np.sin(phi), v])  # outward speed v along e=z
    z = np.zeros(N); alive = np.ones(N, bool); out = {}
    lam_max = max(lams); steps = int(round(lam_max/h)); targets = {int(round(l/h)): l for l in lams}
    for k in range(1, steps+1):
        tn = heat_step(t, u, cdf, rng)
        z = z + 0.5*(t[:, 2] + tn[:, 2])*h; t = tn
        alive &= z > 0
        if k in targets: out[targets[k]] = alive.mean()
    return out
lams = [0.1, 0.3, 1.0, 3.0]
surv = survival(lams)
eps = 1e-4
for l in lams:
    dF = (pmf.rational(l*(1+eps)) - pmf.rational(l*(1-eps)))/(2*l*eps)
    print(f'lambda={l:4g}   2 F\'(lambda) = {2*dF:.4f}    simulated Pr(T>lambda) = {surv[l]:.4f} +/- {math.sqrt(surv[l]*(1-surv[l])/20000):.4f}')'''),
('md', r'''The simulated survival probabilities agree with $2F'$ within sampling error (the practical formula
contributes an error of order $10^{-3}$ to $F'$ near $\lambda\sim1$, where its curvature is largest).

## Consequences checked on the reference table

* $\Pr(T>\lambda)$ cannot increase, so $F'$ is nonincreasing: $F$ is **concave**, and $F(\lambda)/\lambda$ is nonincreasing.
* $0<F'\le\tfrac12$ gives $F\le\lambda/2$ (the pathwise bound $X_c\le L$ alone would only give $F\le\lambda$).
* Subadditivity $F(\lambda_1+\lambda_2)\le F(\lambda_1)+F(\lambda_2)$ follows from concatenating contours.'''),
('code', r'''d1 = np.diff(F)/np.diff(LAM)                      # chord slopes between consecutive nodes
print('all chord slopes positive        :', bool(np.all(d1 > 0)))
print('chord slopes nonincreasing       :', bool(np.all(np.diff(d1) <= 1e-12)), '  (concavity)')
print('F/lambda nonincreasing           :', bool(np.all(np.diff(F/LAM) <= 0)))
print('F <= lambda/2 at every node      :', bool(np.all(F <= LAM/2)))
print('max chord slope                  :', d1.max().round(6), ' (<= 1/2)')
# subadditivity on pairs of nodes whose sum is also a node
pairs = [(a, b) for a in LAM for b in LAM if a <= b and any(np.isclose(a+b, LAM))]
Fi = lambda x: F[np.argmin(np.abs(LAM-x))]
print('subadditivity on', len(pairs), 'node pairs:', all(Fi(a+b) <= Fi(a)+Fi(b)+1e-12 for a, b in pairs))'''),
('md', r'''## The pathwise derivation, one step at a time

The article's derivation has five steps. Each is stated here and then tested on sampled chains.

**Step 1: the maximum advances only at a record.** $M_\tau=\max_{\tau'\le\tau}Z_{\tau'}$ is nondecreasing,
and it changes only while $Z_\tau=M_\tau$ and the contour moves outward. Since $dZ=U\,d\tau$,
$$\frac{dM_\tau}{d\tau}=U_\tau^+\,\mathbf 1_{\{Z_\tau=M_\tau\}},\qquad U^+=\max(U,0),$$
and likewise $-dm_\tau/d\tau=U_\tau^-\mathbf 1_{\{Z_\tau=m_\tau\}}$ for the minimum.

**Step 2: average (the record form of the growth rate).** $F=\langle M_\lambda-m_\lambda\rangle$, and the symmetry $(Z,U)\to(-Z,-U)$ makes
the two extremes contribute equally: $F'(\lambda)=2\langle U_\lambda^+\mathbf 1_{\{Z_\lambda=M_\lambda\}}\rangle$.
Only the *end* of the contour enters.

**Step 3: read the contour backward.** Let $Y_s=Z_\lambda-Z_{\lambda-s}$, $0\le s\le\lambda$, be the height of
the end above the contour point a reduced length $s$ before it. The end is the maximum if and only if
$Y_s\ge0$ for all $s$. Now $dY_s/ds=U_{\lambda-s}$, the reversed tangent. Rotational diffusion in
equilibrium is time reversible, so the reversed tangent is again a stationary rotational diffusion.
$Y$ is therefore the position of a contour that starts on a plane ($Y_0=0$) with speed $U_\lambda=v$
away from it, and the event is "no return to the plane within $\lambda$".

**Step 4: weight the speeds (the first-return identity).** $U_\lambda$ is uniform on $[-1,1]$ (density $\tfrac12$), so
$$F'(\lambda)=2\int_0^1\tfrac12\,v\,\Pr\nolimits_v(T>\lambda)\,dv=\tfrac12\int_0^1 2v\,\Pr\nolimits_v(T>\lambda)\,dv
=\tfrac12\Pr(T>\lambda),$$
with $v$ drawn from the density $2v$. Integrating from $F(0)=0$:
$F(\lambda)=\tfrac12\int_0^\lambda\Pr(T>\ell)\,d\ell=\tfrac12\,\mathbb E\min(T,\lambda)$.

**Step 5: the null touch.** The reversed event in step 3 has a weak inequality ($Y_s\ge0$), whereas $T$
is the first *strictly positive* return. They differ only if the contour touches the plane without
crossing it, which requires $(Z,U)$ to pass through the single point $(0,0)$. For this process that has
probability zero (supplementary material).

**Consequence: the derivative with respect to $p$ .** With $X=pF(L/p)$ at fixed $L$,
$\partial X/\partial p=F-\lambda F'$. From step 4,
$F-\lambda F'=\tfrac12\mathbb E[\min(T,\lambda)-\lambda\mathbf 1_{\{T>\lambda\}}]=\tfrac12\mathbb E[T\,\mathbf 1_{\{T\le\lambda\}}]\ge0$.

### Test of steps 1-2: growth of the span at the end of equilibrium chains

We sample equilibrium chains, look only at their ends, and average $|U_\lambda|$ over the chains whose end
is an outward record. This estimates $F'(\lambda)$ directly from the record form of the growth rate, without any reversal. It is
compared with the slope of the spectral solution, obtained by a central difference of the solver of
Chapter 6 at one angular order.'''),
('code', r'''from wlc_pn_transfer import invert_alias_corrected, BromwichSettings
_st = BromwichSettings(12.0, 32, 32)
def F_and_slope(l, rel=0.005, m=64):
    """F and F' from the spectral solver at one angular order (central difference; the finite-order error largely cancels in the slope)."""
    lo_, mid, hi_ = (invert_alias_corrected(x, m, _st) for x in (l*(1-rel), l, l*(1+rel)))
    return mid, (hi_ - lo_)/(2*rel*l)
SPEC = {l: F_and_slope(l) for l in (0.1, 0.3, 1.0, 3.0)}
Fval = lambda l: SPEC[l][0]; Fprime = lambda l: SPEC[l][1]

def end_record_rate(lam, h=0.005, N=40000, seed=11):
    """Mean growth rate of the span at the chain end: <|U| 1{end is an outward record}> (the record form of the growth rate)."""
    u, cdf, _ = heat_inverse(h)
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(N, 3)); t /= np.linalg.norm(t, axis=1)[:, None]
    pos = np.zeros((N, 3)); mx = pos.copy(); mn = pos.copy()
    for _ in range(int(round(lam/h))):
        tn = heat_step(t, u, cdf, rng)
        turn = t*tn < 0                                      # the projected tangent changes sign inside the step
        ext = pos + 0.5*h*t*t/np.where(turn, t - tn, 1.0)    # turning point of the quadratic position interpolant
        mx = np.where(turn, np.maximum(mx, ext), mx); mn = np.where(turn, np.minimum(mn, ext), mn)
        pos = pos + 0.5*(t + tn)*h; t = tn
        mx = np.maximum(mx, pos); mn = np.minimum(mn, pos)
    at_max = (pos >= mx) & (t > 0)                   # end is the maximum and points outward
    at_min = (pos <= mn) & (t < 0)                   # end is the minimum and points outward
    growth = (np.abs(t)*(at_max | at_min)).mean(axis=1)          # the three axes averaged within each chain
    return growth.mean(), growth.std()/math.sqrt(N), (at_max | at_min).mean()
for l in (0.3, 1.0, 3.0):
    g, se, frac = end_record_rate(l)
    print(f'lambda={l:3g}   <|U| 1(record)> = {g:.4f} +/- {se:.4f}    F\'(lambda), spectral = {Fprime(l):.4f}    (end is a record in {100*frac:.0f} % of chain-axis pairs)')'''),
('md', r'''### Test of steps 3-4: return lengths of flux-weighted starts

Now the other side of the identity. Contours are started *on* a plane with outward speed drawn from the
density $2v$, and the return length $T$ of each is recorded (up to $\lambda=3$). The two forms of the first-return identity and
the expression for $\partial X/\partial p$ are then averages over the same sample of $T$.'''),
('code', r'''def return_lengths(lam_max, h=0.005, N=40000, seed=7):
    """First-return lengths T of contours started on the plane with outward speed of density 2v."""
    u, cdf, _ = heat_inverse(h)
    rng = np.random.default_rng(seed)
    v = np.sqrt(rng.random(N)); phi = 2*np.pi*rng.random(N)
    t = np.column_stack([np.sqrt(1-v*v)*np.cos(phi), np.sqrt(1-v*v)*np.sin(phi), v])
    z = np.zeros(N); T = np.full(N, np.inf)
    for k in range(1, int(round(lam_max/h)) + 1):
        tn = heat_step(t, u, cdf, rng)
        a, b = t[:, 2], tn[:, 2]
        zn = z + 0.5*(a + b)*h
        dip = (a < 0) & (b > 0)                              # interior minimum of the quadratic position interpolant
        zmin = np.where(dip, z + 0.5*h*a*a/np.where(dip, a - b, 1.0), zn)
        hit = np.isinf(T) & (np.minimum(zn, zmin) <= 0)
        c2 = (b[hit] - a[hit])/(2*h)                         # first root of z + a x + c2 x^2 = 0 within the step
        disc = np.maximum(a[hit]**2 - 4*c2*z[hit], 0.0); quad = np.abs(c2) > 1e-14
        x = np.where(quad, (-a[hit] - np.sqrt(disc))/(2*np.where(quad, c2, 1.0)), -z[hit]/np.where(a[hit] != 0, a[hit], -1.0))
        T[hit] = (k-1)*h + np.clip(x, 0, h)
        z, t = zn, tn
    return T
T = return_lengths(3.0); n = len(T)
print('lambda   (1/2)Pr(T>lambda)     F\'(spectral) | (1/2)E min(T,lambda)       F(spectral) | (1/2)E[T 1(T<=lambda)]     F - lambda F\'')
for l in (0.1, 0.3, 1.0, 3.0):
    surv = (T > l).mean(); m = np.minimum(T, l); w = np.where(T <= l, T, 0.0)
    print(f'{l:5g}   {surv/2:.4f} +/- {math.sqrt(surv*(1-surv)/n)/2:.4f}   {Fprime(l):.4f}       | '
          f'{m.mean()/2:.5f} +/- {m.std()/2/math.sqrt(n):.5f}   {Fval(l):.5f}     | '
          f'{w.mean()/2:.5f} +/- {w.std()/2/math.sqrt(n):.5f}      {Fval(l)-l*Fprime(l):.5f}')'''),
('md', r'''The equilibrium-chain average of step 2 and the flux-weighted return statistics of step 4 give the same
$F'$, and both agree with the spectral solution of the kinetic equation, a third route. All uncertainties are
one sampling standard error, and the differences are within about one and a half of them. The paths have
a step of $0.005$; records and returns inside a step are located with the quadratic position interpolant,
because counting them only at the nodes biases both estimates by more than the sampling error.'''),
('md', r'''## Transform and derivatives (additional to the article)

For real $s>0$, $\widehat F(s)=\int_0^\infty e^{-s\lambda}F\,d\lambda=(1-\mathbb Ee^{-sT})/(2s^2)$. At fixed $L$,
$\partial_pX=F-\lambda F'=\tfrac12\mathbb E[T\,\mathbf 1_{T\le\lambda}]\ge0$ and the Hessian of $X$ in $(L,p)$ is
$p^{-1}F''\begin{pmatrix}1&-\lambda\\-\lambda&\lambda^2\end{pmatrix}$, negative semidefinite because
$F''\le0$: $X$ is jointly concave in its two length scales.'''),
('code', r'''lam, p, L = sp.symbols('lambda p L', positive=True)
Fs = sp.Function('F')
X = p*Fs(L/p)
H = sp.hessian(X, (L, p)).applyfunc(sp.simplify)
H = H.subs(L, lam*p).applyfunc(sp.simplify)
display(H)
print('determinant:', sp.simplify(H.det()), '   trace sign = sign of F\'\'(lambda) * (1+lambda^2)/p')'''),
])

# =============================================================== 03 Rg
CH['03_radius_of_gyration'] = nb('Chapter 3: The radius of gyration and the ratio X/R_g', [
('md', r'''## Derivation

$R_g^2=\big\langle L^{-1}\!\int_0^L|\mathbf r(s)-\mathbf r_{\rm cm}|^2ds\big\rangle
=(2L^2)^{-1}\!\int_0^L\!\!\int_0^L\langle|\mathbf r(s)-\mathbf r(s')|^2\rangle\,ds\,ds'$,
and the mean-square separation of two contour points at distance $h$ is $2p^2(h/p-1+e^{-h/p})$.
SymPy performs the double integral:'''),
('code', r'''lam, u = sp.symbols('lambda u', positive=True)
msd = 2*(u-1+sp.exp(-u))                                  # <|r(s)-r(s')|^2>/p^2 with u=|s-s'|/p
g = sp.simplify(1/lam**2 * sp.integrate((lam-u)*msd, (u, 0, lam)))   # (2L^2)^-1 * 2 * int_0^L (L-h) msd dh  (the 2 counts s<s' and s>s')
g_article = lam/3 - 1 + 2/lam - 2*(1-sp.exp(-lam))/lam**2
print('g(lambda) =', g)
print('matches the radius-of-gyration formula:', sp.simplify(g - g_article) == 0)
print('series   :', sp.series(g, lam, 0, 6))
print('limits   : rod R_g^2 -> lambda^2/12 ;  coil R_g^2 -> lambda/3')'''),
('md', r'''## The ratio and its limits

$X/R_g=F/\sqrt g$ tends to $\sqrt3$ for a rod ($F\to\lambda/2$, $g\to\lambda^2/12$) and to $4/\sqrt\pi$ for a
coil ($F\to4\sqrt{\lambda/3\pi}$, $g\to\lambda/3$). With the short-chain span $F=\lambda/2-37\lambda^2/768$ (Chapter 4):'''),
('code', r'''Fshort = lam/2 - sp.Rational(37,768)*lam**2
gser = sp.series(g/lam**2, lam, 0, 3).removeO()                  # g = lambda^2 (1/12 - lambda/60 + ...)
ratio_short = sp.series(Fshort/(lam*sp.sqrt(gser)), lam, 0, 2).removeO()
print('X/Rg (short) / sqrt(3) =', sp.simplify(ratio_short/sp.sqrt(3)), '   coefficient 1/10 - 37/384 =', sp.Rational(1,10)-sp.Rational(37,384))
zM, Bc = sp.symbols('z_M B')
Flong = 4*sp.sqrt(lam/(3*sp.pi)) - 2*zM + Bc/sp.sqrt(lam)
g_long = lam/3 - 1 + 2/lam                                        # exponentially small terms dropped
v = sp.symbols('v', positive=True)                                # v = 1/sqrt(lambda)
expr = (Flong/sp.sqrt(g_long)).subs(lam, 1/v**2)
print('X/Rg (long)  =', sp.series(expr, v, 0, 3).removeO())'''),
('md', r'''The long-chain coefficients are $-2\sqrt3\,z_M$ at order $\lambda^{-1/2}$ and $\sqrt3B+6/\sqrt\pi$ at order
$\lambda^{-1}$, as in the long-chain expansion of X/R_g. **Status:** the limits are exact; the short-chain slope $7/1920$ is
derived; the long-chain terms are conditional because they inherit $z_M$ and $B$.

## Numbers quoted in the article'''),
('code', r'''import plot_main_figures as pmf
ratio = F/np.sqrt(pmf.rg2_over_p2(LAM))
for x in (1, 10, 100, 1000, 5000, 1e5):
    print(f'lambda={x:>7g}   X/Rg = {float(ratio[LAM==x][0]):.4f}')
coil = 4/math.sqrt(math.pi)
print('below coil value at 5000 : %.2f %%' % (100*(1-float(ratio[LAM==5000][0])/coil)))
print('below coil value at 1e5  : %.2f %%' % (100*(1-float(ratio[LAM==1e5][0])/coil)))
lam1 = (2*math.sqrt(3)*Z_M/(0.01*coil))**2
print('1 %% approach (leading term):  lambda ~ %.3g' % lam1)
check('X/Rg at 5000', float(ratio[LAM==5000][0]), 2.2225, 5e-5)'''),
('md', r'''## The identity $\delta=X/2$, step by step

Take a hard wall at height zero with normal $\hat{\mathbf e}$, and dilute chains that the wall affects only
by removing the conformations that cross it. Fix a reference point $s$ on the contour and place it at
height $h$.

1. The conformation survives if its lowest point is above the wall, i.e. if $Z(s)-\min Z<h$, where
   $Z=\mathbf r\cdot\hat{\mathbf e}$. So $c_s(h)/c_{\rm b}=\Pr\{Z(s)-\min Z<h\}$.
2. The depletion thickness is $\delta=\int_0^\infty[1-c_s(h)/c_{\rm b}]\,dh=\int_0^\infty\Pr\{Z(s)-\min Z\ge h\}\,dh
   =\langle Z(s)-\min Z\rangle$, the mean of a nonnegative variable.
3. A rotation by $\pi$ about an axis in the wall plane maps $Z\to-Z$ and leaves the ensemble of
   unoriented chains unchanged, so $\langle Z(s)-\min Z\rangle=\langle\max Z-Z(s)\rangle$.
4. The two sum to $\langle\max Z-\min Z\rangle=X$. Hence $\delta=X/2$ for any reference point $s$.

Nothing in the argument is specific to the wormlike chain. The cell tests it on sampled chains at
$\lambda=1$ for four reference points; the two one-sided averages differ from point to point within sampling
error only.'''),
('code', r'''from sphere_heat_kernel import heat_inverse, heat_step
def projected_paths(lam, h=0.01, N=40000, seed=3):
    u_, cdf, _ = heat_inverse(h)
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(N, 3)); t /= np.linalg.norm(t, axis=1)[:, None]
    n = int(round(lam/h)); Zp = np.zeros((n+1, N))
    for k in range(n):
        tn = heat_step(t, u_, cdf, rng)
        Zp[k+1] = Zp[k] + 0.5*(t[:, 2] + tn[:, 2])*h; t = tn
    return Zp
Zp = projected_paths(1.0); lo, hi = Zp.min(axis=0), Zp.max(axis=0); N = Zp.shape[1]
F1 = float(F[LAM == 1.0][0])
print(f'X/(2p) from the reference table: {F1/2:.4f};  from these chains: {(hi-lo).mean()/2:.4f} +/- {(hi-lo).std()/2/math.sqrt(N):.4f}')
for frac, name in ((0, 'end'), (0.25, 'quarter point'), (0.5, 'midpoint'), (1.0, 'other end')):
    k = int(frac*(Zp.shape[0]-1)); d = Zp[k] - lo
    print(f'reference point: {name:13s}  <Z(s) - min Z> = {d.mean():.4f} +/- {d.std()/math.sqrt(N):.4f}    <max Z - Z(s)> = {(hi-Zp[k]).mean():.4f}')
k = Zp.shape[0]//2; d = Zp[k] - lo; hh = np.linspace(0, 1.0, 6)
print('\nconcentration profile of the midpoint, c(h)/c_b = Pr{Z(s) - min Z < h}:')
print('   h/p     :', np.round(hh, 2)); print('   c/c_b   :', np.round([(d < x).mean() for x in hh], 3))'''),
('md', r'''The same argument gives the partition coefficient of a slit of width $H$ wider than the largest span:
a conformation fits if $X_c<H$, and the accessible heights of its lowest point have total length $H-X_c$,
so $K=\langle H-X_c\rangle/H=1-X/H$ (here $H\ge L$ suffices, since $X_c\le L$).'''),
('md', r'''## The depletion layer at a wall

By the identity of Wang, Hansen, Peters and Hassager (J. Chem. Phys. 129, 074904, 2008) the depletion thickness of
any reference point at a hard wall is $\delta=X/2$. For a Gaussian chain $\delta=2R_g/\sqrt\pi$. With the long-chain
expansions of $F$ and $g$, the wall offset $2R_g/\sqrt\pi-\delta$ tends to $z_Mp$:
$$\frac{2R_g}{\sqrt\pi}-\delta=z_Mp-\Big(\frac B2+\sqrt{\frac3\pi}\Big)\frac{p}{\sqrt\lambda}+\dots,$$
so the approach is slow. The cell computes the offset from the spectral table (Fig. 3d).'''),
('code', r'''g = pmf.rg2_over_p2(LAM)
offset = 2*np.sqrt(g)/math.sqrt(math.pi) - F/2
for x in (10, 100, 1000, 1e4, 1e5):
    o = float(offset[LAM==x][0])
    print(f'lambda={x:>7g}   offset/p = {o:.4f}   asymptotic form = {Z_M-(B_COEF/2+math.sqrt(3/math.pi))/math.sqrt(x):.4f}')
print('limit z_M =', Z_M)
print('short chains: (L/4 - delta)/(L^2/p) at 0.01 =', (0.01/4-float(F[LAM==0.01][0])/2)/1e-4, ' (37/1536 =', 37/1536, ')')'''),
])

# =============================================================== 04 short chains
CH['04_short_chains'] = nb('Chapter 4: Short chains and the coefficient 37/768', [
('md', r'''## Why the first correction is of order $\lambda^2$

For a chain much shorter than $p$, a tangent that starts at an angle bounded away from $\pi/2$ keeps its
sign along $\hat{\mathbf e}$ over the whole contour, so the projected contour is monotone and its span
equals its projected path length. The deficit $\lambda/2-F$ comes from **grazing** orientations,
$U_0=O(\sqrt\lambda)$. Writing $U_{\lambda t}=\sqrt\lambda\,\xi_t$ for $0\le t\le1$, the tangent SDE becomes
$d\xi=dW+O(\lambda)$, and $Z_{\lambda t}-Z_0=\lambda^{3/2}\int_0^t\xi_s\,ds$: the tangent is Brownian and the
position is **integrated Brownian motion** (the randomly accelerated particle). The angular measure of
the grazing sector is $O(\sqrt\lambda)$, the range lost per grazing path is $O(\lambda^{3/2})$, and their
product is the $\lambda^2$ correction.

## The return kernel and the integral $J=37/192$

For velocity generator $\partial_v^2$, the first-return density of integrated Brownian motion from
outward speed $v_0$ to inward speed $v$ at time $t$ is (McKean 1963; Burkhardt 2014)
$$\mathcal K(v,v_0;t)=\frac{\sqrt3}{2\pi}\frac v{t^2}\exp\Big[-\frac{v^2-vv_0+v_0^2}t\Big]\operatorname{erf}\Big(\sqrt{\tfrac{3vv_0}t}\Big).$$
By scale invariance the speed-weighted return integral $J=\int_0^\infty v_0\,dv_0\int_0^\infty\mathcal K(v,v_0;1)\,dv$
does not depend on $t$. We evaluate it three ways.'''),
('code', r'''from scipy import integrate
K = lambda v, v0, t=1.0: math.sqrt(3)/(2*math.pi)*v/t**2*math.exp(-(v*v-v*v0+v0*v0)/t)*math.erf(math.sqrt(3*v*v0/t))
J_num, err = integrate.dblquad(lambda v, v0: v0*K(v, v0), 0, 12, 0, 12)
print('direct 2D quadrature       J =', J_num, '+/-', err)
# substitution v = eta v0 and the a-integral in closed form gives a 1D integral
eta = sp.symbols('eta', positive=True)
J_eta = sp.Rational(9,8)/sp.pi*sp.integrate(eta**sp.Rational(3,2)*(eta**2+eta+1)/((eta**2-eta+1)**2*(eta+1)**3), (eta, 0, sp.oo))
print('symbolic eta-integral      J =', sp.nsimplify(sp.simplify(J_eta)))
r = sp.symbols('r', real=True)
J_r = sp.Rational(9,16)/sp.pi*sp.integrate((3+4*r**2)/((1+r**2)**2*(1+4*r**2)**2), (r, 0, sp.oo))
print('rational r-integral        J =', sp.simplify(J_r), '=', float(J_r))
print('37/192 =', 37/192)'''),
('md', r'''### The reduction of $J$ to a rational integral, step by step

1. **Scale invariance.** Under $v\to\sqrt t\,v$, $v_0\to\sqrt t\,v_0$ the kernel scales as
   $\mathcal K(v,v_0;t)=t^{-3/2}\mathcal K(v/\sqrt t,v_0/\sqrt t;1)$, and the measure $v_0\,dv_0\,dv$ as $t^{3/2}$,
   so $J$ is the same at every $t$. The speed-weighted probability of a return within unit time is
   therefore $\int_0^1J\,dt=J$.
2. **Polar-type substitution.** Put $v=\eta v_0$ at $t=1$:
   $J=\frac{\sqrt3}{2\pi}\int_0^\infty\eta\,d\eta\int_0^\infty v_0^3\,e^{-\alpha v_0^2}\operatorname{erf}(\beta v_0)\,dv_0$,
   with $\alpha=\eta^2-\eta+1$ and $\beta=\sqrt{3\eta}$.
3. **The inner integral** is elementary:
   $\int_0^\infty a^3e^{-\alpha a^2}\operatorname{erf}(\beta a)\,da=\dfrac{\beta}{2\alpha^2\sqrt{\alpha+\beta^2}}+\dfrac{\beta}{4\alpha(\alpha+\beta^2)^{3/2}}$
   (minus the derivative with respect to $\alpha$ of $\int_0^\infty a\,e^{-\alpha a^2}\operatorname{erf}(\beta a)\,da=\beta/[2\alpha\sqrt{\alpha+\beta^2}]$).
4. **The square.** $\alpha+\beta^2=\eta^2+2\eta+1=(\eta+1)^2$, so the square roots disappear and the bracket
   becomes $\sqrt{3\eta}\,\dfrac{2(\eta+1)^2+\alpha}{4\alpha^2(\eta+1)^3}=\sqrt{3\eta}\,\dfrac{3(\eta^2+\eta+1)}{4\alpha^2(\eta+1)^3}$,
   which gives the first form of $J$ quoted in the article, with the prefactor $\frac{\sqrt3}{2\pi}\cdot\frac{3\sqrt3}{4}=\frac9{8\pi}$.
5. **Rational form.** With $r=(\sqrt\eta-1/\sqrt\eta)/2$, i.e. $\sqrt\eta=r+\sqrt{1+r^2}$, the integrand is
   rational and even in $r$, and $0<\eta<\infty$ maps to $-\infty<r<\infty$; partial fractions give $37/192$.

The cell checks steps 1, 3, 4 and 5 separately.'''),
('code', r'''import mpmath as mp
# step 1: J does not depend on t
for tt in (0.5, 1.0, 2.0):
    Jt = integrate.dblquad(lambda v, v0: v0*K(v, v0, tt), 0, 12*math.sqrt(tt), 0, 12*math.sqrt(tt))[0]
    print(f'step 1:  J(t={tt}) = {Jt:.10f}')
# step 3: the inner integral, at arbitrary alpha and beta
for al, be in ((1.7, 0.9), (0.75, 1.2247), (3.0, 2.4)):
    lhs = mp.quad(lambda a_: a_**3*mp.exp(-al*a_**2)*mp.erf(be*a_), [0, mp.inf])
    rhs = be/(2*al**2*mp.sqrt(al+be**2)) + be/(4*al*(al+be**2)**1.5)
    print(f'step 3:  alpha={al}, beta={be}:  quadrature {float(lhs):.12f}   closed form {float(rhs):.12f}')
# step 4: the algebra, symbolically
eta = sp.symbols('eta', positive=True)
al, be = eta**2 - eta + 1, sp.sqrt(3*eta)
print('step 4:  alpha + beta^2 - (eta+1)^2 =', sp.expand(al + be**2 - (eta+1)**2))
bracket = be/(2*al**2*(eta+1)) + be/(4*al*(eta+1)**3)
integrand = sp.sqrt(3)/(2*sp.pi)*eta*bracket
target = sp.Rational(9, 8)/sp.pi*eta**sp.Rational(3, 2)*(eta**2+eta+1)/(al**2*(eta+1)**3)
print('         integrand - first form of J =', sp.simplify(integrand - target))
# step 5: the substitution sqrt(eta) = r + sqrt(1+r^2), including the Jacobian
r = sp.symbols('r', real=True)
eta_r = (r + sp.sqrt(1+r**2))**2
lhs = target.subs(eta, eta_r)*sp.diff(eta_r, r)
rhs = sp.Rational(9, 16)/sp.pi*(3+4*r**2)/((1+r**2)**2*(1+4*r**2)**2)
ratios = [round(float((lhs/rhs).subs(r, x)), 12) for x in (-2, -0.3, 0, 0.7, 5)]
print('step 5:  [eta-integrand x d eta/dr] / [r-integrand] at r = -2, -0.3, 0, 0.7, 5:', ratios)
assert all(abs(x - 0.5) < 1e-10 for x in ratios)
print('         the ratio is 1/2 everywhere: eta from 0 to infinity maps to r from -infinity to infinity, and the r-integrand is even,')
print('         so the integral over the whole r-axis of half of it equals the integral from 0 to infinity in the second form of J')
print('         partial fractions:', sp.apart((3+4*r**2)/((1+r**2)**2*(1+4*r**2)**2), r))'''),
('md', r'''## From $J$ to the WLC coefficient

The WLC grazing layer has velocity generator $\tfrac12\partial_\xi^2$, not $\partial_v^2$; rescaling speeds by
$1/\sqrt2$ maps one to the other and multiplies the speed-weighted return by $\tfrac12$:
$\int_0^\infty\xi\,Q_{\rm ret}(\xi)\,d\xi=J/2=37/384$. Inserting the scaling form
$\int_0^1v\,q(0^+,-v;\lambda)\,dv=\lambda\int_0^\infty\xi Q_{\rm ret}(\xi)\,d\xi+o(\lambda)$ into the flux identity of
Chapter 2, $F'=\tfrac12-\tfrac{37}{384}\lambda+o(\lambda)$, and integrating,
$$F(\lambda)=\frac\lambda2-\frac{37}{768}\lambda^2+o(\lambda^2).$$
**Status: derived.** The scaling limit is taken under the integral by dominated convergence: a return
from outward speed $v$ requires the tangent to reach zero, and the exponential martingale inequality
gives $\Pr(T_v\le t)\le\exp[-(v-t)^2/(2t)]$ for $v>t$, hence the $t$-independent integrable envelope
$\xi Q_t(\xi)\le\xi\mathbf 1_{\xi\le2}+\xi e^{-\xi^2/8}\mathbf 1_{\xi>2}$. No rate for the remainder follows.

## Numerical evidence (Fig. 3a)

The refined spectral solutions at angular orders up to 512 with first-alias-corrected inversion
(`data/stiff_alias_summary.csv`) give the deficit $(\lambda/2-F)/\lambda^2$:'''),
('code', r'''stiff = pd.read_csv(ROOT/'data'/'stiff_alias_summary.csv')
stiff['deficit'] = (stiff['lambda']/2 - stiff.F_extrap)/stiff['lambda']**2
print(stiff[['lambda', 'F_extrap', 'extrapolation_sensitivity', 'deficit']].to_string(index=False))
slope, intercept = np.polyfit(stiff['lambda'], stiff['deficit'], 1)
print(f'\nlinear fit: intercept {intercept:.6f} (37/768 = {37/768:.6f}),  slope {slope:.5f})')
check('intercept vs 37/768', intercept, 37/768, 2e-5)'''),
('md', r'''### The limit function $Q_{\rm ret}$ and the dominated convergence step (supplementary material)

Equation (10) needs the limit $\lambda\to0$ under the integral over starting speeds. Two ingredients:

* **The limit function.** For the WLC layer (generator $\tfrac12\partial_\xi^2$), $Q_{\rm ret}(\xi)$ is the
  probability of a return within unit time from outward speed $\xi$. In McKean's units (generator
  $\partial_v^2$) the speed is $v_0=\sqrt2\,\xi$, and by the scaling of step 1,
  $Q_{\rm ret}(\xi)=\int_0^1dt\int_0^\infty\mathcal K(v,\sqrt2\xi;t)\,dv=2\int_{\sqrt2\xi}^\infty\frac{G(a)}a\,da$,
  with $G(a)=\int_0^\infty\mathcal K(w,a;1)\,dw$. Since $\xi\,d\xi=\tfrac12v_0\,dv_0$, this gives
  $\int_0^\infty\xi\,Q_{\rm ret}\,d\xi=J/2=37/384$.
* **The envelope.** From the tangent equation, $U_s=v-\int_0^sU_r\,dr+N_s\ge v-s+N_s$, where $N$ is a martingale with
  quadratic variation at most $s$. A return within $t$ requires $U$ to reach zero, hence $N_s\le-(v-t)$
  for some $s\le t$, and the exponential martingale inequality gives $\Pr(T_v\le t)\le e^{-(v-t)^2/(2t)}$
  for $v>t$. With $v=\sqrt t\,\xi$ and $t\le1$: $(v-t)^2/(2t)=(\xi-\sqrt t)^2/2\ge\xi^2/8$ for $\xi>2$. So
  $\xi\,Q_t(\xi)\le\xi\mathbf 1_{\{\xi\le2\}}+\xi e^{-\xi^2/8}\mathbf 1_{\{\xi>2\}}$ for every $t\le1$, which is integrable.

The cell evaluates $Q_{\rm ret}$, its speed-weighted integral, and the envelope. As a consistency check,
$Q_{\rm ret}(0^+)$ computed over *all* times must be 1 (a randomly accelerated particle returns with certainty);
in the present variables that is $2\int_0^\infty G(a)\,da/a=1$.'''),
('code', r'''Kw = lambda w, a_: math.sqrt(3)/(2*math.pi)*w*math.exp(-(w*w-w*a_+a_*a_))*math.erf(math.sqrt(3*w*a_))
G = lambda a_: integrate.quad(lambda w: Kw(w, a_), 0, np.inf)[0]
Qmck = lambda v0: 2*integrate.quad(lambda a_: G(a_)/a_, v0, np.inf, limit=200)[0]      # return within unit time, McKean units
total = 4*integrate.quad(lambda x_: G(x_*x_)/x_, 0, np.inf, limit=200)[0]                  # a = x^2 removes the square-root singularity at a = 0
check('total return probability, 2 int_0^inf G(a)/a da', total, 1.0, 1e-6)
print('\n   xi     Q_ret(xi)      envelope')
for xi in (0.25, 0.5, 1, 2, 3, 4):
    print(f'{xi:5g}   {Qmck(math.sqrt(2)*xi):.4e}     {1.0 if xi <= 2 else math.exp(-xi*xi/8):.4e}')
val = integrate.quad(lambda xi: xi*Qmck(math.sqrt(2)*xi), 0, 8, limit=100)[0]
check('int xi Q_ret d xi = 37/384', val, 37/384, 1e-8)'''),
('md', r'''## The endpoint excess $9/256$

Define $Y=1-\rho^2/\lambda^2\in[0,1]$. A Burkholder-Davis-Gundy estimate gives $\mathbb EY^3=O(\lambda^3)$, so
$\mathbb E\rho/\lambda=1-\tfrac12\mathbb EY-\tfrac18\mathbb EY^2+O(\lambda^3)$ with $\mathbb EY=1-m_2/\lambda^2$ and
$\mathbb EY^2=1-2m_2/\lambda^2+m_4/\lambda^4$:'''),
('code', r'''lam = sp.symbols('lambda', positive=True)
m2 = 2*(lam-1+sp.exp(-lam))
m4 = sp.Rational(4,27)*(45*lam**2-156*lam+214-54*(lam+4)*sp.exp(-lam)+2*sp.exp(-3*lam))
EY = 1 - m2/lam**2; EY2 = 1 - 2*m2/lam**2 + m4/lam**4
Erho_over_lam = sp.series(1 - EY/2 - EY2/8, lam, 0, 3).removeO()
print('E rho / lambda =', Erho_over_lam)
EZ = sp.series(Erho_over_lam*lam/2, lam, 0, 3).removeO()          # E|Z_lambda| = E rho / 2
excess = sp.expand((lam/2 - sp.Rational(37,768)*lam**2) - EZ)
print('F - E|R.e|/p   =', excess, '   (9/256 =', sp.Rational(9,256), ')')'''),
('code', r'''ex = pd.read_csv(ROOT/'data'/'endpoint_excursion_diagnostic.csv')
ex = ex[ex['lambda'] <= 0.1].sort_values(['lambda','steps']).groupby('lambda').tail(1)
print(ex[['lambda','steps','n','excursion_over_lambda2','excursion_se']].to_string(index=False))
print('\nall within 2 SE of 9/256:', bool(np.all(np.abs(ex.excursion_over_lambda2 - 9/256) <= 2*ex.excursion_se)))'''),
('md', r'''## Generalization and dimension independence

The same argument applies to any stationary, sign-symmetric velocity diffusion with stationary density
$\varpi$ and local variance rate $a(0)$ at zero velocity:
$\mathbb E[\text{range}]=\mu t-\tfrac{37}{384}\varpi(0)a(0)t^2+o(t^2)$, $\mu=\mathbb E|U|$. For isotropic
rotational diffusion in $D$ dimensions (first tangent harmonic decaying as $e^{-t}$) one has
$\varpi_D(u)=c_D(1-u^2)^{(D-3)/2}$ and $a_D(0)=2/(D-1)$, and the identity $\varpi_D(0)a_D(0)=\mu_D$ makes the
*relative* coefficient dimension independent: $F_D=\mu_D\lambda[1-\tfrac{37}{384}\lambda+o(\lambda)]$.'''),
('code', r'''D = sp.symbols('D', positive=True)
cD = sp.gamma(D/2)/(sp.sqrt(sp.pi)*sp.gamma((D-1)/2))
muD = sp.gamma(D/2)/(sp.sqrt(sp.pi)*sp.gamma((D+1)/2))
diff = sp.gammasimp(sp.combsimp((cD*2/(D-1))/muD))
print('ratio varpi_D(0) a_D(0) / mu_D =', diff, '  (symbolically, using Gamma((D+1)/2) = ((D-1)/2) Gamma((D-1)/2))')
print('numerically for D=2..6:', [float((cD*2/(D-1)-muD).subs(D, k)) for k in range(2, 7)])
print('mu_2 =', sp.simplify(muD.subs(D, 2)), '  mu_3 =', sp.simplify(muD.subs(D, 3)))'''),
])

# =============================================================== 05 long chains
CH['05_long_chains'] = nb('Chapter 5: Long chains, the Milne length and the coefficient B', [
('md', r'''## The leading Gaussian law (derived)

$\langle U_\tau U_0\rangle=e^{-\tau}/3$, so the projected position becomes Brownian with diffusivity
$D=\int_0^\infty\langle U_\tau U_0\rangle d\tau=\tfrac13$, i.e. $\langle Z_\lambda^2\rangle\to2\lambda/3$. The mean range of
Brownian motion with variance $\sigma^2\lambda$ over $[0,\lambda]$ is $\sqrt{8\sigma^2\lambda/\pi}$, hence
$$F(\lambda)\sim\sqrt{16\lambda/(3\pi)}=\frac4{\sqrt{3\pi}}\sqrt\lambda .$$
The supplementary material proves this by a martingale functional central limit theorem, with the martingale clock
$Q_\lambda=\int_0^\lambda(1-U_s^2)ds$ of mean $2\lambda/3$. Its variance, computed below, shows that $Q_\lambda/\lambda$
concentrates at $2/3$:'''),
('code', r'''tau, s_, r_ = sp.symbols('tau s r', positive=True)
cov = sp.Rational(4,45)*sp.exp(-3*sp.Abs(s_-r_))          # Cov(U_s^2, U_r^2) from the P_2 harmonic (eigenvalue 3)
VarQ = sp.simplify(2*sp.integrate(sp.integrate(cov.subs(sp.Abs(s_-r_), s_-r_), (r_, 0, s_)), (s_, 0, tau)))
print('Var Q_tau =', sp.factor(VarQ))
print('E Q_tau   = 2 tau/3   (from <1-U^2> = 2/3)')'''),
('md', r'''## The kinetic boundary layer and the Milne problem (conditional)

Fig. 3(c) shows that the actual deficit $4\sqrt{\lambda/3\pi}-F$ tends to a constant. Within a fraction of a
persistence length of a new maximum the tangents are not equilibrated: the contour approached the record
moving outward. The bulk diffusive solution therefore matches the kinetic boundary at an effective plane
displaced by an extrapolation length. Near the target the stationary problem is
$$u\,\partial_y\Psi=\mathcal L_u\Psi,\qquad\Psi(0,u>0)=0,\qquad\Psi\sim y-u+z_M\ (y\to\infty),$$
the angular-diffusion Schwarzschild-Milne problem of Amic, Luck and Nieuwenhuizen (1996), with
$z_M=0.71821164$. Both extremes contribute, so $F=4\sqrt{\lambda/3\pi}-2z_M+\dots$

We solve the Milne problem live at increasing angular order with `code/build_boundary_moments.py`,
which eliminates the two stationary modes analytically and matches the decaying modes on the incoming
Gauss nodes.'''),
('code', r'''from build_boundary_moments import stationary_milne, moment_record
rows = []
for m in (16, 32, 64, 128, 256):
    rec = moment_record(stationary_milne(m)); rows.append(rec)
tab = pd.DataFrame(rows)[['m','z_M','M1','M2','M3','B_from_moments','B_lower_bound','B_upper_bound']]
print(tab.to_string(index=False))
print('\nz_M converges as m^-3; the published value is 0.71821164.')
check('z_M at m=256', float(tab.z_M.iloc[-1]), Z_M, 1e-6)'''),
('md', r'''## From the boundary layer to the offset, one step at a time

**Step 1: the transform of $F$.** With $H_s=s\widehat q$, the Laplace transform of the kinetic equation is
$(s+u\partial_y-\mathcal L_u)H_s=0$ with $H_s(0,u>0)=1$, and since $F=\int_0^\infty\!\int_{-1}^1q\,du\,dy=2\int_0^\infty\langle q\rangle_u\,dy$,
$$\widehat F(s)=\frac2s\int_0^\infty\langle H_s\rangle_u\,dy .$$

**Step 2: the bulk mode.** Far from the target, $H_s=A\,\psi_\kappa(u)\,e^{-\kappa y}$ with
$(\mathcal L_u+\kappa u)\psi_\kappa=s\psi_\kappa$, $\langle\psi_\kappa\rangle_u=1$. Perturbation theory in $\kappa$ (carried out
symbolically at the end of this chapter) gives $\psi_\kappa=1+\kappa u+\kappa^2(u^2-\tfrac13)/3+\dots$ and
$s=\kappa^2/3-11\kappa^4/135+\dots$, i.e. $\kappa=\sqrt{3s}\,[1+\tfrac{11}{30}s+\dots]$.

**Step 3: the inner expansion.** At fixed $y$, write $H_s=1-\kappa\Psi+\kappa^2\Phi+\dots$ Since $s=\kappa^2/3+\dots$,
the orders $\kappa$ and $\kappa^2$ of the equation are, with $\mathcal T=u\partial_y-\mathcal L_u$,
$$\mathcal T\Psi=0,\quad\Psi(0,u>0)=0;\qquad\qquad\mathcal T\Phi=-\tfrac13,\quad\Phi(0,u>0)=0 .$$

**Step 4: matching.** Expand the bulk mode with amplitude $A=1-z_M\kappa+a_2\kappa^2+\dots$ at fixed $y$:
$$A\,\psi_\kappa e^{-\kappa y}=1-\kappa\,(y-u+z_M)+\kappa^2\big[\tfrac12y^2-yu+z_M(y-u)+a_2+\tfrac13(u^2-\tfrac13)\big]+\dots$$
The order-$\kappa$ term is the far field of $\Psi$: $\Psi\sim y-u+z_M$, which defines the Milne problem and
its extrapolation length $z_M$. The order-$\kappa^2$ term is the far field of $\Phi$, which
contains the constant $a_2$ needed for $B$.

**Step 5: integrate.** The bulk mode contributes $\int_0^\infty A\langle\psi_\kappa\rangle_ue^{-\kappa y}dy=A/\kappa$, and
the boundary-layer part of $\Psi$ contributes $-\kappa I$ with $I=\int_0^\infty\langle\Psi-y+u-z_M\rangle_u\,dy$:
$$\int_0^\infty\langle H_s\rangle_u\,dy=\frac1\kappa-z_M+(a_2-I)\,\kappa+\dots$$

**Step 6: invert.** Inserting $\kappa(s)$,
$\widehat F=\dfrac2{\sqrt3\,s^{3/2}}-\dfrac{2z_M}s+\dfrac{\sqrt\pi B}{\sqrt s}+\dots$ with
$\sqrt\pi B=2\sqrt3\,(a_2-I)-\tfrac{11}{15\sqrt3}$, and the transforms
$s^{-3/2}\leftrightarrow2\sqrt{\lambda/\pi}$, $s^{-1}\leftrightarrow1$, $s^{-1/2}\leftrightarrow1/\sqrt{\pi\lambda}$ give the long-chain expansion:
$F=\dfrac4{\sqrt{3\pi}}\sqrt\lambda-2z_M+\dfrac B{\sqrt\lambda}+\dots$

The cell carries out steps 3-6 symbolically.'''),
('code', r'''u, y, kap, zM, a2, I_, s = sp.symbols('u y kappa z_M a_2 I s')
Lu = lambda f: sp.Rational(1, 2)*sp.diff((1-u**2)*sp.diff(f, u), u)
Top = lambda f: u*sp.diff(f, y) - Lu(f)                                   # the transport operator T
avg = lambda f: sp.integrate(f, (u, -1, 1))/2
# step 4: expansion of the bulk mode at fixed y
psi = 1 + kap*u + kap**2*(u**2 - sp.Rational(1, 3))/3
mode = sp.series((1 - zM*kap + a2*kap**2)*psi*sp.exp(-kap*y), kap, 0, 3).removeO()
Psi_far = -sp.expand(mode.coeff(kap, 1)); Phi_far = sp.expand(mode.coeff(kap, 2))
print('step 4:  far field of Psi =', Psi_far)
print('         far field of Phi =', Phi_far)
# step 3: the far fields solve the first- and second-order equations
print('step 3:  T Psi_far =', sp.simplify(Top(Psi_far)), '     T Phi_far =', sp.simplify(Top(Phi_far)))
# steps 5-6: integrate, insert kappa(s), read off the coefficients
kappa_s = sp.sqrt(3*s)*(1 + sp.Rational(11, 30)*s)
integral = 1/kap - zM + (a2 - I_)*kap
Fhat = sp.expand(sp.series((2/s*integral).subs(kap, kappa_s), s, 0, 0).removeO())
print('step 6:  F_hat(s) =', Fhat)
coefB = sp.simplify(Fhat.coeff(1/sp.sqrt(s)))
print('         coefficient of s^(-1/2) =', coefB, '   minus [2 sqrt(3)(a_2 - I) - 11/(15 sqrt(3))] =',
      sp.simplify(coefB - (2*sp.sqrt(3)*(a2 - I_) - sp.Rational(11, 15)/sp.sqrt(3))))
lam = sp.symbols('lambda', positive=True); sv = sp.symbols('s', positive=True)
for f in (sp.sqrt(lam), sp.Integer(1), 1/sp.sqrt(lam)):
    print(f'         Laplace transform of {str(f):15s} =', sp.laplace_transform(f, lam, sv, noconds=True))'''),
('md', r'''## The moment representation of $B$

Define $h(v)=\Psi(0,-v)$ and $M_n=\int_0^1v^nh\,dv$. Three moment identities of the Milne equation,
$\partial_y\langle u\Psi\rangle=0$, $\partial_y\langle u^2\Psi\rangle=-\langle u\Psi\rangle$,
$\partial_y\langle u^3\Psi\rangle=\langle\Psi\rangle-3\langle u^2\Psi\rangle$, give with the far field
$M_1=\tfrac23$, $M_2=\tfrac23z_M$, and $I=\int_0^\infty\langle\Psi-y+u-z_M\rangle dy=M_3/2-\tfrac15$. The table above
confirms $M_1$ and $M_2$ to machine precision. Second-order matching by reciprocity then gives
(article and supplementary material)
$$\sqrt\pi\,B=\sqrt3\Big(2z_M^2-2M_3+\tfrac{17}{45}\Big),\qquad
B=\sqrt{\tfrac3\pi}\Big(\tfrac23z_M^2+\tfrac{17}{45}-\tfrac43\operatorname{Var}_M(v)\Big),$$
where $p_M(v)=\tfrac32v\,h(v)$ is the normalized boundary-speed distribution with mean $z_M$. Because
$0\le v\le1$, $\operatorname{Var}_M\le z_M(1-z_M)$ and $B$ is bounded on both sides.'''),
('code', r'''rec = rows[-1]
B_formula = math.sqrt(3/math.pi)*(2*rec['z_M']**2 - 2*rec['M3'] + 17/45)
print('B from (2 z^2 - 2 M3 + 17/45)      :', B_formula)
print('B from moment record (variance form):', rec['B_from_moments'])
print('analytic bounds                     :', rec['B_lower_bound'], '<= B <=', rec['B_upper_bound'])
print('small-s transfer fits (article)     : 0.6468 +/- 0.0004')
check('B from moments', rec['B_from_moments'], 0.6468, 1e-4)'''),
('md', r'''## The constant $a_2$ by reciprocity, one step at a time (supplementary material)

**Moment identities.** Multiply $u\,\partial_y\Psi=\mathcal L_u\Psi$ by $1$, $u$ and $u^2$ and average over $u$.
$\mathcal L_u$ is self-adjoint, and $\mathcal L_u1=0$, $\mathcal L_uu=-u$, $\mathcal L_uu^2=1-3u^2$, so
$$\partial_y\langle u\Psi\rangle=0,\qquad\partial_y\langle u^2\Psi\rangle=-\langle u\Psi\rangle,\qquad
\partial_y\langle u^3\Psi\rangle=\langle\Psi\rangle-3\langle u^2\Psi\rangle .$$
At the target $\Psi=0$ for $u>0$, so $\langle u^n\Psi\rangle(0)=\tfrac12\int_{-1}^0u^n\Psi(0,u)\,du=(-1)^nM_n/2$.
In the far field, $\langle u(y-u+z_M)\rangle=-\tfrac13$, $\langle u^2(y-u+z_M)\rangle=(y+z_M)/3$ and
$\langle u^3(y-u+z_M)\rangle=-\tfrac15$. Therefore:

* first identity: $-M_1/2=-\tfrac13$, so $M_1=\tfrac23$;
* second identity: $\langle u^2\Psi\rangle=M_2/2+y/3$ exactly; comparison with the far field gives $M_2=\tfrac23z_M$;
* third identity: its right-hand side is $\langle\Psi\rangle-y-z_M=\langle\Psi-y+u-z_M\rangle$, so integrating from $0$ to
  $\infty$ gives $I=-\tfrac15-(-M_3/2)=M_3/2-\tfrac15$.

**Reciprocity.** $G(y,u)=\Psi(y,-u)$ solves the adjoint problem $-u\partial_yG=\mathcal L_uG$, with $G(0,u<0)=0$ and
$G\sim y+u+z_M$. Using $u\partial_y\Phi=\mathcal L_u\Phi-\tfrac13$ and the self-adjointness of $\mathcal L_u$,
$$\partial_y\langle u\,\Phi\,G\rangle=\langle(\mathcal L_u\Phi-\tfrac13)G\rangle-\langle\Phi\,\mathcal L_uG\rangle=-\tfrac13\langle\Psi\rangle .$$
At $y=0$ the product $\Phi G$ vanishes for every $u$ ($\Phi=0$ for $u>0$, $G=0$ for $u<0$). Integrating to $Y$:
$$\langle u\,\Phi\,G\rangle(Y)=-\tfrac13\int_0^Y\langle\Psi\rangle\,dy=-\tfrac16Y^2-\tfrac13z_MY-\tfrac13I+o(1).$$
The left-hand side can also be evaluated from the far fields of step 4. Equating the constants gives
$a_2=z_M^2-\tfrac4{45}-I$: the second-order problem never has to be solved.

**Assembly.** $a_2-I=z_M^2-\tfrac4{45}-2I=z_M^2-M_3+\tfrac{14}{45}$, so
$\sqrt\pi B=2\sqrt3(a_2-I)-\tfrac{11}{15\sqrt3}=\sqrt3\,(2z_M^2-2M_3+\tfrac{17}{45})$. With the normalized speed
distribution $p_M(v)=\tfrac32vh(v)$ (norm $\tfrac32M_1=1$, mean $\tfrac32M_2=z_M$, second moment $\tfrac32M_3$),
$M_3=\tfrac23[\operatorname{Var}_M(v)+z_M^2]$ and the moment representation of B follows.

The first cell does the algebra symbolically. The second solves the second-order boundary problem
*directly* (which the argument avoids) and compares.'''),
('code', r'''u, y, Y, zM, a2, I_, M3, VarM = sp.symbols('u y Y z_M a_2 I M_3 Var_M')
Lu = lambda f: sp.Rational(1, 2)*sp.diff((1-u**2)*sp.diff(f, u), u)
avg = lambda f: sp.integrate(f, (u, -1, 1))/2
print('L 1 =', Lu(sp.Integer(1)), '   L u =', sp.expand(Lu(u)), '   L u^2 =', sp.expand(Lu(u**2)))
far = y - u + zM
print('far-field moments: <u Psi> =', avg(u*far), '   <u^2 Psi> =', sp.factor(avg(u**2*far)), '   <u^3 Psi> =', avg(u**3*far))
# reciprocity: far-field value of <u Phi G> at y = Y
Phi_far = Y**2/2 - Y*u + zM*(Y - u) + a2 + (u**2 - sp.Rational(1, 3))/3
G_far = Y + u + zM
lhs = sp.expand(avg(u*Phi_far*G_far))
rhs = -Y**2/6 - zM*Y/3 - I_/3
print('<u Phi G> from the far fields =', lhs)
a2_sol = sp.solve(sp.expand(lhs - rhs), a2)[0]
print('a_2 =', sp.expand(a2_sol), '      (Y-dependent terms cancel:', sp.expand(lhs - rhs).has(Y) is False, ')')
# assembly
I_M3 = M3/2 - sp.Rational(1, 5)
sqrtpiB = sp.expand((2*sp.sqrt(3)*(a2 - I_) - sp.Rational(11, 15)/sp.sqrt(3)).subs(a2, a2_sol).subs(I_, I_M3))
print('sqrt(pi) B =', sp.simplify(sqrtpiB/sp.sqrt(3)), 'x sqrt(3)')
B_var = sp.sqrt(3/sp.pi)*(sp.Rational(2, 3)*zM**2 + sp.Rational(17, 45) - sp.Rational(4, 3)*VarM)
print('the moment representation of B minus this, with M_3 = (2/3)(Var_M + z_M^2):',
      sp.simplify(B_var - (sqrtpiB/sp.sqrt(sp.pi)).subs(M3, sp.Rational(2, 3)*(VarM + zM**2))))
lower = sp.simplify(B_var.subs(VarM, zM*(1 - zM))/sp.sqrt(3/sp.pi) - (2*(zM - sp.Rational(1, 3))**2 + sp.Rational(7, 45)))
print('lower bound, Var_M <= z_M(1 - z_M): difference from 2(z_M - 1/3)^2 + 7/45 =', lower)'''),
('code', r'''from check_second_order_matching import solve as second_order
chk = pd.DataFrame([second_order(m) for m in (32, 64, 128)])
print(chk[['m', 'z_M', 'M1_minus_2/3', 'M2_minus_2zM/3', 'I_minus_(M3/2-1/5)']].to_string(index=False))
print()
print(chk[['m', 'a2_direct', 'a2_reciprocity', 'a2_difference', 'B_moment_formula', 'B_direct']].to_string(index=False))
check('a_2: direct solution minus reciprocity (largest)', float(chk.a2_difference.abs().max()), 0.0, 1e-10)'''),
('md', r'''In the discretized problem the three moment identities and the reciprocity value of $a_2$ hold to
rounding error at every order, because the Gauss quadrature integrates the polynomials involved exactly.
What converges with the order $m$ is $z_M$ and $M_3$, and with them $B$.

**Why $\Psi\ge0$ (supplementary material).** $\mathcal T\Psi=0$ says that $\Psi$ is harmonic for the contour
that moves toward the target plane. Evaluated along such a contour, $\Psi$ is a martingale until the contour
reaches the plane (where $\Psi=0$, since it arrives with $u\ge0$) or a large distance $R$ (where
$\Psi\simeq R$). Optional stopping gives $\Psi(y,u)=\lim_{R\to\infty}R\,\Pr_{(y,u)}\{\text{distance }R\text{ is reached before the plane}\}\ge0$,
the analog of the harmonic function $y$ for Brownian motion killed at zero. In particular $h(v)\ge0$, so
$p_M$ is a probability density and the bounds on $B$ hold.'''),
('md', r'''## The bulk dispersion relation

The slow bulk mode $\psi_\kappa e^{-\kappa y}$ with $(\mathcal L_u+\kappa u)\psi_\kappa=s\psi_\kappa$ is an ordinary
perturbation problem; the $\kappa^4$ coefficient $s_4=-11/135$ enters the constant $17/45$ above.'''),
('code', r'''u, kap = sp.symbols('u kappa')
L = lambda f: sp.Rational(1,2)*sp.diff((1-u**2)*sp.diff(f, u), u)
avg = lambda f: sp.integrate(f, (u, -1, 1))/2
phi1 = u
s2 = avg(u*phi1)
# solve L phi2 = s2 - u phi1 with zero mean, polynomial ansatz
a2, b2 = sp.symbols('a2 b2')
phi2 = a2*u**2 + b2
sol = sp.solve([sp.Poly(sp.expand(L(phi2) - (s2 - u*phi1)), u).coeffs()[0], avg(phi2)], [a2, b2])
phi2 = phi2.subs(sol)
a3, b3 = sp.symbols('a3 b3'); phi3 = a3*u**3 + b3*u
eq = sp.Poly(sp.expand(L(phi3) - (s2*phi1 - u*phi2)), u)
phi3 = phi3.subs(sp.solve(eq.coeffs(), [a3, b3]))
s4 = sp.simplify(avg(u*phi3))
print('phi2 =', phi2, '   phi3 =', phi3)
print('s2 =', s2, '   s4 =', s4, '   (article: 1/3 and -11/135)')
print('<u^2 phi2> =', sp.simplify(avg(u**2*phi2)), '  (4/135)')'''),
('md', r'''## The telegraph process: the same structure, exactly

A particle with speed $c$ whose velocity correlation decays as $c^2e^{-at}$ (reversals at rate $a/2$) has
diffusivity $D=c^2/a$, $\ell_{\rm f}=c/a$ (speed times velocity correlation time), and the exactly solvable mean range
$$F_T(t)=\frac{2c}a\big[g_0(x)+at\{g_0(x)+g_1(x)\}-1\big],\quad x=\frac{at}2,\quad g_\nu(x)=e^{-x}I_\nu(x).$$
Its large-$t$ expansion has the Gaussian range minus twice the one-dimensional extrapolation length $\ell_{\rm f}$:'''),
('code', r'''from scipy.special import ive
c, a = 0.5, 0.75
FT = lambda t: (2*c/a)*(ive(0, a*t/2) + a*t*(ive(0, a*t/2) + ive(1, a*t/2)) - 1)
D, ell = c*c/a, c/a
for t in (1e2, 1e3, 1e4, 1e5):
    resid = FT(t) - (4*math.sqrt(D*t/math.pi) - 2*ell)
    print(f't={t:>8g}  F_T - [4 sqrt(Dt/pi) - 2 l] = {resid:.6f}   x sqrt(t) = {resid*math.sqrt(t):.5f}   (l/sqrt(pi a) = {ell/math.sqrt(math.pi*a):.5f})')
# symbolic: large-x expansion of e^{-x} I_nu(x) = (2 pi x)^{-1/2} [1 + (1-4nu^2)/(8x) + ...]
x = sp.symbols('x', positive=True); cs, as_ = sp.symbols('c a', positive=True)
g0 = (2*sp.pi*x)**sp.Rational(-1,2)*(1 + sp.Rational(1,8)/x + sp.Rational(9,128)/x**2)
g1 = (2*sp.pi*x)**sp.Rational(-1,2)*(1 - sp.Rational(3,8)/x - sp.Rational(15,128)/x**2)
expr = (2*cs/as_)*(g0 + 2*x*(g0+g1) - 1)
y = sp.symbols('y', positive=True)            # y = 1/x
ser = sp.series(expr.subs(x, 1/y), y, 0, 2).removeO()
print('\nF_T ~', sp.expand(ser), '   with y = 2/(a t)')
print('   i.e. (2c/a)[ sqrt(8x/pi) - 1 + 1/(2 sqrt(2 pi x)) + ... ] = 4 sqrt(D t/pi) - 2 l + l/sqrt(pi a t) + ...')
print('short-time: F_T = c t - a c t^2/8 + ..., check:', (c*0.01 - FT(0.01))/0.01**2, 'vs', a*c/8)'''),
('md', r'''**Conclusion of the chapter.** The WLC offset $-2z_Mp$ is conditional on the matching assumption; the
telegraph case shows that the structure "Brownian range minus twice the Milne extrapolation length"
holds exactly in a solvable persistent walk, and the spectral spans of Chapter 6 agree with the
three-term expansion to $4\times10^{-8}$ relative for $\lambda\ge5000$. The planar run-and-tumble particle obeys the same
rule exactly (discussed in the article). Neither is a proof of the matching.'''),
])

# =============================================================== 06 spectral solver
CH['06_spectral_solver'] = nb('Chapter 6: The spectral first-return solver, step by step', [
('md', r'''## The problem to solve

$q(y,u;\lambda)$ obeys $\partial_\lambda q+u\,\partial_yq=\mathcal L_uq$ with $q(0^+,u>0)=1$, $q(\cdot,\cdot;0)=0$ and
$q\to0$ as $y\to\infty$; $F=\int_0^\infty\!\int_{-1}^1q\,du\,dy$. We build the solver of `code/wlc_pn_transfer.py`
one step at a time and then run it.

### Step 1: Laplace transform in $\lambda$

$u\,\partial_y\widehat q=\mathcal L_u\widehat q-s\widehat q$ with $\widehat q(0^+,u>0)=1/s$. The time variable is gone.

### Step 2: angular discretization

In the orthonormal Legendre basis $\phi_\ell$, $\mathcal L_u$ is diagonal ($-\ell(\ell+1)/2$) and $u\cdot$ is
tridiagonal. Evaluating at the $m$ Gauss-Legendre nodes gives $\partial_y\mathbf g=A_m(s)\mathbf g$,
$A_m=U^{-1}(R_m-sI)$, $R_m=-V^T\Lambda_mV$.'''),
('code', r'''from wlc_pn_transfer import angular_system, laplace_transform, invert_euler_bromwich, BromwichSettings, telegraph_p1_transform
sysm = angular_system(8)
print('nodes u_k  :', np.round(sysm.nodes, 4))
print('Lambda diag:', np.diag(sysm.Lambda))
J = sysm.V @ np.diag(sysm.nodes) @ sysm.V.T
print('streaming matrix J (moment space) is tridiagonal:', np.allclose(J - np.diag(np.diag(J)) - np.diag(np.diag(J,1),1) - np.diag(np.diag(J,-1),-1), 0))
print('(J)_{l,l+1} =', np.round(np.diag(J,1), 5), ' vs (l+1)/sqrt((2l+1)(2l+3)) =', np.round([(l+1)/math.sqrt((2*l+1)*(2*l+3)) for l in range(7)], 5))'''),
('md', r'''### Step 3: the half-space boundary-value problem

The decaying solutions span the stable invariant subspace of $A_m(s)$. Its dimension must equal the
number of incoming nodes, $m/2$, so that the incoming values determine the solution. The code checks
this at every transform point. The outgoing values at $y=0$ (the return-time distribution) are then an
*output*. Spatial integration is analytic: $\int_0^\infty\mathbf g\,dy=-A_m^{-1}\mathbf g(0)$.

The production solver (`build_reference_curve.METHOD = "schur"`) obtains an orthonormal basis of the
stable subspace from an ordered complex Schur factorization; `data/conditioning_audit_full.csv` records
the condition number of $E_+Z_-$ in that basis at orders 8 to 128 and at the 40 transform points of the
primary inversion for $\lambda=0.01$ to $10^5$ (maximum 1.45). A plain eigendecomposition
(`method="eig"`) spans the same subspace with a non-orthogonal basis; it reproduces the Schur transform
to $10^{-13}$ and the reference values to $1.3\times10^{-10}$ and is kept as a regression check
(`code/solver_regression_audit.py`).'''),
('code', r'''s = 0.7 + 0.3j
for m in (8, 16, 32):
    A = (angular_system(m).R.astype(complex) - s*np.eye(m)) / angular_system(m).nodes[:, None]
    ev = np.linalg.eigvals(A)
    print(f'm={m:3d}: eigenvalues with Re<0: {int((ev.real<0).sum())}  (incoming nodes: {m//2})')
val, diag = laplace_transform(s, 32, diagnostics=True)
print('\nF_hat_32(s) =', val, '   stable dim', diag.stable_dimension, '  cond(E+ Z-) =', round(diag.incoming_condition_number, 4))
val_eig = laplace_transform(s, 32, method='eig')
print('eigendecomposition route (regression check): |difference| =', abs(val - val_eig))
cond = pd.read_csv(ROOT/'data'/'conditioning_audit_full.csv')
print('conditioning audit: orders', sorted(cond.m.unique()), ' max cond(E+ Z-) =', cond.condition_max.max() if 'condition_max' in cond else cond.iloc[:, -1].max())'''),
('md', r'''### Step 3 in formulas

1. **Nodal variables.** Write $\widehat q(y,u)=\sum_{\ell<m}a_\ell(y)\phi_\ell(u)$ and $g_k=\sqrt{w_k}\,\widehat q(y,u_k)$,
   i.e. $\mathbf g=V^T\mathbf a$ with $V_{\ell k}=\sqrt{w_k}\phi_\ell(u_k)$. $V$ is orthogonal because the $m$-point Gauss
   rule integrates products $\phi_\ell\phi_{\ell'}$ ($\ell,\ell'<m$) exactly.
2. **The system.** $u\,\partial_y\widehat q=(\mathcal L_u-s)\widehat q$ becomes $D_u\partial_y\mathbf g=(R_m-sI)\mathbf g$ with
   $R_m=-V^T\Lambda_mV$, hence $\partial_y\mathbf g=A_m\mathbf g$, $A_m=D_u^{-1}(R_m-sI)$.
3. **Decay.** A solution that vanishes as $y\to\infty$ is $\mathbf g(y)=Z_-e^{y\,T_-}\mathbf c$, where the columns of
   $Z_-$ span the stable invariant subspace of $A_m$ and $T_-$ is $A_m$ restricted to it.
4. **Boundary condition.** On the incoming nodes $u_k>0$, $\widehat q=1/s$, i.e. $g_k=\sqrt{w_k}/s$:
   $E_+Z_-\mathbf c=\mathbf h_+/s$. This is a square system when the stable subspace has dimension $m/2$.
5. **Integration.** $\int_0^\infty\mathbf g\,dy=-A_m^{-1}\mathbf g(0)$ for a decaying solution, and
   $\int_{-1}^1\widehat q\,du=\sum_kw_k\widehat q(u_k)=\mathbf h^T\mathbf g$ with $h_k=\sqrt{w_k}$. Therefore
   $\widehat F_m(s)=\int_0^\infty\!\int_{-1}^1\widehat q\,du\,dy=-\tfrac1s\,\mathbf h^TA_m^{-1}Z_-(E_+Z_-)^{-1}\mathbf h_+$.

The cell repeats these five steps with plain eigenvectors at $m=8$, integrates each decaying mode
separately, and compares with the production routine.'''),
('code', r'''m_, s_ = 8, 0.7 + 0.3j
sy = angular_system(m_); un = sy.nodes
wk = np.polynomial.legendre.leggauss(m_)[1]
print('V is orthogonal:', np.allclose(sy.V @ sy.V.T, np.eye(m_)))
A = (sy.R.astype(complex) - s_*np.eye(m_))/un[:, None]
mu, vec = np.linalg.eig(A)
dec = mu.real < 0                                                   # decaying modes
c_ = np.linalg.solve(vec[np.ix_(un > 0, dec)], np.sqrt(wk[un > 0])/s_)   # boundary condition on incoming nodes
F_modes = sum(c_[j]*(np.sqrt(wk) @ vec[:, dec][:, j])/(-mu[dec][j]) for j in range(dec.sum()))   # mode-by-mode integral
print('mode-by-mode integral :', F_modes)
print('production routine    :', laplace_transform(s_, m_))
print('outgoing trace s*q(0,-v) at the outgoing nodes (an output of the solve, between 0 and 1 for real s):')
s_r = 0.7; A_r = (sy.R - s_r*np.eye(m_))/un[:, None]; mu_r, vec_r = np.linalg.eig(A_r); dr = mu_r.real < 0
g0 = vec_r[:, dr] @ np.linalg.solve(vec_r[np.ix_(un > 0, dr)], np.sqrt(wk[un > 0])/s_r)
print('   v =', np.round(-un[un < 0], 3), '\n   s*q =', np.round((s_r*g0/np.sqrt(wk)).real[un < 0], 4))'''),
('md', r'''### Step 4: inversion and continuum extrapolation

The Euler-accelerated Fourier series (Abate and Whitt 1992) inverts $\widehat F_m$ with $N+K+1$ evaluations
at $s_j=(A_E+2\pi ij)/(2\lambda)$. The ideal series equals $\sum_{j\ge0}e^{-jA_E}F((2j+1)\lambda)$, so the
combination $F_A(\lambda)-e^{-A_E}F_A(3\lambda)$ removes the first alias (`invert_alias_corrected`). This allows
the small parameter $A_E=12$ (with $N=K=32$), which matters because the prefactor $e^{A_E/2}$ multiplies
floating-point noise in $\widehat F_m$: with the uncorrected $A_E=22$ of an earlier version of the table, that
noise grew from a few $10^{-8}$ relative at $\lambda=10^3$ to nearly $10^{-6}$ for $\lambda\ge5\times10^4$
(`validation/inversion_noise_audit.csv`). Each angular
order gives $F_m(\lambda)$; the orders 32, 48, 64, 96 are extrapolated with $F_\infty+c_3m^{-3}+c_4m^{-4}$, and a
fifth order, $m=128$, enters only the sensitivity. The algorithm box of the supplementary material lists the whole procedure. We run
it live at $\lambda=1$ and compare with the reference table.'''),
('code', r'''from build_reference_curve import evaluate_node, continuum_fit
t0 = time.time(); row = evaluate_node(1.0); dt = time.time() - t0
ref1 = REF[REF['lambda'] == 1.0].iloc[0]
print(f'{dt:.1f} s for one node on one core')
for k in ('F_m32','F_m48','F_m64','F_m96','F_continuum','epsilon_internal_rel'):
    print(f'{k:22s} live {row[k]:.10g}    archived {ref1[k]:.10g}')
check('live F_continuum(1) vs archive', row['F_continuum'], ref1['F_continuum'], 1e-9)'''),
('md', r'''### Where the alias series comes from, and the bound on what remains

The Fourier-series method replaces the Bromwich integral by a trapezoidal sum with step $\pi/\lambda$ in
$\Im s$. By the Poisson summation formula, a trapezoidal sum of a Fourier integral returns the function
plus its periodic copies, here damped by the contour shift $A_E/(2\lambda)$:
$$F_A(\lambda)=\sum_{j\ge0}e^{-jA_E}F\big((2j+1)\lambda\big)=F(\lambda)+e^{-A_E}F(3\lambda)+e^{-2A_E}F(5\lambda)+\dots$$
Apply the same series at $3\lambda$ and subtract $e^{-A_E}$ times it:
$$F_A(\lambda)-e^{-A_E}F_A(3\lambda)=F(\lambda)-e^{-2A_E}\big[F(9\lambda)-F(5\lambda)\big]+O(e^{-3A_E}).$$
The remaining alias is negative because $F$ increases. Subadditivity (Chapter 2) bounds it:
$F(9\lambda)\le F(5\lambda)+F(4\lambda)\le F(5\lambda)+4F(\lambda)$, so its magnitude is below $4e^{-2A_E}F(\lambda)$,
which is $1.5\times10^{-10}F$ at $A_E=12$.

The cell shows both statements on the two-mode ($m=2$) problem, whose exact inverse is the telegraph
range $F_T$ of Chapter 5: the uncorrected series is off by $e^{-A_E}F_T(3\lambda)$, and the corrected one by the
predicted remainder.'''),
('code', r'''from scipy.special import ive
from wlc_pn_transfer import invert_alias_corrected
c_t = 1/math.sqrt(3)
FT2 = lambda t: 2*c_t*(ive(0, t/2) + t*(ive(0, t/2) + ive(1, t/2)) - 1)       # telegraph range with a = 1
st = BromwichSettings(12.0, 32, 32); AE = st.A
print(' lambda   uncorrected error    e^-A F(3 lam) + e^-2A F(5 lam)   corrected error     -e^-2A [F(9 lam) - F(5 lam)]    bound 4 e^-2A F')
for lam_ in (0.1, 1.0, 10.0, 1e3):
    raw = invert_euler_bromwich(lam_, 2, st) - FT2(lam_)
    cor = invert_alias_corrected(lam_, 2, st) - FT2(lam_)
    print(f'{lam_:7g}   {raw:+.6e}        {math.exp(-AE)*FT2(3*lam_) + math.exp(-2*AE)*FT2(5*lam_):+.6e}              '
          f'{cor:+.3e}          {-math.exp(-2*AE)*(FT2(9*lam_) - FT2(5*lam_)):+.3e}                 {4*math.exp(-2*AE)*FT2(lam_):.3e}')'''),
('md', r'''### Convergence in angular order

The finite-order error falls as $m^{-3}$ (the slope of the log-log plot). The article attributes this to the
corner singularity of $q$ at the grazing point $(y,u)=(0,0)$, where the prescribed and free halves of the
boundary meet.'''),
('code', r'''ms = np.array([32, 48, 64, 96]); vals = np.array([row[f'F_m{m}'] for m in ms])
err = np.abs(vals/row['F_continuum'] - 1)
slope = np.polyfit(np.log(ms), np.log(err), 1)[0]
print('relative errors:', err, '\nlog-log slope  :', round(slope, 2), ' (m^-3 expected)')
# withheld orders 128, 192, 256 from the archived spot check (code/high_order_spot_checks.py):
# an m^-3 + m^-4 fit to these three orders alone is an extrapolation independent of the one in the table
spot = pd.read_csv(ROOT/'data'/'high_order_spot_checks.csv')
cols = ['lambda', 'relative_difference_m128_pct', 'relative_difference_extrapolation_pct',
        'scaled_residual_m128', 'scaled_residual_m192', 'scaled_residual_m256']
print(spot[cols].to_string(index=False))
print('\n(F_m - F_inf) m^3 is constant within 2.5 % for 0.1 <= lambda <= 100: the m^-3 law holds there.')
print('At lambda = 1000 the high-order differences are at the noise level (about 1e-9 relative).')
print('At lambda = 0.01 the m^-3 regime is not reached at m <= 256, which is why the short-chain diagnostics go to m = 512.')
ver = pd.read_csv(ROOT/'validation'/'inversion_noise_audit.csv')
print('\nDeviation from the long-chain expansion for lambda >= 5000, former A=22 table vs current table:')
print(ver[ver['lambda'] >= 5000][['lambda', 'dev_longchain_schur_A22', 'dev_longchain_current', 'eps_rel_current']].to_string(index=False))'''),
('md', r'''### An exactly solvable test of the inversion

At $m=2$ the transfer problem is a telegraph process with speed $1/\sqrt3$ and decay rate 1, whose transform
is known in closed form (`telegraph_p1_transform`). Comparing the numerical transform with it isolates
the linear-algebra part; inverting it with the production series tests the Fourier inversion against the
Bessel-function range. The remaining alias of the corrected series is $-e^{-2A_E}[F(9\lambda)-F(5\lambda)]$,
negative and below $4e^{-2A_E}F$ by subadditivity; `code/telegraph_inversion_test.py` checks this at all 67
lengths (`validation/telegraph_inversion_test.csv`).'''),
('code', r'''for s in (0.5, 1+2j, 5-0.5j):
    print(f's={s!s:12}  numerical {laplace_transform(s, 2):.12g}   analytic {telegraph_p1_transform(s):.12g}')
from scipy.special import ive
c, a = 1/math.sqrt(3), 1.0
FT = lambda t: (2*c/a)*(ive(0, a*t/2) + a*t*(ive(0, a*t/2) + ive(1, a*t/2)) - 1)
from wlc_pn_transfer import invert_alias_corrected
from build_reference_curve import PRIMARY
for lam in (0.1, 10.0, 1e3, 1e5):
    num = invert_alias_corrected(lam, 2, PRIMARY)
    pred = -math.exp(-2*PRIMARY.A)*(FT(9*lam) - FT(5*lam))
    print(f'lambda={lam:8g}  relative error {num/FT(lam)-1:+.3e}   predicted remaining alias {pred/FT(lam):+.3e}')
tel = pd.read_csv(ROOT/'validation'/'telegraph_inversion_test.csv')
print('\nall 67 lengths, analytic transform: errors from %.2e to %.2e; max |error - predicted| = %.1e' % (
      tel.rel_error_analytic_transform.min(), tel.rel_error_analytic_transform.max(),
      (tel.rel_error_analytic_transform - tel.predicted_rel_residual_alias).abs().max()))'''),
('md', r'''### Cost and what limits the accuracy

One node costs seconds, so the limit is accuracy, not cost, and it has one cause: the full-range Legendre
basis resolves the grazing-point singularity only algebraically. Two improvements are possible but
are not implemented here: a half-range (double-$P_N$) basis, or subtraction of the known local
random-acceleration solution at the corner, after which the remainder should converge exponentially. The
stationary Milne problem of Chapter 5, with its known $z_M$, is the natural test bed.'''),
])

# =============================================================== 07 checks
CH['07_independent_checks'] = nb('Chapter 7: Independent checks', [
('md', r'''## Why two more calculations

The spectral solver discretizes the kinetic equation in one particular way. A second discretization of
the same equation (finite volumes) checks the solver; sampling chain conformations checks the kinetic
equation itself, since the chains never see the kinetic equation.

## Finite-volume solution

`code/wlc_finite_volume.py` uses a cell-centered grid in $(y,u)$, conservative angular fluxes, upwind
transport, the incoming ghost value 1 at $y=0$, and the exact compact support $q=0$ for $y>\lambda$; the
semidiscrete system is propagated in $\lambda$ by a sparse matrix exponential. We run a small grid live
to show the mechanism, then load the archived extrapolations.'''),
('code', r'''from wlc_finite_volume import FVSettings, solve_fv
ref1 = float(REF.loc[REF['lambda']==1.0, 'F_continuum'].iloc[0])
for Ny, Nu in ((100, 16), (200, 24), (400, 32)):
    t0 = time.time(); v = solve_fv(1.0, FVSettings(Ny=Ny, Nu=Nu, Ymax=1.0))
    print(f'Ny={Ny:4d} Nu={Nu:3d}  F={v:.6f}  rel. diff from spectral {100*(v/ref1-1):+.3f} %   ({time.time()-t0:.1f} s)')
fv = pd.read_csv(ROOT/'data'/'finite_volume_comparison.csv')
print('\narchived extrapolations:'); print(fv[['lambda','F_spectral','F_continuum_quad','relative_difference_pct','FV_combined_sensitivity_rel_pct']].to_string(index=False))'''),
('md', r'''The coarse grids converge toward the spectral value from below; the archived extrapolations agree
within 0.018 % when the grids are extrapolated with the integer convergence orders of the scheme; at
$\lambda=0.1$ and 10 that exceeds the finite-volume sensitivity by factors of 2.0 and 1.5, and all three
residuals are positive. The extended grid (`code/build_finite_volume_extended_audit.py`, $N_y=1600$, $N_u=64$)
shows why: the observed spatial order (`code/finite_volume_order_analysis.py`) is 1.3-2.0 at $\lambda=0.1$ but
only 0.77-0.94 at $\lambda=1$ and 10, so the integer-power model leaves a positive bias. A free-order
(Aitken) extrapolation removes it: the residuals become +0.004, -0.003 and -0.003 %. Methods independent of
the spectral solver therefore corroborate the curve to about $4\times10^{-5}$ (finite volume) and
$4\times10^{-4}$-$9\times10^{-4}$ (path sampling, one standard error, including the long-chain runs at
$L/p=10^3$ and $10^4$ in `data/heat_path_long.csv` and `data/heat_path_long_pooled.csv`). At $L/p=100$ a separable
finite-volume extrapolation (`code/build_finite_volume_long.py`) agrees with the spectral value to $3\times10^{-5}$.'''),
('code', r'''ext = pd.read_csv(ROOT/'data'/'finite_volume_extended_comparison.csv')
print(ext.to_string(index=False))
fvo = pd.read_csv(ROOT/'data'/'finite_volume_order_analysis.csv')
print(); print(fvo.to_string(index=False))'''),
('md', r'''## Heat-kernel path sampling

The archived campaign (`code/sphere_heat_path_validation.py`) used 40 000 chains at each of
$\lambda=0.1,1,10,100$ and four contour resolutions sharing the same tangent paths, so step-size effects
are paired differences. Chapter 1 ran a small version of the same sampler.'''),
('code', r'''paths = pd.read_csv(ROOT/'data'/'heat_path_coupled.csv')
refmap = REF.set_index('lambda').F_continuum
fine = paths.sort_values('steps').groupby('lam').tail(1).copy()
fine['F_spec'] = fine.lam.map(refmap); fine['dev (%)'] = 100*(fine.span/fine.F_spec-1); fine['2SE (%)'] = 200*fine.span_se/fine.F_spec
print(fine[['lam','steps','n','span','span_se','F_spec','dev (%)','2SE (%)']].to_string(index=False))
print('\nall finest-grid estimates within 2 SE of the spectral values:', bool(np.all(np.abs(fine['dev (%)']) <= fine['2SE (%)'])))'''),
('code', r'''paths_long = pd.concat([pd.read_csv(ROOT/'data'/'heat_path_long.csv').query('lam < 5e3'),
                        pd.read_csv(ROOT/'data'/'heat_path_long_pooled.csv')])      # 1e4: four pooled batches
fl = paths_long.sort_values('steps').groupby('lam').tail(1).copy()
fl['F_spec'] = fl.lam.map(refmap); fl['dev/SE'] = (fl.span-fl.F_spec)/fl.span_se
print(fl[['lam','steps','n','span','span_se','F_spec','dev/SE']].to_string(index=False))'''),
('md', r'''Fig. S1 of the supplementary material (`pmf.figure_checks()`) plots these deviations, together with the
finite-volume results and the paired step-size differences. Sampling errors exclude heat-kernel truncation,
the inverse-CDF grid and between-node interpolation.'''),
('code', r'''import plot_main_figures as pmf
fig = pmf.figure_checks(); fig'''),
])

# =============================================================== 08 formula
CH['08_practical_formula'] = nb('Chapter 8: The practical formula', [
('md', r'''## Construction

With $w=\sqrt\lambda$,
$$F_R=\frac{w^2}2\,\frac{1+a_Rw+b_Rw^2+c_Rw^3}{1+a_Rw+d_Rw^2+e_Rw^3+f_Rw^4}.$$
Six coefficients, four constraints: the rod limit is built in; $d_R-b_R=37/384$ enforces the coefficient
$37/768$; and the large-$w$ expansion must reproduce $4/\sqrt{3\pi}$, $-2z_M$ and $B$. The remaining two
parameters were fitted by minimax to 34 calibration nodes (`code/fit_correlation.py`); the other 33 nodes,
including the nine above $\lambda=5000$, were withheld.'''),
('code', r'''coef = json.load(open(ROOT/'data'/'rational_fit_coefficients.json'))['central']; print(coef)
w = sp.symbols('w', positive=True)
a_, b_, c_, d_, e_, f_ = [sp.nsimplify(round(coef[k], 8), rational=True) for k in 'abcdef']
FR = w**2/2*(1+a_*w+b_*w**2+c_*w**3)/(1+a_*w+d_*w**2+e_*w**3+f_*w**4)
print('short:', sp.N(sp.series(FR, w, 0, 5).removeO(), 8), '   (37/768 =', float(sp.Rational(37,768)), ')')
print('d_R - b_R =', float(d_-b_), '   37/384 =', 37/384)
v = sp.symbols('v', positive=True)
longexp = sp.series(FR.subs(w, 1/v), v, 0, 2).removeO()
print('long :', sp.N(longexp, 8), '   expected 4/sqrt(3 pi) =', round(A_GAUSS, 8), ', -2 z_M =', -2*Z_M, ', B =', B_COEF)'''),
('md', r'''## Sign audit

Treating the printed coefficients as exact rationals, the numerators of $F_R'$, $-F_R''$,
$-d(F_R/\lambda)/d\lambda$ and $\lambda/2-F_R$ are polynomials in $w$ whose signs can be certified (`code/rational_symbolic_audit.py`).
The formula is therefore increasing, concave, has nonincreasing $F_R/\lambda$ and obeys $F_R<\lambda/2$ for
all $\lambda>0$, like the exact $F$.'''),
('code', r'''from rational_symbolic_audit import audit_variant
rows, summary = audit_variant(coef, 8)
pd.DataFrame(summary)'''),
('md', '## Error at every node (Fig. 4)'),
('code', r'''import plot_main_figures as pmf
err = 100*(pmf.rational(LAM)/F - 1)
split = pd.read_csv(ROOT/'data'/'reference_nodes_with_split.csv').set_index('lambda')['set']
sets = pd.Series(LAM).map(split).fillna('holdout (added)')
for name, idx in pd.Series(LAM).groupby(sets.values).groups.items():
    print(f'{name:16s} n={len(idx):2d}  max |error| = {np.abs(err[list(idx)]).max():.4f} %')
check('max error on all 67 nodes (%)', np.abs(err).max(), 0.0924, 5e-4)
fig = pmf.figure4(); fig'''),
])

# =============================================================== 09 verification
CH['09_verification_summary'] = nb('Chapter 9: The quoted numbers, recomputed', [
('md', r'''This chapter recomputes the quantitative statements of the article from the archived data and prints
a pass/fail line for each. Two kinds of test are used: `rec` checks a quoted value to the rounding of
the printed number; `upto` checks a stated bound one-sidedly ("below", "within", "at most"). The list
covers the numerical solution, the results and the numerical methods of the supplementary material.'''),
('code', r'''import plot_main_figures as pmf
from scipy.optimize import brentq
results = []
def rec(name, value, expected, tol):
    results.append(check(name, value, expected, tol))
def upto(name, value, bound):
    ok = value <= bound
    print(('PASS' if ok else 'FAIL') + f'  {name}: {value:.4g}  (bound {bound:g})')
    results.append(ok)
Fi = lambda x: float(F[np.isclose(LAM, x)][0])
coil = 4/math.sqrt(math.pi); Rg = np.sqrt(pmf.rg2_over_p2(LAM)); ratio = F/Rg
short = pmf.short_expansion(LAM); longc = pmf.long_expansion(LAM)
e_short = np.abs(short/F-1); e_long = np.abs(longc/F-1)
# Numerical solution: the table and its accuracy
rec('number of nodes', len(LAM), 67, 0)
rec('max internal relative sensitivity (1.9e-5)', REF.epsilon_internal_rel.max(), 1.86e-5, 5e-7)
rec('max internal absolute sensitivity (1.5e-5, at 7.5e4)', REF.epsilon_internal_abs.max(), 1.5e-5, 5e-7)
upto('all eps_int components below 4.2e-8 for lambda>=5000', REF.epsilon_internal_rel[LAM>=5000].max(), 4.2e-8)
upto('order dependence |F96-Finf|/F below 6e-8 for lambda>1e3', (np.abs(REF.F_m96-REF.F_continuum)/REF.F_continuum)[LAM>1e3].max(), 6e-8)
spot = pd.read_csv(ROOT/'data'/'high_order_spot_checks.csv').set_index('lambda')
for lam, q in ((0.1, 5e-9), (1.0, 3e-9), (10.0, 9e-10), (100.0, 4e-11), (1000.0, 2e-9)):
    rec(f'high-order extrapolation vs table at {lam:g} ({q:g})', abs(spot.loc[lam, 'relative_difference_extrapolation_pct'])/100, q, 0.5*q)
rec('high-order extrapolation vs table at 0.01 (1.4e-6)', spot.loc[0.01, 'relative_difference_extrapolation_pct']/100, 1.4e-6, 5e-8)
sc = spot[['scaled_residual_m128', 'scaled_residual_m192', 'scaled_residual_m256']]
upto('(F_m-F_inf) m^3 constant within 2.5 % for 0.1<=lambda<=100', float(((sc.max(axis=1)-sc.min(axis=1))/sc.max(axis=1)).loc[[0.1, 1.0, 10.0, 100.0]].max()), 0.025)
stiff = pd.read_csv(ROOT/'data'/'stiff_alias_summary.csv').set_index('lambda')
for lam, q in ((0.0125, 7.8e-7), (0.025, 9e-8)):
    rec(f'table vs m<=512 diagnostics at {lam:g} ({q:g})', (stiff.loc[lam, 'F_extrap']-Fi(lam))/stiff.loc[lam, 'F_extrap'], q, 0.05*q)
upto('table vs m<=512 diagnostics, lambda>=0.05 (< 1e-8)', max(abs(Fi(l)/stiff.loc[l, 'F_extrap']-1) for l in stiff.index if l >= 0.05), 1e-8)
cond = pd.read_csv(ROOT/'data'/'conditioning_audit_full.csv')
upto('cond(E+ Z-) at every production point (below 1.47)', cond.condition_max.max(), 1.47)
rec('stable dimension = m/2 everywhere', float((cond.stable_dimension_min == cond.m/2).all() and (cond.stable_dimension_max == cond.m/2).all()), 1, 0)
reg = pd.read_csv(ROOT/'data'/'solver_regression_audit.csv')
upto('eig vs Schur transform (< 6e-12)', reg.rel_eig_schur.max(), 6e-12)
ver = pd.read_csv(ROOT/'validation'/'reference_table_versions.csv')
upto('Schur vs eig table, former A=22 inversion (<= 1.3e-10)', ver.rel_schur_A22_minus_eig_A22.abs().max(), 1.35e-10)
# The crossover
rec('crossing point 64/(3 pi)', 64/(3*math.pi), 6.79, 0.005)
rec('F at crossing (2.20)', float(np.interp(64/(3*math.pi), LAM, F)), 2.20, 0.005)
rec('overestimate at crossing (54 %)', 100*((64/(3*math.pi))/2/np.interp(64/(3*math.pi), LAM, F)-1), 54, 0.5)
rec('short-chain error at lambda=0.3 (0.10 %)', 100*e_short[LAM==0.3][0], 0.10, 0.005)
upto('short-chain error within 0.3 % for lambda<=0.5', 100*e_short[LAM<=0.5].max(), 0.3)
upto('long-chain error within 0.3 % for lambda>=5', 100*e_long[LAM>=5].max(), 0.3)
rec('long-chain error at 10 (0.07 %)', 100*e_long[LAM==10][0], 0.07, 0.005)
rec('long-chain error at 100 (0.0006 %)', 100*e_long[LAM==100][0], 0.0006, 0.00005)
upto('long-chain error within 4e-8 for lambda>=5000', e_long[LAM>=5000].max(), 4e-8)
rec('Gaussian law overestimate at 100 (12 %)', 100*(pmf.A_GAUSS*10/Fi(100)-1), 12, 0.5)
rec('Gaussian law overestimate at 5000 (1.6 %)', 100*(pmf.A_GAUSS*math.sqrt(5000)/Fi(5000)-1), 1.6, 0.05)
# Span and radius of gyration
rec('X/Rg at 5000 (2.2225)', float(ratio[LAM==5000][0]), 2.2225, 5e-5)
rec('X/Rg at 1e5 (2.2489)', float(ratio[LAM==1e5][0]), 2.2489, 5e-5)
rec('below coil at 5000 (1.5 %)', 100*(1-float(ratio[LAM==5000][0])/coil), 1.5, 0.05)
rec('below coil at 1e5 (0.35 %)', 100*(1-float(ratio[LAM==1e5][0])/coil), 0.35, 0.005)
c1 = math.sqrt(3)*pmf.B_COEF + 6/math.sqrt(math.pi)
ratio_long = lambda l: coil - 2*math.sqrt(3)*Z_M/math.sqrt(l) + c1/l
rec('1 % approach from the long-chain expansion of X/R_g (1.2e4)', brentq(lambda l: 1 - ratio_long(l)/coil - 0.01, 1e3, 1e6), 1.2e4, 500)
# Independent tests, Fig. S1
paths = pd.read_csv(ROOT/'data'/'heat_path_coupled.csv'); fine = paths.sort_values('steps').groupby('lam').tail(1).set_index('lam')
upto('max paired step difference, L/p<=100 (< 0.03 %)', 100*np.abs(paths.pairwise_minus_fine/paths.lam.map(REF.set_index('lambda').F_continuum)).max(), 0.03)
p01 = paths[paths.lam == 0.1]
upto('paired step differences at L/p=0.1 (< 0.002 %)', 100*np.abs(p01.pairwise_minus_fine).max()/float(REF.loc[REF['lambda']==0.1,'F_continuum'].iloc[0]), 0.002)
fv = pd.read_csv(ROOT/'data'/'finite_volume_comparison.csv').set_index('lambda')
for lam, q in ((0.1, 0.011), (1.0, 0.005), (10.0, 0.018)):
    rec(f'finite volume, integer orders, at {lam:g} ({q} %)', fv.loc[lam, 'relative_difference_pct'], q, 0.0006)
rec('finite volume residual / sensitivity at 0.1 (2.0)', fv.loc[0.1, 'relative_difference_pct']/fv.loc[0.1, 'FV_combined_sensitivity_rel_pct'], 2.0, 0.05)
rec('finite volume residual / sensitivity at 10 (1.5)', fv.loc[10.0, 'relative_difference_pct']/fv.loc[10.0, 'FV_combined_sensitivity_rel_pct'], 1.5, 0.05)
fvo = pd.read_csv(ROOT/'data'/'finite_volume_order_analysis.csv').set_index('lambda')
upto('finite volume, observed orders, |residual| (< 4e-5)', fvo.relative_difference_pct.abs().max()/100, 4e-5)
rec('observed order at lambda=1, smallest (0.77)', fvo.loc[1.0, 'observed_order_Ny_min'], 0.77, 0.005)
rec('observed order at lambda=10, largest (0.94)', fvo.loc[10.0, 'observed_order_Ny_max'], 0.94, 0.006)
# The practical formula
err = 100*np.abs(pmf.rational(LAM)/F-1)
split = pd.read_csv(ROOT/'data'/'reference_nodes_with_split.csv').set_index('lambda')['set']; sets = pd.Series(LAM).map(split)
rec('calibration / holdout / extrapolation counts (34/24/9)', 10000*(sets=='calibration').sum()+100*(sets=='holdout').sum()+(sets=='extrapolation').sum(), 342409, 0)
rec('fit error on calibration nodes (0.089 %)', err[(sets=='calibration').values].max(), 0.0887, 0.0005)
rec('fit error on withheld nodes (0.092 %)', err[(sets!='calibration').values].max(), 0.0924, 0.0005)
upto('fit error on nodes > 5000 (< 1.5e-6)', err[LAM>5000].max()/100, 1.5e-6)
# Milne length and B from archived moments
bm = json.load(open(ROOT/'data'/'boundary_moment_summary.json'))
rec('B from boundary moments at m=512 (0.6468)', bm['highest_order']['B_from_moments'], 0.6468, 2e-5)
rec('z_M at m=512 (0.71821164)', bm['highest_order']['z_M'], 0.71821164, 2e-8)
mm = pd.read_csv(ROOT/'data'/'milne_boundary_moments.csv').set_index('m')
for m, b in ((32, 0.646722), (64, 0.646791), (128, 0.646799), (192, 0.646800)):
    rec(f'B(m={m}) ({b})', mm.loc[m, 'B_from_moments'], b, 5e-7)
# Second-round additions: every remaining revised number
ex = pd.read_csv(ROOT/'data'/'endpoint_excursion_diagnostic.csv'); ex = ex[ex['lambda'] <= 0.1]
fin = ex.sort_values('steps').groupby('lambda').tail(1).set_index('lambda')
dev = 100*(fin.excursion_over_lambda2/(9/256)-1); z = (fin.excursion_over_lambda2-9/256)/fin.excursion_se
for lam, q in ((0.005, -0.60), (0.01, 0.10), (0.02, -0.81), (0.05, -0.65), (0.1, -1.28)):
    rec(f'Fig. 3(b) deviation from 9/256 at {lam:g} ({q} %)', dev.loc[lam], q, 0.006)
upto('Fig. 3(b): all points within two standard errors', float(np.abs(z).max()), 2.0)
rec('Gaussian value 4Rg/sqrt(pi) above X at 100 (10 %)', 100*(coil*float(Rg[LAM == 100][0])/Fi(100)-1), 10, 0.5)
rec('noise amplification ratio e^(22/2)/e^(12/2) (e^5 ~ 150)', math.exp(5), 148.4, 0.1)
rec('formula error at lambda=6000, printed coefficients (1.46e-6)', err[LAM == 6000][0]/100, 1.46e-6, 0.01e-6)
ratios = {l: REF.epsilon_internal_rel[np.isclose(LAM, l)].values[0]/abs(spot.loc[l, 'relative_difference_extrapolation_pct']/100) for l in (0.01, 1000.0)}
rec('eps_int / accuracy at 0.01 (14)', ratios[0.01], 14, 0.5)
rec('eps_int / accuracy at 1000 (35)', ratios[1000.0], 35, 1)
nz = pd.read_csv(ROOT/'validation'/'inversion_noise_audit.csv').set_index('lambda')
upto('former A=22 table vs the long-chain expansion, lambda>=5000 (< 1e-6)', nz.dev_longchain_schur_A22[nz.index >= 5000].abs().max(), 1e-6)
upto('current table vs the long-chain expansion, lambda>=5000 (< 4e-8)', nz.dev_longchain_current[nz.index >= 5000].abs().max(), 4e-8)
# Third-round additions
rec('former A=22 table vs the long-chain expansion, lambda>=5000: largest (9e-7)', nz.dev_longchain_schur_A22[nz.index >= 5000].abs().max(), 9.3e-7, 0.1e-7)
tv = pd.read_csv(ROOT/'validation'/'reference_table_versions.csv').set_index('lambda')
rec('A=22 inversion noise at lambda=1000 (a few 1e-8)', abs(tv.loc[1000.0, 'rel_current_minus_schur_A22']), 3.5e-8, 1.5e-8)
tel = pd.read_csv(ROOT/'validation'/'telegraph_inversion_test.csv')
rec('telegraph test: most negative error (-1.5e-10)', tel.rel_error_analytic_transform.min(), -1.5e-10, 0.05e-10)
rec('telegraph test: least negative error (-3e-11)', tel.rel_error_analytic_transform.max(), -2.9e-11, 0.15e-11)
upto('telegraph test: error equals predicted alias (1e-14)', (tel.rel_error_analytic_transform - tel.predicted_rel_residual_alias).abs().max(), 1.5e-14)
upto('telegraph test: solver m=2 vs analytic (7e-12)', (tel.rel_error_solver_m2 - tel.rel_error_analytic_transform).abs().max(), 7e-12)
upto('remaining alias bound 4 e^(-2A) at A=12 (1.6e-10)', 4*math.exp(-24), 1.6e-10)
lc = json.load(open(ROOT/'data'/'long_chain_remainder_check.json'))
lc_dev100 = float(pd.read_csv(ROOT/'data'/'long_chain_remainder_check.csv').set_index('lambda').loc[100.0, 'rel_dev_from_longchain'])
upto('the long-chain expansion + remainder, 1e3<=lambda<=2e4 (2.7e-9)', lc['max_abs_rel_residual_1e3_to_2e4'], 2.7e-9)
upto('  same, as a fraction of eps_int (0.13)', lc['max_abs_rel_residual_over_eps_1e3_to_2e4'], 0.13)
upto('  same with a lambda^-1 or lambda^-1/2 term (< 3e-9)', max(lc['max_abs_rel_residual_1e3_to_2e4_with_lambda_m1'], lc['max_abs_rel_residual_1e3_to_2e4_with_lambda_m1_2']), 3e-9)
upto('  shift of B from the lambda^-1/2 refit (1e-6)', abs(lc['refit_B_shift']), 1.5e-6)
upto('  same with lambda^-2 in place of lambda^-5/2 (< 3e-9)', lc['max_abs_rel_residual_1e3_to_2e4_with_lambda_m2_instead'], 3e-9)
upto('  same with the fit on 100..700 only (2.7e-9)', lc['max_abs_rel_residual_1e3_to_2e4_fit_100_to_700'], 2.7e-9)
rec('deviation from the long-chain expansion at L/p=100 (5.5e-6)', abs(lc_dev100), 5.5e-6, 0.05e-6)
rec('lambda^-3/2 coefficient of the remainder (-0.064)', lc['c2_lambda_m3_2'], -0.064, 0.001)
rec('residual at 7.5e4 (4e-8, 0.96 eps_int)', lc['residual_over_eps_above_2e4']['75000.0'], 0.96, 0.01)
rec('A = 10, 14 change of F_inf for lambda>=1e4 (up to 1.4 eps_int)', lc['max_abs_A_change_over_eps_lambda_ge_1e4'], 1.4, 0.05)
upto('FV: extrapolation-model sensitivity (up to 4e-5)', fvo.spatial_model_change_pct.abs().max()/100, 4.1e-5)
upto('FV: angular-window sensitivity (< 5e-6)', fvo.angular_fit_change_pct.abs().max()/100, 5e-6)
for lam, q in ((0.1, 0.0025), (1.0, 0.0009), (10.0, 0.0003)):
    rec(f'FV, order-trend variant at {lam:g} (+{q} %)', fvo.loc[lam, 'relative_difference_order_trend_pct'], q, 0.00006)
# Long-chain paths (supplementary material)
pl1 = pd.read_csv(ROOT/'data'/'heat_path_long.csv'); pool = pd.read_csv(ROOT/'data'/'heat_path_long_pooled.csv')
pl = pd.concat([pl1[pl1.lam < 5e3], pool])            # L/p = 1e4: four pooled batches of 20 000 chains
v = pool.sort_values('steps').iloc[-1]
rec('pooled path estimate at L/p=1e4 (128.890)', v.span, 128.890, 0.0005)
rec('  its standard error (0.079)', v.span_se, 0.079, 0.0005)
for bm, q in zip([float(x) for x in v.batch_means.split()], (129.09, 128.56, 129.13, 128.77)):
    rec(f'  batch mean ({q})', bm, q, 0.005)
b1 = pl1[pl1.lam == 1e4].sort_values('steps').iloc[-1]
rec('  one batch: standard error (0.16)', b1.span_se, 0.16, 0.005)
pair = {lam: pl[(pl.lam == lam)].sort_values('steps').iloc[-2] for lam in (1000.0, 10000.0)}
rec('paired difference, two finest grids, L/p=1e3 (0.03 %)', 100*pair[1000.0].pairwise_minus_fine/Fi(1000), 0.03, 0.005)
rec('paired difference, two finest grids, L/p=1e4 (0.06 %)', 100*pair[10000.0].pairwise_minus_fine/Fi(10000), 0.06, 0.01)
upto('finest-step bias h^2/24 at L/p=1e4 (< 3e-4)', (10000/131072)**2/24, 3e-4)
sec = [100*float(r.span_se)/Fi(float(r.lam)) for r in pd.concat([paths, pl]).sort_values('steps').groupby('lam').tail(1).itertuples()]
rec('sampling SE, smallest (4e-4)', min(sec)/100, 4.2e-4, 0.3e-4)
rec('sampling SE, largest (9e-4)', max(sec)/100, 9.1e-4, 0.3e-4)
allp = pd.concat([paths, pl]).sort_values('steps').groupby('lam').tail(1)
upto('all six lengths: |path - table| / SE (within 1)', max(abs(float(r.span) - Fi(float(r.lam)))/float(r.span_se) for r in allp.itertuples()), 1.0)
fvl = json.load(open(ROOT/'data'/'finite_volume_long_summary.json'))
rec('FV at L/p=100, observed order (-0.0031 %)', 100*fvl['rel_diff_observed_order'], -0.0031, 0.00006)
rec('FV at L/p=100, integer order (-0.0021 %)', 100*fvl['rel_diff_integer_order'], -0.0021, 0.00006)
rec('FV at L/p=100: observed spatial order (0.98)', fvl['observed_spatial_orders'][-1], 0.98, 0.006)
upto('FV at L/p=100: additivity defect (9e-6)', abs(fvl['additivity_defect_rel']), 9.5e-6)
rec('long-chain offset / F at L/p=100 (12 %)', 100*2*Z_M/Fi(100), 12, 0.5)
# The long-chain offset and the depletion layer
rec('long-chain offset 2 z_M (1.436)', 2*Z_M, 1.436, 0.0005)
rec('wall offset z_M (0.718)', Z_M, 0.718, 0.0005)
g = lambda l: l/3 - 1 + 2/l - 2*(1-math.exp(-l))/l**2
for lam in (1e3, 1e4, 1e5):           # delta = X/2 = 2 Rg/sqrt(pi) - z_M p + O(lambda^-1/2), the wall-offset expansion
    rem = float(F[LAM==lam][0])/2 - (2*math.sqrt(g(lam))/math.sqrt(math.pi) - Z_M)
    rec(f'the wall-offset expansion: remainder x sqrt(L/p) at L/p={lam:g} -> B/2 + sqrt(3/pi) (1.3006)', rem*math.sqrt(lam), B_COEF/2 + math.sqrt(3/math.pi), 0.005)
wall = lambda l: 2*math.sqrt(g(l))/math.sqrt(math.pi) - Fi(l)/2
for lam, q in ((100.0, 0.59), (1000.0, 0.68), (10000.0, 0.71)):
    rec(f'wall offset at L/p={lam:g} ({q} p)', wall(lam), q, 0.005)
rec('short-chain depletion: (L/4 - delta)/(L^2/p) -> 37/1536 at 0.01', (0.01/4 - float(F[LAM==0.01][0])/2)/0.01**2, 37/1536, 2e-4)
print(f'\n{sum(results)} of {len(results)} checks passed')'''),
])

def build(names):
    for name in names:
        n = CH[name]
        path = TUT / f'{name}.ipynb'
        nbf.write(n, path)
        subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                        '--inplace', '--ExecutePreprocessor.timeout=1200', str(path)], cwd=TUT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('built and executed', path.name, flush=True)


if __name__ == '__main__':
    sel = sys.argv[1:]
    names = [k for k in CH if not sel or any(k.startswith(x) for x in sel)]
    build(names)
