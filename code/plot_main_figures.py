#!/usr/bin/env python3
"""Figures of "Mean span dimension of an ideal wormlike chain" (main text and supplementary material).

Figure 1  Concept: an illustrative conformation and its span; the projected trace
          whose running extrema grow only at records; the same trace read
          backward from a growing record (first-return reading of the record form of the growth rate); the
          angle-averaged Milne solution and its extrapolation length z_M.
Figure 2  Crossover: compensated span with the short- and long-chain expansions;
          span-to-gyration ratio.
Figure 3  Asymptotic coefficients: 37/768, 9/256, the boundary offset 2 z_M, and the
          wall offset z_M of the depletion layer.
Figure 4  The practical formula and its error at every node.
Fig. S1   (supplementary material) Tests independent of the spectral solver: heat-kernel
          paths and finite-volume solutions; paired step-size differences.

Every quantitative mark is read from an archived CSV/JSON in ``data/``; no
value is refitted here.  The illustrative conformation in Fig. 1 is generated
from a seeded small-step rotational diffusion and carries no quantitative claim.
JCP column widths: single 8.5 cm, double 17.2 cm.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, MultipleLocator, NullFormatter

from figure_io import save_figure

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'

CM = 1 / 2.54
DOUBLE = 17.2 * CM

# Restrained palette: one signal family (numerical solution), one theory accent,
# one neutral family for bounds and reference lines.
BLUE = '#1f5f8b'      # spectral first-return solution
VERM = '#c4572a'      # asymptotic theory
GREEN = '#4f7f3a'     # independent path sampling
PURPLE = '#6b4c9a'   # finite-volume discretization
GREY = '#8c8c8c'
BAND = '#e3e3e3'

Z_M = 0.71821164      # Milne extrapolation length (Amic et al. 1996)
B_COEF = 0.64680      # conditional flexible coefficient from the Milne boundary moments (this work)
A_GAUSS = 4 / math.sqrt(3 * math.pi)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic',
    'mathtext.bf': 'Liberation Sans:bold',
    'mathtext.sf': 'Liberation Sans',
    # Math sub/superscripts render at ~70% of the parent size; keeping every
    # parent at >= 7.2 pt holds all rendered glyphs above the 5 pt floor.
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.2,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.45,
    'ytick.minor.width': 0.45,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.7,
    'ytick.minor.size': 1.7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.frameon': False,
    'legend.handlelength': 1.8,
    'lines.linewidth': 1.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


# ----------------------------------------------------------------- data access
def read(name: str) -> list[dict]:
    with (DATA / name).open(newline='') as f:
        return list(csv.DictReader(f))


def reference():
    rows = read('wlc_mean_span_reference.csv')
    lam = np.array([float(r['lambda']) for r in rows])
    F = np.array([float(r['F_continuum']) for r in rows])
    return lam, F


def rg2_over_p2(lam):
    """Exact <R_g,c^2>/p^2 of the WLC with a cancellation-safe short series."""
    x = np.atleast_1d(np.asarray(lam, dtype=float))
    out = np.empty_like(x)
    s = x < 0.3
    z = x[s]
    out[s] = z * z * (1/12 + z*(-1/60 + z*(1/360 + z*(-1/2520 + z*(1/20160
             + z*(-1/181440 + z*(1/1814400 - z/19958400)))))))
    z = x[~s]
    out[~s] = z/3 - 1 + 2/z + 2*np.expm1(-z)/(z*z)
    return out


def short_expansion(lam):
    """Two-term short-chain span, F = lam/2 - 37 lam^2/768."""
    return lam/2 - 37/768*lam**2


def long_expansion(lam):
    """Three-term long-chain span (offset 2 z_M and conditional B)."""
    return A_GAUSS*np.sqrt(lam) - 2*Z_M + B_COEF/np.sqrt(lam)


def panel_label(ax, tag, x=-0.02, y=1.0):
    ax.text(x, y, tag, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, fontweight='bold')


def log_x(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*.1, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------- figure 1
def illustrative_chain(lam=6.0, n=3000, seed=20261028):
    """Seeded small-step rotational diffusion; <t(s).t(s')> ~ exp(-|s-s'|/p)."""
    rng = np.random.default_rng(seed)
    h = lam / n
    t = rng.normal(size=3)
    t /= np.linalg.norm(t)
    T = np.empty((n + 1, 3))
    T[0] = t
    for k in range(n):
        xi = rng.normal(size=3) * math.sqrt(h)
        xi -= xi.dot(t) * t
        t = t + xi
        t /= np.linalg.norm(t)
        T[k + 1] = t
    r = np.vstack([np.zeros(3), np.cumsum(0.5*(T[1:] + T[:-1])*h, axis=0)])
    s = np.linspace(0, lam, n + 1)
    return s, r


def choose_chain():
    """Pick, deterministically, a seed whose span visibly exceeds |R.e|."""
    best = None
    for seed in range(20261028, 20261028 + 400):
        s, r = illustrative_chain(seed=seed)
        z = r[:, 2]
        span = z.max() - z.min()
        endp = abs(z[-1] - z[0])
        imax, imin = z.argmax(), z.argmin()
        interior = 0.08*len(z) < imax < 0.92*len(z) and 0.08*len(z) < imin < 0.92*len(z)
        score = (span - endp)/span
        if interior and 2.0 < span < 3.4 and score > 0.5:
            if best is None or score > best[0]:
                best = (score, seed)
    return best[1]


def figure1():
    seed = choose_chain()
    s, r = illustrative_chain(seed=seed)
    x, z = r[:, 0] - r[:, 0].mean(), r[:, 2]
    zmax, zmin = z.max(), z.min()
    M = np.maximum.accumulate(z)
    m = np.minimum.accumulate(z)

    fig = plt.figure(figsize=(DOUBLE, 7.0*CM))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.5, 1.1], left=0.03, right=0.985,
                          bottom=0.13, top=0.94, wspace=0.34, hspace=0.62)
    a = fig.add_subplot(gs[:, 0])
    b = fig.add_subplot(gs[:, 1])
    c = fig.add_subplot(gs[0, 2])
    d = fig.add_subplot(gs[1, 2])

    # (a) conformation projected onto a plane containing e
    a.plot(x, z, color=GREY, lw=1.1, solid_capstyle='round', zorder=2)
    a.plot(x[0], z[0], 'o', ms=4.2, mfc='white', mec='black', mew=0.8, zorder=4)
    a.plot(x[-1], z[-1], 's', ms=3.8, mfc='black', mec='black', zorder=4)
    for zz in (zmax, zmin):
        a.axhline(zz, color=BLUE, lw=0.7, ls=(0, (4, 2)), zorder=1)
    for i in (z.argmax(), z.argmin()):
        a.plot(x[i], z[i], 'o', ms=3.0, color=BLUE, zorder=5)
    xr = x.max() + 1.05
    a.annotate('', xy=(xr, zmax), xytext=(xr, zmin), annotation_clip=False,
               arrowprops=dict(arrowstyle='<->', lw=0.8, color=BLUE, shrinkA=0, shrinkB=0))
    a.text(xr + 0.12, 0.5*(zmax + zmin), r'$X_c$', color=BLUE, va='center', ha='left', fontsize=8)
    xe = x.max() - 0.2
    a.annotate('', xy=(xe, z[-1]), xytext=(xe, z[0]), annotation_clip=False,
               arrowprops=dict(arrowstyle='<->', lw=0.7, color='black', shrinkA=0, shrinkB=0))
    a.text(xe + 0.2, 0.5*(z[0] + z[-1]), r'$|\mathbf{R}\cdot\hat{\mathbf{e}}|$', va='center',
           ha='left', fontsize=7.2)
    xa = x.min() - 0.75
    a.annotate('', xy=(xa, zmin + 0.95), xytext=(xa, zmin + 0.25), annotation_clip=False,
               arrowprops=dict(arrowstyle='-|>', lw=0.8, color='black', mutation_scale=7))
    a.text(xa + 0.1, zmin + 0.62, r'$\hat{\mathbf{e}}$', ha='left', va='center', fontsize=8)
    # 'box' keeps the requested limits (with 'datalim' matplotlib overrode the
    # x-limits and clipped the X_c and e arrows).
    a.set_xlim(x.min() - 1.0, xr + 0.6)
    a.set_ylim(zmin - 0.35, zmax + 0.35)
    a.set_aspect('equal', adjustable='box')
    a.axis('off')
    fig.text(0.012, 0.955, '(a)', fontsize=8.5, fontweight='bold', ha='left', va='bottom')

    # (b) projected trace, running extrema, record segments
    b.plot(s, z, color=GREY, lw=1.0, zorder=2, label=r'$Z(s)=\mathbf{r}(s)\cdot\hat{\mathbf{e}}$')
    b.plot(s, M, color=BLUE, lw=0.9, ls=(0, (4, 2)), zorder=3, label='Running maximum and minimum')
    b.plot(s, m, color=BLUE, lw=0.9, ls=(0, (4, 2)), zorder=3)
    tol = 1e-12
    rec = (np.abs(z - M) < tol) | (np.abs(z - m) < tol)
    zrec = np.where(rec, z, np.nan)
    b.plot(s, zrec, color=BLUE, lw=2.4, solid_capstyle='butt', zorder=4,
           label='Records: span grows')
    L = s[-1]
    b.annotate('', xy=(L + 0.12, M[-1]), xytext=(L + 0.12, m[-1]),
               arrowprops=dict(arrowstyle='<->', lw=0.8, color=BLUE, shrinkA=0, shrinkB=0))
    b.text(L + 0.22, 0.5*(M[-1] + m[-1]), r'$X_c$', color=BLUE, va='center', fontsize=8)
    # a point s* on the growing running-minimum segment, used in panel (c)
    grad = np.gradient(z, s)
    seg = np.where((np.abs(z - m) < tol) & (grad < 0) & (s > 1.0))[0]
    i_star = int(seg[int(0.8*len(seg))])
    b.plot(s[i_star], z[i_star], 'D', ms=3.6, mfc='white', mec=VERM, mew=0.9, zorder=6)
    b.annotate(r'$s^*$', xy=(s[i_star], z[i_star]), xytext=(s[i_star] - 0.55, z[i_star] - 0.12),
               color=VERM, fontsize=7.5, ha='right', va='center',
               arrowprops=dict(arrowstyle='-', lw=0.5, color=VERM, shrinkA=0, shrinkB=2))
    b.set_xlim(0, L + 0.6)
    b.set_xlabel(r'Contour position $s/p$')
    b.set_ylabel(r'Projected position $Z/p$')
    b.legend(loc='upper left', ncol=1, borderaxespad=0.3)
    b.set_ylim(zmin - 0.25, zmax + 0.95)
    panel_label(b, '(b)')

    # (c) the same trace read backward from the record point s*: the distance below
    #     the record stays positive, i.e. the reversed contour has not returned.
    t = s[i_star] - s[: i_star + 1][::-1]
    below = z[: i_star + 1][::-1] - z[i_star]       # height above the record (a running minimum)
    v = float(-grad[i_star])
    c.axhline(0, color=BLUE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    c.text(t[-1], 0.03, 'record plane', color=BLUE, fontsize=7.2, ha='right', va='bottom')
    c.plot(t, below, color=GREY, lw=1.1, zorder=2)
    tt = np.array([0, 0.45])
    c.plot(tt, v*tt, color=VERM, lw=1.0, zorder=3)
    c.text(0.5, v*0.45, r'slope $v$', color=VERM, fontsize=7.2, ha='left', va='center')
    c.set_xlim(0, t[-1]*1.02)
    c.set_ylim(-0.12, max(below)*1.15)
    c.set_xlabel(r'Length read backward from $s^*$, $\sigma/p$', labelpad=1.5)
    c.set_ylabel(r'$[Z(s^*-\sigma)-Z(s^*)]/p$', fontsize=7.2)
    panel_label(c, '(c)')

    # (d) angle-averaged Milne solution and its linear far field y + z_M
    from build_boundary_moments import stationary_milne
    sol = stationary_milne(128)
    yy = np.linspace(0, 2.0, 300)
    psi = 0.5*(sol.evaluate(yy) @ sol.weights)
    d.axhline(0, color='black', lw=0.6, zorder=1)
    d.axvline(0, color=BLUE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    d.text(-0.5, 1.9, 'beyond\nthe record', color=BLUE, fontsize=6.6, ha='center', va='top', linespacing=0.95)
    ya = np.linspace(-sol.z, 2.0, 60)
    d.plot(yy, psi, color=GREY, lw=1.6, zorder=3, label=r'$\langle\Psi\rangle_u$')
    d.plot(ya, ya + sol.z, color=VERM, lw=0.9, ls=(0, (1.5, 1.2)), zorder=4, label=r'$y/p+z_M$')
    d.annotate('', xy=(-sol.z, -0.22), xytext=(0, -0.22), annotation_clip=False,
               arrowprops=dict(arrowstyle='<->', lw=0.7, color=VERM, shrinkA=0, shrinkB=0))
    d.text(-sol.z/2, -0.3, r'$z_M$', color=VERM, fontsize=7.5, ha='center', va='top')
    d.set_xlim(-1.0, 1.25)
    d.set_ylim(-0.8, 2.3)
    d.set_xlabel(r'Distance from the record, $y/p$', labelpad=1.5)
    d.set_ylabel(r'$\langle\Psi\rangle_u$', labelpad=1)
    d.legend(loc='lower right', borderaxespad=0.2, handlelength=1.6)
    panel_label(d, '(d)')
    return fig, {'seed': seed, 'lambda': float(L), 's_star': float(s[i_star]), 'v': v,
                 'psi_wall_mean': float(psi[0]), 'z_M': sol.z}


# --------------------------------------------------------------- figure 2
def figure2():
    lam, F = reference()
    xx = np.geomspace(0.01, 1e5, 1200)

    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 6.4*CM))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.175, top=0.93, wspace=0.30)

    # (a) compensated span
    xs = xx[xx <= 1.0]
    a.plot(xs, short_expansion(xs)/np.sqrt(xs), color=VERM, ls=(0, (5, 2)), lw=1.0, zorder=4,
           label=r'$\lambda/2-37\lambda^2/768$')
    xl = xx[xx >= 3.0]
    a.plot(xl, long_expansion(xl)/np.sqrt(xl), color=VERM, ls=(0, (1.2, 1.2)), lw=1.2, zorder=4,
           label=r'$4(\lambda/3\pi)^{1/2}-2z_M+B\lambda^{-1/2}$')
    a.axhline(A_GAUSS, color='black', lw=0.6, ls=(0, (2, 2)), zorder=1)
    a.text(0.012, A_GAUSS + 0.025, 'Gaussian coil, 4/√(3π)', fontsize=7.2, va='bottom', ha='left')
    a.plot(lam, F/np.sqrt(lam), 'o', ms=2.3, mfc='white', mec=BLUE, mew=0.6, zorder=3,
           label='First-return solution')
    log_x(a)
    a.set_xlim(0.008, 1.3e5)
    a.set_ylim(0, 1.45)
    a.set_xlabel(r'$\lambda=L/p$')
    a.set_ylabel(r'$X/(Lp)^{1/2}$')
    h, l = a.get_legend_handles_labels()
    a.legend([h[2], h[0], h[1]], [l[2], l[0], l[1]], loc='lower right',
             borderaxespad=0.3, handlelength=2.2)
    panel_label(a, '(a)')

    # (b) span-to-gyration ratio and depletion thickness
    ratio = F/np.sqrt(rg2_over_p2(lam))
    rod, coil = math.sqrt(3), 4/math.sqrt(math.pi)
    b.axhline(rod, color='black', lw=0.6, ls=(0, (2, 2)))
    b.axhline(coil, color='black', lw=0.6, ls=(0, (2, 2)))
    b.text(0.012, rod + 0.012, 'Rod, √3', fontsize=7.2, va='bottom')
    b.text(0.012, coil + 0.008, 'Gaussian coil, 4/√π', fontsize=7.2, va='bottom')
    xs = xx[xx <= 8.0]
    b.plot(xs, rod*(1 + 7/1920*xs), color=VERM, ls=(0, (5, 2)), lw=1.0, zorder=4,
           label=r'√3 $(1+7\lambda/1920)$')
    xl = xx[xx >= 30.0]
    coil_ratio = coil - 2*math.sqrt(3)*Z_M/np.sqrt(xl) + (math.sqrt(3)*B_COEF + 6/math.sqrt(math.pi))/xl
    b.plot(xl, coil_ratio, color=VERM, ls=(0, (1.2, 1.2)), lw=1.2, zorder=4,
           label=r'4/√π $-$ 2√3 $z_M\lambda^{-1/2}+c_1\lambda^{-1}$')
    b.plot(lam, ratio, 'o', ms=2.3, mfc='white', mec=BLUE, mew=0.6, zorder=3,
           label='First-return solution')
    log_x(b)
    b.set_xlim(0.008, 1.3e5)
    b.set_ylim(1.68, 2.32)
    b.set_xlabel(r'$\lambda=L/p$')
    b.set_ylabel(r'$X/R_{\mathrm{g}}$')
    h, l = b.get_legend_handles_labels()
    b.legend([h[2], h[0], h[1]], [l[2], l[0], l[1]], loc='upper left',
             bbox_to_anchor=(0.0, 0.84), borderaxespad=0.3, handlelength=2.2)
    panel_label(b, '(b)')
    return fig


# --------------------------------------------------------------- figure 3
def figure3():
    stiff = read('stiff_alias_summary.csv')
    sx = np.array([float(r['lambda']) for r in stiff])
    sf = np.array([float(r['F_extrap']) for r in stiff])
    deficit = (sx/2 - sf)/sx**2

    ex = read('endpoint_excursion_diagnostic.csv')
    sel = {}
    for r in ex:
        lmb = float(r['lambda'])
        if lmb <= 0.1 and (lmb not in sel or int(r['steps']) > int(sel[lmb]['steps'])):
            sel[lmb] = r
    ex_l = np.array(sorted(sel))
    ex_v = np.array([float(sel[k]['excursion_over_lambda2']) for k in ex_l])
    ex_e = np.array([2*float(sel[k]['excursion_se']) for k in ex_l])

    lam, F = reference()
    mask = lam >= 5
    u = 1/np.sqrt(lam[mask])
    off = A_GAUSS*np.sqrt(lam[mask]) - F[mask]

    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 11.4*CM))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.095, top=0.955, wspace=0.30, hspace=0.42)
    (a, b), (c, d) = axs

    a.axhline(37/768, color=VERM, lw=1.0, ls=(0, (5, 2)))
    a.text(0.006, 37/768 + 0.00005, r'$37/768$', color=VERM, ha='left', va='bottom', fontsize=7.5)
    a.plot(sx, deficit, 'o', ms=3.4, color=BLUE)
    a.set_xlim(0, 0.215)
    a.set_ylim(0.04685, 0.04840)
    a.set_xlabel(r'$\lambda=L/p$')
    a.set_ylabel(r'$(\lambda/2-X/p)/\lambda^{2}$')
    a.yaxis.set_major_locator(MultipleLocator(0.0005))
    a.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    panel_label(a, '(a)')

    b.axhline(9/256, color=VERM, lw=1.0, ls=(0, (5, 2)))
    b.text(0.0315, 9/256 + 0.0002, r'$9/256$', color=VERM, ha='center', va='bottom', fontsize=7.5)
    b.errorbar(ex_l, ex_v, yerr=ex_e, fmt='s', ms=3.2, color=GREEN, capsize=1.6, elinewidth=0.7,
               capthick=0.7)
    log_x(b)
    b.set_xlim(0.0035, 0.14)
    b.set_ylim(0.0330, 0.0372)
    b.set_xlabel(r'$\lambda=L/p$')
    b.set_ylabel(r'$\left[X-\langle|\mathbf{R}\cdot\hat{\mathbf{e}}|\rangle\right]/(p\lambda^{2})$')
    panel_label(b, '(b)')

    t = np.linspace(0, 0.47, 100)
    c.plot(t, 2*Z_M - B_COEF*t, color=VERM, lw=0.9, ls=(0, (1.2, 1.2)),
           label=r'$2z_M-B\lambda^{-1/2}$')
    c.axhline(2*Z_M, color=VERM, lw=1.0, ls=(0, (5, 2)))
    c.text(0.46, 2*Z_M + 0.01, r'$2z_M$', color=VERM, ha='right', va='bottom', fontsize=7.5)
    c.plot(u, off, 'o', ms=3.0, mfc='white', mec=BLUE, mew=0.8, zorder=3)
    c.set_xlim(0, 0.47)
    c.set_ylim(1.12, 1.475)
    c.set_xlabel(r'$\lambda^{-1/2}=(p/L)^{1/2}$')
    c.set_ylabel(r'$4(\lambda/3\pi)^{1/2}-X/p$')
    c.legend(loc='lower left', borderaxespad=0.3)
    panel_label(c, '(c)')

    # (d) wall offset of the depletion layer, 2 R_g/sqrt(pi) - delta with delta = X/2
    g = rg2_over_p2(lam[mask])
    wall = 2*np.sqrt(g)/math.sqrt(math.pi) - F[mask]/2
    d.plot(t, Z_M - (B_COEF/2 + math.sqrt(3/math.pi))*t, color=VERM, lw=0.9, ls=(0, (1.2, 1.2)),
           label=r'$z_M-(B/2+\sqrt{3/\pi})\lambda^{-1/2}$')
    d.axhline(Z_M, color=VERM, lw=1.0, ls=(0, (5, 2)))
    d.text(0.46, Z_M + 0.01, r'$z_M$', color=VERM, ha='right', va='bottom', fontsize=7.5)
    d.plot(u, wall, 'o', ms=3.0, mfc='white', mec=BLUE, mew=0.8, zorder=3)
    d.set_xlim(0, 0.47)
    d.set_ylim(0.0, 0.86)
    d.set_xlabel(r'$\lambda^{-1/2}=(p/L)^{1/2}$')
    d.set_ylabel(r'$(2R_{\mathrm{g}}/\sqrt{\pi}-\delta)/p$')
    d.legend(loc='lower left', borderaxespad=0.3)
    panel_label(d, '(d)')
    return fig


# --------------------------------------------------------------- supplementary Fig. S1
def figure_checks():
    """Supplementary Fig. S1: tests independent of the spectral solver (a) and paired step-size differences (b)."""
    # the pooled four-batch estimate replaces the single batch at lambda = 1e4
    paths = (read('heat_path_coupled.csv') + [r for r in read('heat_path_long.csv') if float(r['lam']) < 5e3]
             + read('heat_path_long_pooled.csv'))
    ref = {float(r['lambda']): float(r['F_continuum']) for r in read('wlc_mean_span_reference.csv')}
    fv = read('finite_volume_comparison.csv')
    with (DATA/'finite_volume_long_summary.json').open() as f:
        fvl = json.load(f)
    fvo = {float(r['lambda']): float(r['relative_difference_pct']) for r in read('finite_volume_order_analysis.csv')}

    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 6.4*CM))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.175, top=0.93, wspace=0.30)

    a.axhline(0, color='black', lw=0.6, ls=(0, (2, 2)))
    first = True
    for t in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        rows = [r for r in paths if float(r['lam']) == t]
        v = max(rows, key=lambda r: int(r['steps']))
        a.errorbar(t/1.13, 100*(float(v['span'])/ref[t] - 1), yerr=200*float(v['span_se'])/ref[t],
                   fmt='o', ms=3.4, color=GREEN, capsize=1.6, elinewidth=0.7, capthick=0.7,
                   label='Chain paths' if first else None)
        first = False
    first = True
    for r in fv:
        t = float(r['lambda'])
        a.errorbar(t*1.13, float(r['relative_difference_pct']),
                   yerr=float(r['FV_combined_sensitivity_rel_pct']), fmt='s', ms=3.2, color=PURPLE,
                   capsize=1.6, elinewidth=0.7, capthick=0.7,
                   label='FV, integer order' if first else None)
        a.plot(t*1.13, fvo[t], 's', ms=3.2, mfc='white', mec=PURPLE, mew=0.8, zorder=4,
               label='FV, free order' if first else None)
        first = False
    # lambda = 100: separable extrapolation (code/build_finite_volume_long.py)
    sens = 100*(abs(fvl['rel_diff_integer_order'] - fvl['rel_diff_observed_order']) + abs(fvl['additivity_defect_rel']))
    a.errorbar(100*1.13, 100*fvl['rel_diff_integer_order'], yerr=sens, fmt='s', ms=3.2, color=PURPLE,
               capsize=1.6, elinewidth=0.7, capthick=0.7)
    a.plot(100*1.13, 100*fvl['rel_diff_observed_order'], 's', ms=3.2, mfc='white', mec=PURPLE, mew=0.8, zorder=4)
    log_x(a)
    a.set_xlim(0.05, 2.5e4)
    a.set_ylim(-0.42, 0.48)
    h, l = a.get_legend_handles_labels()
    order = [l.index(k) for k in ('Chain paths', 'FV, integer order', 'FV, free order')]
    a.legend([h[i] for i in order], [l[i] for i in order], loc='lower left', borderaxespad=0.3,
             handlelength=1.2, fontsize=7.0)
    a.set_xlabel(r'$\lambda=L/p$')
    a.set_ylabel(r'Deviation from spectral solution (%)')
    panel_label(a, '(a)')

    coupled = read('heat_path_coupled.csv')
    b.axhline(0, color='black', lw=0.6, ls=(0, (2, 2)))
    for t, col, mk in ((1.0, '#9cc184', 'o'), (10.0, GREEN, 's'), (100.0, '#22401a', '^')):
        rows = sorted([r for r in coupled if float(r['lam']) == t], key=lambda r: float(r['dt']))[1:]
        b.errorbar([float(r['dt']) for r in rows],
                   [100*float(r['pairwise_minus_fine'])/ref[t] for r in rows],
                   yerr=[200*float(r['pairwise_se'])/ref[t] for r in rows],
                   fmt=mk + '-', ms=3.2, lw=0.8, color=col, capsize=1.6, elinewidth=0.7,
                   capthick=0.7, label=fr'$\lambda={t:g}$')
    log_x(b)
    b.set_xlabel(r'Reduced contour step $\Delta s/p$')
    b.set_ylabel(r'$(F_{\Delta s}-F_{\mathrm{finest}})/F$ (%)')
    b.legend(loc='lower left', borderaxespad=0.3)
    panel_label(b, '(b)')
    return fig


# --------------------------------------------------------------- figure 4
def rational(lam):
    with (DATA/'rational_fit_coefficients.json').open() as f:
        c = {k: round(v, 8) for k, v in json.load(f)['central'].items()}   # printed coefficients
    w = np.sqrt(lam)
    return 0.5*w*w*(1 + c['a']*w + c['b']*w**2 + c['c']*w**3) / (
        1 + c['a']*w + c['d']*w**2 + c['e']*w**3 + c['f']*w**4)


def figure4():
    lam, F = reference()
    xx = np.geomspace(0.01, 1e5, 1200)

    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 6.4*CM))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.175, top=0.93, wspace=0.30)

    a.plot(lam, F/np.sqrt(lam), 'o', ms=3.4, mfc='white', mec=BLUE, mew=0.8, zorder=2,
           label='First-return solution')
    a.plot(xx, rational(xx)/np.sqrt(xx), color=VERM, lw=1.1, zorder=3, label='Practical formula')
    a.axhline(A_GAUSS, color='black', lw=0.6, ls=(0, (2, 2)), zorder=1)
    a.text(0.012, A_GAUSS + 0.025, 'Gaussian coil, 4/√(3π)', fontsize=7.2, va='bottom', ha='left')
    log_x(a)
    a.set_xlim(0.008, 1.3e5)
    a.set_ylim(0, 1.45)
    a.set_xlabel(r'$\lambda=L/p$')
    a.set_ylabel(r'$X/(Lp)^{1/2}$')
    a.legend(loc='lower right', borderaxespad=0.3)
    panel_label(a, '(a)')

    err = 100*(rational(lam)/F - 1)
    split = {float(r['lambda']): r['set'] for r in read('reference_nodes_with_split.csv')}
    cal = np.array([split.get(float(x)) == 'calibration' for x in lam])
    hold = np.array([split.get(float(x)) == 'holdout' for x in lam])
    ext = ~(cal | hold)          # the nine nodes above the calibration range
    b.axhline(0, color='black', lw=0.6, ls=(0, (2, 2)))
    b.axvspan(0.008, 5000, color='#f2f2f2', lw=0, zorder=0)
    b.axvline(5000, color=GREY, lw=0.6, ls=(0, (1.2, 1.2)), zorder=1)
    b.text(3800, 0.113, '← calibration range', fontsize=7.2, ha='right', va='top', color='#555555')
    b.plot(lam[cal], err[cal], 'o', ms=3.0, mfc=BLUE, mec=BLUE, mew=0.8, zorder=3,
           label='Calibration nodes (34)')
    b.plot(lam[hold], err[hold], 'o', ms=3.0, mfc='white', mec=BLUE, mew=0.8, zorder=3,
           label='Withheld, interleaved (24)')
    b.plot(lam[ext], err[ext], 'D', ms=2.8, mfc='white', mec='#333333', mew=0.8, zorder=4,
           label=r'Withheld, $\lambda>5000$ (9)')
    b.legend(loc='lower left', borderaxespad=0.3, handlelength=1.2)
    log_x(b)
    b.set_xlim(0.008, 1.3e5)
    b.set_ylim(-0.17, 0.12)
    b.set_xlabel(r'$\lambda=L/p$')
    b.set_ylabel('Relative error of the formula (%)')
    panel_label(b, '(b)')
    return fig

def save(fig, name):
    save_figure(fig, FIG/(name + '.pdf'))
    save_figure(fig, FIG/(name + '.png'), dpi=600)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig, meta = figure1()
    save(fig, 'fig1_concept')
    print('Fig. 1 illustrative conformation:', meta)
    save(figure2(), 'fig2_crossover')
    save(figure3(), 'fig3_asymptotics')
    save(figure4(), 'fig4_practical_formula')
    save(figure_checks(), 'figSM1_independent_checks')
    print('Wrote main-article Figs. 1-4 and supplementary Fig. S1.')


if __name__ == '__main__':
    main()
