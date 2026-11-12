#!/usr/bin/env python3
"""Animations for the repository README (figures/readme/).

span_records.gif   The span of one conformation grows only while the chain end sets a new
                   record along the axis; the conformation is the one of Fig. 1.
rod_to_coil.gif    One long chain grown from a rod to a coil, with the mean span of the
                   reference table along the way.

Illustrations only: the chains are single seeded samples, and no number of the article is
computed here.  Run from the repository root:  python scripts/make_readme_media.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import plot_main_figures as pmf  # noqa: E402  (style, colours, the seeded chain of Fig. 1)

OUT = ROOT / 'figures' / 'readme'
BLUE, VERM, GREY = pmf.BLUE, pmf.VERM, pmf.GREY
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 9.5,
                     'ytick.labelsize': 9.5, 'legend.fontsize': 9.5, 'axes.linewidth': 0.9})


def box(ax):
    ax.tick_params(direction='in', top=True, right=True, which='both')


# --------------------------------------------------------------------------------------------
def span_records(fps=20):
    seed = pmf.choose_chain()
    s, r = pmf.illustrative_chain(seed=seed)
    x, z = r[:, 0] - r[:, 0].mean(), r[:, 2]
    n = len(s) - 1
    idx = np.unique(np.r_[np.linspace(0, n, 150).astype(int), n])
    hold = 30
    M, m = np.maximum.accumulate(z), np.minimum.accumulate(z)
    at_rec = np.r_[True, (z[1:] >= M[:-1]) | (z[1:] <= m[:-1])]

    fig = plt.figure(figsize=(9.6, 4.2), dpi=90)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.45], left=0.03, right=0.985, bottom=0.15,
                          top=0.86, wspace=0.2)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    pad = 0.15
    a.set_xlim(x.min() - pad, x.max() + 0.9)
    a.set_ylim(z.min() - pad, z.max() + pad)
    a.set_aspect('equal')
    a.axis('off')
    a.annotate('', xy=(x.min() - 0.05, z.min() + 0.9), xytext=(x.min() - 0.05, z.min() + 0.3),
               arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
    a.text(x.min() + 0.03, z.min() + 0.6, 'axis', fontsize=9.5, va='center')
    b.set_xlim(0, s[-1] * 1.07)
    b.set_ylim(z.min() - 0.3, z.max() + 0.55)
    b.set_xlabel('Contour position $s/p$')
    b.set_ylabel('Position along the axis $Z/p$')
    box(b)

    chain, = a.plot([], [], color=GREY, lw=1.6, solid_capstyle='round')
    end_a, = a.plot([], [], 'o', ms=7, zorder=5)
    hmax = a.axhline(0, color=BLUE, lw=1.0, ls=(0, (4, 2)))
    hmin = a.axhline(0, color=BLUE, lw=1.0, ls=(0, (4, 2)))
    span_a = a.annotate('', xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='<->', lw=1.2, color=BLUE, shrinkA=0, shrinkB=0))
    span_t = a.text(0, 0, '', color=BLUE, fontsize=11, va='center')
    trace, = b.plot([], [], color=GREY, lw=1.3, label='$Z(s)$')
    runM, = b.plot([], [], color=BLUE, lw=1.0, ls=(0, (4, 2)), label='running maximum and minimum')
    runm, = b.plot([], [], color=BLUE, lw=1.0, ls=(0, (4, 2)))
    rec = b.scatter([], [], s=6, color=BLUE, zorder=4, label='records: the span grows')
    end_b, = b.plot([], [], 'o', ms=7, zorder=5)
    b.legend(loc='lower left', frameon=False)
    title = fig.suptitle('', fontsize=12.5, y=0.97)
    xr = x.max() + 0.35

    def draw(f):
        k = idx[min(f, len(idx) - 1)]
        grow = at_rec[k] and k > 0 and f < len(idx)
        col = VERM if grow else 'black'
        chain.set_data(x[:k + 1], z[:k + 1])
        end_a.set_data([x[k]], [z[k]]); end_a.set_color(col)
        hmax.set_ydata([M[k]] * 2); hmin.set_ydata([m[k]] * 2)
        span_a.xy, span_a.xyann = (xr, M[k]), (xr, m[k])
        span_t.set_position((xr + 0.08, 0.5 * (M[k] + m[k]))); span_t.set_text(f'$X_c$ = {M[k] - m[k]:.2f}$p$')
        trace.set_data(s[:k + 1], z[:k + 1])
        runM.set_data(s[:k + 1], M[:k + 1]); runm.set_data(s[:k + 1], m[:k + 1])
        sel = at_rec[:k + 1]
        rec.set_offsets(np.c_[s[:k + 1][sel], z[:k + 1][sel]])
        end_b.set_data([s[k]], [z[k]]); end_b.set_color(col)
        if f >= len(idx) - 1:
            title.set_text('The span of a conformation grows only while its end sets a new record')
        else:
            title.set_text('End at a record: the span grows' if grow else 'End inside the range: the span stays the same')
            title.set_color(VERM if grow else 'black')
        if f >= len(idx) - 1:
            title.set_color('black')
        return chain,

    anim = FuncAnimation(fig, draw, frames=len(idx) + hold, blit=False)
    anim.save(OUT / 'span_records.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


# --------------------------------------------------------------------------------------------
def rod_to_coil(fps=12, lam_max=1000.0, n=200_000, seed=7):
    s, r = pmf.illustrative_chain(lam=lam_max, n=n, seed=seed)
    ref = pd.read_csv(ROOT / 'data' / 'wlc_mean_span_reference.csv')
    lam_ref, F_ref = ref['lambda'].to_numpy(), ref['F_continuum'].to_numpy()
    lams = np.geomspace(0.03, lam_max, 110)
    hold = 25

    fig = plt.figure(figsize=(9.6, 4.2), dpi=90)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.45], left=0.03, right=0.975, bottom=0.15,
                          top=0.86, wspace=0.22)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    a.set_aspect('equal')
    a.axis('off')
    w = np.geomspace(0.01, 1e5, 400)
    b.semilogx(lam_ref, F_ref / np.sqrt(lam_ref), 'o', ms=4, mfc='white', mec=BLUE, mew=1.0,
               label='mean span, this work')
    b.semilogx(w[w < 3], np.sqrt(w[w < 3]) / 2, color=VERM, ls='--', lw=1.3, label='rigid rod, $X = L/2$')
    b.axhline(pmf.A_GAUSS, color='black', ls=':', lw=1.0, label=r'Gaussian coil, $X = 4\sqrt{Lp/3\pi}$')
    b.set_xlim(0.01, 1e5); b.set_ylim(0, 1.45)
    b.set_xlabel('Contour length in persistence lengths, $L/p$')
    b.set_ylabel(r'Mean span $X/\sqrt{Lp}$')
    b.legend(loc='lower right', frameon=False)
    box(b)
    dot, = b.plot([], [], 'o', ms=10, color=VERM, zorder=5)
    chain, = a.plot([], [], color=GREY, lw=1.0)
    lines = [a.axhline(0, color=BLUE, lw=0.9, ls=(0, (4, 2))) for _ in range(2)]
    span_a = a.annotate('', xy=(1.0, 0), xytext=(1.0, 0),
                        arrowprops=dict(arrowstyle='<->', lw=1.1, color=BLUE, shrinkA=0, shrinkB=0))
    a.text(1.05, 0, 'span of\nthis chain', color=BLUE, fontsize=9.5, va='center', ha='left')
    title = fig.suptitle('', fontsize=12.5, y=0.97)

    def draw(f):
        lam = lams[min(f, len(lams) - 1)]
        k = max(2, int(round(lam / lam_max * n)))
        step = max(1, k // 1500)
        xx, zz = r[:k + 1:step, 0], r[:k + 1:step, 2]
        xx, zz = np.r_[xx, r[k, 0]], np.r_[zz, r[k, 2]]
        cx, cz = 0.5 * (xx.max() + xx.min()), 0.5 * (zz.max() + zz.min())
        half = 0.55 * max(xx.max() - xx.min(), zz.max() - zz.min(), 1e-9)
        chain.set_data((xx - cx) / half, (zz - cz) / half)
        chain.set_linewidth(1.8 if lam < 3 else 1.0 if lam < 100 else 0.6)
        top, bot = (zz.max() - cz) / half, (zz.min() - cz) / half
        for ln, v in zip(lines, (top, bot)):
            ln.set_ydata([v] * 2)
        span_a.xy, span_a.xyann = (1.0, top), (1.0, bot)
        a.set_xlim(-1.1, 1.55); a.set_ylim(-1.1, 1.1)
        dot.set_data([lam], [np.interp(math.log(lam), np.log(lam_ref), F_ref) / math.sqrt(lam)])
        regime = 'nearly a rod' if lam < 0.3 else 'semiflexible' if lam < 30 else 'a coil'
        shown = f'{lam:.2g}' if lam < 10 else f'{lam:.0f}'
        title.set_text(f'One chain grown to $L/p$ = {shown}: {regime}')
        return chain,

    anim = FuncAnimation(fig, draw, frames=len(lams) + hold, blit=False)
    anim.save(OUT / 'rod_to_coil.gif', writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    span_records()
    rod_to_coil()
    for p in sorted(OUT.glob('*.gif')):
        print(p.relative_to(ROOT), f'{p.stat().st_size / 1e6:.2f} MB')
