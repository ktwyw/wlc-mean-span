#!/usr/bin/env python3
"""Stationary Milne boundary moments: deterministic checks of analytical identities.

The null/Jordan sector is eliminated analytically in Legendre coordinates.
No small-s fit and no Monte Carlo simulation enters B_from_moments.
All moments are quadrature approximations at the reported angular order.
"""
from __future__ import annotations
import argparse, csv, json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, solve
from wlc_pn_transfer import angular_system

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class MilneSolution:
    m: int
    z: float
    nodes: np.ndarray
    weights: np.ndarray
    boundary: np.ndarray
    decay_lengths: np.ndarray
    modes: np.ndarray
    amplitudes: np.ndarray
    incoming_residual: float

    def evaluate(self, y: np.ndarray) -> np.ndarray:
        """Return Psi(y,u) at the original Gauss nodes; first axis is y."""
        heights = np.atleast_1d(np.asarray(y, dtype=float))
        if np.any(heights < 0):
            raise ValueError('heights must be nonnegative')
        return (heights[:,None] - self.nodes[None,:] + self.z
                + (np.exp(-heights[:,None]/self.decay_lengths[None,:])
                   * self.amplitudes[None,:]) @ self.modes.T)


def stationary_milne(m: int) -> MilneSolution:
    """Eliminate a_1=0 and a_0=-2 a_2/sqrt(5) for decaying modes."""
    if m < 6 or m % 2:
        raise ValueError('m must be even and at least 6')
    system = angular_system(m)
    ell = np.arange(1,m,dtype=float)
    off = ell / np.sqrt((2*ell-1)*(2*ell+1))
    J = np.diag(off,1)+np.diag(off,-1)
    invroot = 1/np.sqrt(np.diag(system.Lambda)[2:])
    reduced = invroot[:,None]*J[2:,2:]*invroot[None,:]
    alpha, vec = eigh(reduced,check_finite=False)
    pos = alpha > 0
    if np.count_nonzero(pos) != m//2-1:
        raise RuntimeError('unexpected reduced stationary mode count')
    alpha = alpha[pos]
    moment_modes = np.zeros((m,len(alpha)))
    moment_modes[2:] = invroot[:,None]*vec[:,pos]
    moment_modes[0] = -2/np.sqrt(5)*moment_modes[2]
    weighted_modes = system.V.T @ moment_modes
    incoming=system.positive
    match=np.column_stack((system.h[incoming],weighted_modes[incoming]))
    coeff=solve(match,system.h[incoming]*system.nodes[incoming],check_finite=False)
    z=float(coeff[0]); amplitudes=coeff[1:]
    modes=weighted_modes/system.h[:,None]
    boundary=z-system.nodes+modes@amplitudes
    residual=float(np.max(np.abs(boundary[incoming])))
    return MilneSolution(m,z,system.nodes,system.weights,boundary,alpha,modes,amplitudes,residual)


def moment_record(sol: MilneSolution) -> dict[str,float|int]:
    neg=sol.nodes<0; v=-sol.nodes[neg]; h=sol.boundary[neg]; w=sol.weights[neg]
    M=np.array([np.dot(w,v**n*h) for n in range(5)])
    I_direct=float(0.5*np.dot(sol.weights,sol.modes@(sol.decay_lengths*sol.amplitudes)))
    I_moment=float(M[3]/2-1/5)
    B=float(np.sqrt(3/np.pi)*(2*sol.z**2-2*M[3]+17/45))
    variance=float(1.5*M[3]-sol.z**2)
    lower=float(np.sqrt(3/np.pi)*(2*sol.z**2-4*sol.z/3+17/45))
    upper=float(np.sqrt(3/np.pi)*(2*sol.z**2/3+17/45))
    a2=float(sol.z**2-4/45-I_direct)
    B_alt=float((2*np.sqrt(3)*(a2-I_direct)-11/(15*np.sqrt(3)))/np.sqrt(np.pi))
    rec={'m':sol.m,'z_M':sol.z,**{f'M{i}':float(M[i]) for i in range(5)},
         'B_from_moments':B,
         'variance_flux':variance,'B_lower_bound':lower,'B_upper_bound':upper,
         'I_direct':I_direct,'I_from_M3':I_moment,'a2_reciprocal':a2,
         'M1_residual':float(M[1]-2/3),'M2_residual':float(M[2]-2*sol.z/3),
         'I_residual':I_direct-I_moment,'B_representation_residual':B-B_alt,
         'incoming_residual':sol.incoming_residual,'outgoing_min':float(h.min())}
    return rec


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--orders',type=int,nargs='+',default=[32,64,96,128,192,256,512])
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[];boundary=[];solutions={}
    for m in args.orders:
        sol=stationary_milne(m);solutions[m]=sol;r=moment_record(sol);rows.append(r)
        for i in np.flatnonzero(sol.nodes<0)[::-1]:
            v=-sol.nodes[i];h=sol.boundary[i]
            boundary.append({'m':m,'v':float(v),'h_M':float(h),'p_M':float(1.5*v*h),'weight':float(sol.weights[i])})
        print(f'm={m:3d} z={r["z_M"]:.10f} Bmom={r["B_from_moments"]:.10f}')
    write_rows(out/'milne_boundary_moments.csv',rows)
    write_rows(out/'milne_boundary_flux_distribution.csv',boundary)
    old_file=out/'milne_and_B_convergence.csv'
    old={}
    if old_file.exists():
        with old_file.open() as f:
            old={int(r['m']):r for r in csv.DictReader(f)}
    comparison=[]
    for r in rows:
        prior=old.get(int(r['m']),{})
        try: B_fit=float(prior.get('B','nan'))
        except ValueError: B_fit=float('nan')
        if np.isfinite(B_fit):
            comparison.append({'m':r['m'],'B_from_stationary_moments':r['B_from_moments'],
                               'B_from_archived_small_s_fit':B_fit,'difference':r['B_from_moments']-B_fit})
    if comparison: write_rows(out/'boundary_moment_B_comparison.csv',comparison)
    # Independent representation of stationary solution, as an algorithmic
    # regression, not an independent physical model.
    from milne_analysis import milne_length
    zcheck={str(m):abs(solutions[m].z-milne_length(m)) for m in [64,128] if m in solutions}
    finite=[r for r in rows]
    summary={'method':'reduced symmetric stationary pencil; analytic removal of null/Jordan sector',
             'maximum_M1_residual':max(abs(r['M1_residual']) for r in rows),
             'maximum_M2_residual':max(abs(r['M2_residual']) for r in rows),
             'maximum_integrated_remainder_residual':max(abs(r['I_residual']) for r in rows),
             'maximum_incoming_residual':max(r['incoming_residual'] for r in rows),
             'old_nodal_z_discrepancies':zcheck,
             'highest_order':rows[-1],
             'provenance':'Stationary calculations executed in this revision; small-s comparisons use archived values.',
             'qualification':'Finite-order values are not certified continuum enclosures.'}
    assert summary['maximum_M1_residual']<1e-9
    assert summary['maximum_M2_residual']<1e-9
    assert summary['maximum_integrated_remainder_residual']<1e-9
    assert all(r['B_lower_bound']<=r['B_from_moments']<=r['B_upper_bound'] and r['outgoing_min']>0 for r in rows)
    with (out/'boundary_moment_summary.json').open('w') as f:json.dump(summary,f,indent=2)
    # Tables are generated directly from these finite-order records.
    gen=ROOT/'generated';gen.mkdir(exist_ok=True)
    lines=[r'\begin{tabular}{rrrrr}',r'\toprule',
           r'$m$ & $z_M^{(m)}$ & $M_3^{(m)}$ & $B_{\rm mom}^{(m)}$ & $|M_1-2/3|$\\',r'\midrule']
    for r in rows:
        lines.append(f"{r['m']} & {r['z_M']:.9f} & {r['M3']:.9f} & {r['B_from_moments']:.9f} & ${abs(r['M1_residual']):.1e}$\\\\")
    lines += [r'\bottomrule',r'\end{tabular}']
    (gen/'boundary_moments_table.tex').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
