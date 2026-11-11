#!/usr/bin/env python3
"""Generate LaTeX tables directly from released CSV/JSON data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
G = ROOT / "generated"
G.mkdir(exist_ok=True)
ROW_END = r"\\" + "\n"


def read(name: str) -> list[dict[str, str]]:
    with (D / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


# Selected main-text table: the released full CSV is the sole numerical source.
selected_nodes = (0.01, 0.1, 1.0, 2.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
reference_by_node = {float(row["lambda"]): row for row in read("wlc_mean_span_reference.csv")}
selected_rows = [reference_by_node[node] for node in selected_nodes]
def rg_over_p(lam: float) -> float:
    """Exact root-mean-square radius of gyration of the WLC, in units of p."""
    if lam < 0.3:
        g = lam*lam*(1/12 + lam*(-1/60 + lam*(1/360 + lam*(-1/2520 + lam*(1/20160
            + lam*(-1/181440 + lam*(1/1814400 - lam/19958400)))))))
    else:
        import math
        g = lam/3 - 1 + 2/lam + 2*math.expm1(-lam)/lam**2
    return g**0.5


with (G / "table_selected_reference.tex").open("w") as handle:
    handle.write(r'''\begin{table}[ht]
\caption{Mean span from the spectral first-return solution at selected contour lengths. \(\epsilon_{\rm int}\) is the absolute internal numerical sensitivity of \(X/p\) (a diagnostic, not a confidence interval). The last column divides by the exact root-mean-square radius of gyration, Eq.~\eqref{eq:gyration}. All 67 nodes are tabulated in Table~S1 of the supplementary material.}
\label{tab:reference}
\centering
\begin{tabular}{r@{\hspace{1.6em}}r@{\hspace{1.6em}}r@{\hspace{1.6em}}r}
\toprule
{$L/p$} & {$X/p$} & {$\epsilon_{\rm int}$} & {$X/R_{\mathrm g}$}\\
\midrule
''')
    for row in selected_rows:
        lam = float(row["lambda"])
        fvalue = float(row["F_continuum"])
        epsilon = float(row["epsilon_internal_abs"])
        handle.write(
            f"{lam:g} & " + r"\num{" + f"{fvalue:.6g}" + "} & "
            + r"\num{" + f"{epsilon:.1e}" + "} & " + f"{fvalue/rg_over_p(lam):.4f}" + ROW_END
        )
    handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')
with (D / "selected_reference_values.csv").open("w", newline="") as handle:
    fields = ["lambda", "F_continuum", "epsilon_internal_abs", "epsilon_internal_rel"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in selected_rows)

# Complete reference table
rows = read("wlc_mean_span_reference.csv")
with (G / "table_reference_all_orders.tex").open("w") as handle:
    handle.write(r'''\begingroup
\scriptsize
\setlength{\tabcolsep}{2.0pt}
\renewcommand{\arraystretch}{0.96}
\setlength{\LTleft}{\fill}
\setlength{\LTright}{\fill}
\begin{longtable}{@{}rrrrrrr@{}}
\caption{Complete finite-order and continuum-extrapolated mean-span reference table.  The internal sensitivity is an empirical numerical diagnostic, not a confidence interval.}\label{tab:complete_reference}\\
\toprule
{$\lambda$}&{$F_{32}$}&{$F_{48}$}&{$F_{64}$}&{$F_{96}$}&{$F_\infty$}&{$\epsilon_{\rm int}$}\\
\midrule
\endfirsthead
\toprule
{$\lambda$}&{$F_{32}$}&{$F_{48}$}&{$F_{64}$}&{$F_{96}$}&{$F_\infty$}&{$\epsilon_{\rm int}$}\\
\midrule
\endhead
''')
    for row in rows:
        handle.write(
            f"{float(row['lambda']):.6g} & {float(row['F_m32']):.7g} & "
            f"{float(row['F_m48']):.7g} & {float(row['F_m64']):.7g} & "
            f"{float(row['F_m96']):.7g} & {float(row['F_continuum']):.7g} & "
            f"{float(row['epsilon_internal_abs']):.3e}" + ROW_END
        )
    handle.write(r'''\bottomrule
\end{longtable}
\endgroup
''')

# Finite-volume continuum comparison
rows = read("finite_volume_comparison.csv")
with (G / "table_fv_comparison.tex").open("w") as handle:
    handle.write(r'''\begin{table}[ht]
\centering
\caption{Independent finite-volume continuum extrapolation compared with the spectral reference.  The finite-volume sensitivity is the conservative sum of the spatial and angular extrapolation variations.}
\label{tab:fv_comparison}
\begin{tabular}{rrrrr}
\toprule
{$\lambda$}&{$F_{\rm spec}$}&{$F_{\rm FV}$}&{FV-spec (\%)}&{FV sens. (\%)}\\
\midrule
''')
    for row in rows:
        handle.write(
            f"{float(row['lambda']):.1f} & {float(row['F_spectral']):.9f} & "
            f"{float(row['F_continuum_quad']):.9f} & {float(row['relative_difference_pct']):.5f} & "
            f"{float(row['FV_combined_sensitivity_rel_pct']):.5f}" + ROW_END
        )
    handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Rational fit metrics
metrics = json.load((D / "rational_fit_metrics.json").open())
with (G / "table_fit_metrics.tex").open("w") as handle:
    handle.write(r'''\begin{table}[ht]
\centering
\caption{Error of the rod--coil asymptotically constrained rational correlation against the released spectral reference nodes.}
\label{tab:fit_metrics}
\begin{tabular}{lrrrr}
\toprule
Set & Nodes & Max abs. error (\%) & RMS error (\%) & $\lambda$ at max\\
\midrule
''')
    groups = [
        ("calibration", "Calibration"),
        ("holdout_interleaved", "Withheld, interleaved"),
        ("extrapolation_lambda_gt_5000", r"Withheld, $\lambda>5000$"),
        ("all", "All 67 nodes"),
        ("crossover_0p1_10", r"Crossover $0.1\le\lambda\le10$"),
    ]
    for key, label in groups:
        item = metrics[key]
        handle.write(
            f"{label} & {item['n']} & {item['max_abs_rel_error_pct']:.5f} & "
            f"{item['rms_rel_error_pct']:.5f} & {item['lambda_at_max']:.6g}" + ROW_END
        )
    handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Sign summary
rows = read("rational_sign_polynomial_summary.csv")
with (G / "table_sign_summary.tex").open("w") as handle:
    handle.write(r'''\begin{table}[ht]
\centering
\caption{Exact sign audit of numerator polynomials after setting $\lambda=w^2$.  Printed decimal fit parameters are treated as exact rational numbers.}
\label{tab:sign_summary}
\begin{tabular}{lrrc}
\toprule
Quantity represented as positive rational function & Degree & Nonzero coefficients & Sign\\
\midrule
''')
    labels = {
        "F_prime": r"$F_R'$",
        "minus_F_second": r"$-F_R''$",
        "minus_d_F_over_lambda": r"$-\mathrm d(F_R/\lambda)/\mathrm d\lambda$",
        "rod_gap": r"$\lambda/2-F_R$",
    }
    for row in rows:
        sign = "positive" if int(row["minimum_sign"]) > 0 else "mixed"
        handle.write(
            f"{labels[row['quantity']]} & {int(row['degree'])} & "
            f"{int(row['nonzero_coefficients'])} & {sign}" + ROW_END
        )
    handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Stable-subspace conditioning audit
rows = read("conditioning_audit_full.csv")
with (G / "table_conditioning.tex").open("w") as handle:
    handle.write(r'''\begin{table}[ht]
\centering
\caption{Conditioning of the incoming half-range matching matrix and invariant-subspace residuals over the audited right-half-plane transform samples.  The stable-space dimension equals $m/2$ in every audit.}
\label{tab:conditioning}
\begin{tabular}{rrrrr}
\toprule
$m$ & median condition & maximum condition & maximum residual & stable dimension\\
\midrule
''')
    for row in rows:
        handle.write(
            f"{int(float(row['m']))} & {float(row['condition_median']):.6f} & "
            f"{float(row['condition_max']):.6f} & {float(row['residual_max']):.2e} & "
            f"{int(float(row['stable_dimension_min']))}" + ROW_END
        )
    handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Seeded path validation table
rows = read("seeded_path_validation_regenerated.csv")
with (G / "table_path_validation.tex").open("w") as handle:
    handle.write(r'''\begin{table}[ht]
\centering
\caption{Seeded polygonal-chain consistency calculations.  Standard errors quantify sampling only; the nonmonotone step sequence does not establish a resolved continuum extrapolation.}
\label{tab:path_validation}
\begin{tabular}{rrrrrr}
\toprule
$\lambda$ & $\Delta\tau$ & paths & seed & $F_{\rm path}$ & SEM\\
\midrule
''')
    for row in rows:
        handle.write(
            f"{float(row['lambda']):.1f} & {float(row['delta_tau']):.4f} & "
            f"{int(row['paths']):,} & {int(row['seed'])} & {float(row['F_MC']):.9f} & "
            f"{float(row['SEM']):.3e}" + ROW_END
        )
    handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Milne and B convergence table
rows = read("milne_and_B_convergence.csv")
with (G / "table_milne_B.tex").open("w") as handle:
    handle.write(r'''\begin{table}[ht]
\centering
\caption{Angular-order sequences for the Milne displacement and the small-$s$ coefficient.  Blank entries indicate orders used only for the stationary Milne calculation.}
\label{tab:milne_B}
\begin{tabular}{rrrr}
\toprule
$m$ & $z_M(m)$ & $B(m)$ & polynomial-degree sensitivity\\
\midrule
''')
    for row in rows:
        b_text = "--" if row["B"].lower() == "nan" else f"{float(row['B']):.8f}"
        sens_text = "--" if row["B_fit_degree_sensitivity"].lower() == "nan" else f"{float(row['B_fit_degree_sensitivity']):.2e}"
        handle.write(f"{int(row['m'])} & {float(row['z_M']):.9f} & {b_text} & {sens_text}" + ROW_END)
    handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')


# Withheld high-angular-order spot checks
path = D / "high_order_spot_checks.csv"
if path.exists():
    rows = read("high_order_spot_checks.csv")
    with (G / "table_high_order_spot.tex").open("w") as handle:
        handle.write(r'''\begin{table}[ht]
\centering
\caption{High-angular-order spectral spot checks not used in the primary $m=32,48,64,96$ continuum fit.}
\label{tab:high_order_spot}
\begin{tabular}{rrrr}
\toprule
$\lambda$ & $F_\infty$ & $F_{128}$ & $(F_{128}/F_\infty-1)$ (\%)\\
\midrule
''')
        for row in rows:
            handle.write(
                f"{float(row['lambda']):.4g} & {float(row['F_continuum']):.10g} & "
                f"{float(row['F_m128']):.10g} & {float(row['relative_difference_m128_pct']):.6f}" + ROW_END
            )
        handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Extended finite-volume withheld-grid audit
path = D / "finite_volume_extended_comparison.csv"
if path.exists():
    rows = read("finite_volume_extended_comparison.csv")
    with (G / "table_fv_extended.tex").open("w") as handle:
        handle.write(r'''\begin{table}[ht]
\centering
\caption{Extended finite-volume spot audit using $N_y=1600$ and $N_u=64$ in addition to the primary tensor grid.}
\label{tab:fv_extended}
\begin{tabular}{rrrrr}
\toprule
$\lambda$ & $F_{\rm spec}$ & $F_{\rm FV,ext}$ & FV-spec (\%) & FV sensitivity (\%)\\
\midrule
''')
        for row in rows:
            handle.write(
                f"{float(row['lambda']):.1f} & {float(row['F_spectral']):.9f} & "
                f"{float(row['F_FV_extended']):.9f} & {float(row['relative_difference_pct']):.5f} & "
                f"{float(row['combined_sensitivity_rel_pct']):.5f}" + ROW_END
            )
        handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# B-sensitivity summary
path = D / "B_summary.json"
if path.exists():
    summary = json.load(path.open())
    with (G / "table_B_sensitivity.tex").open("w") as handle:
        handle.write(r'''\begin{table}[ht]
\centering
\caption{Audit of the numerical coefficient $B$.  The reported parenthetical range is conservative relative to the stable fit-window/order sweep.}
\label{tab:B_sensitivity}
\begin{tabular}{lr}
\toprule
Quantity & Value\\
\midrule
''')
        handle.write(f"Reported $B$ & {summary['reported_B']:.4f}" + ROW_END)
        handle.write(f"Reported half-range & {summary['reported_half_range']:.4f}" + ROW_END)
        handle.write(f"Stable-audit minimum & {summary['stable_audit_min']:.7f}" + ROW_END)
        handle.write(f"Stable-audit maximum & {summary['stable_audit_max']:.7f}" + ROW_END)
        handle.write(f"Stable-audit median & {summary['stable_audit_median']:.7f}" + ROW_END)
        handle.write(f"Audited combinations & {summary['stable_audit_count']}" + ROW_END)
        handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Milne zero-mode threshold audit
path = D / "milne_zero_mode_threshold.csv"
if path.exists():
    all_rows = read("milne_zero_mode_threshold.csv")
    rows = [r for r in all_rows if float(r['zero_threshold']) in (1e-5, 3e-5, 1e-4)]
    with (G / "table_milne_threshold.tex").open("w") as handle:
        handle.write(r'''\begin{table}[ht]
\centering
\caption{Stationary Milne zero-mode threshold audit.  Once the double near-zero sector is excluded, $z_M$ is threshold independent over the audited range.}
\label{tab:milne_threshold}
\begin{tabular}{rrrrr}
\toprule
$m$ & threshold & near-zero count & first retained decay rate $|\rho|$ & $z_M$\\
\midrule
''')
        for row in rows:
            z = row['z_M']
            z_text = '--' if z.lower() == 'nan' else f"{float(z):.9f}"
            handle.write(
                f"{int(row['m'])} & {float(row['zero_threshold']):.0e} & {int(row['near_zero_count'])} & "
                f"{float(row['first_retained_decay_abs']):.6g} & {z_text}" + ROW_END
            )
        handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Equivalent-solver regression audit summary
path = D / "solver_regression_audit.csv"
if path.exists():
    rows = read("solver_regression_audit.csv")
    eig_schur = max(float(r['rel_eig_schur']) for r in rows)
    real_rows = [r for r in rows if r['sample_type'] == 'positive_real']
    eig_sym = max(float(r['rel_eig_symmetric']) for r in real_rows)
    with (G / "table_solver_regression.tex").open("w") as handle:
        handle.write(r'''\begin{table}[ht]
\centering
\caption{Regression agreement among equivalent finite-order solver formulations over the audited transform samples.}
\label{tab:solver_regression}
\begin{tabular}{lr}
\toprule
Comparison & maximum relative discrepancy\\
\midrule
''')
        handle.write(f"Nodal eigensolve vs ordered Schur & {eig_schur:.3e}" + ROW_END)
        handle.write(f"Nodal eigensolve vs symmetric pencil (real $s>0$) & {eig_sym:.3e}" + ROW_END)
        handle.write(r'''\bottomrule
\end{tabular}
\end{table}
''')

# Coefficient displays are generated from the same JSON used in the figures.
central = json.load((D / "rational_fit_coefficients.json").open())["central"]
for name, decimals, suppress_first in (("main", 8, True), ("si", 10, False)):
    v = {k: f"{central[k]:.{decimals}f}" for k in ("a","b","c","d","e","f")}
    lines = [r"\begin{align}"]
    line1 = f" a_R&={v['a']}, & b_R&={v['b']}, & c_R&={v['c']},"
    if suppress_first: line1 += r"\nonumber"
    lines.append(line1 + r"\\")
    lines.append(f" d_R&={v['d']}, & e_R&={v['e']}, & f_R&={v['f']}.")
    if name == "main": lines.append(r" \label{eq:rational_coeffs}")
    lines.append(r"\end{align}")
    (G / f"rational_coefficients_{name}.tex").write_text("\n".join(lines)+"\n")



# Compact full reference table for the supplementary material (three column blocks).
with (G / "table_full_reference.tex").open("w") as handle:
    handle.write(r"""\begin{table}[ht]
\caption{Continuum-extrapolated spectral mean span $F=X/p$ at all 67 nodes and its internal absolute sensitivity $\epsilon_{\rm int}$ (a numerical diagnostic, not a confidence interval).}
\label{tab:full}
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrr@{\hspace{10pt}}rrr@{\hspace{10pt}}rrr}
\toprule
{$\lambda$} & {$F$} & {$\epsilon_{\rm int}$} & {$\lambda$} & {$F$} & {$\epsilon_{\rm int}$} & {$\lambda$} & {$F$} & {$\epsilon_{\rm int}$}\\
\midrule
""")
    allrows = sorted(read("wlc_mean_span_reference.csv"), key=lambda r: float(r["lambda"]))
    n = len(allrows); per = (n + 2) // 3
    def cell(r):
        return f"{float(r['lambda']):g} & " + r"\num{" + f"{float(r['F_continuum']):#.7g}" + "} & " + r"\num{" + f"{float(r['epsilon_internal_abs']):.1e}" + "}"
    for i in range(per):
        parts = []
        for k in range(3):
            j = i + k * per
            parts.append(cell(allrows[j]) if j < n else "&&")
        handle.write(" & ".join(parts) + ROW_END)
    handle.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
