#!/usr/bin/env python3
"""Restore supplemental table alt text after regenerating numerical tables.

The table generators remain independent of the journal submission format. This
idempotent postprocessor adds the required descriptions to their LaTeX output.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'generated'
ALTS = {
    'table_conditioning.tex': 'The incoming matching matrices remain well conditioned, and invariant-subspace residuals remain small across the audited angular orders.',
    'table_solver_regression.tex': 'Equivalent spectral solver implementations agree closely at the audited transform parameters.',
    'table_reference_all_orders.tex': 'Span estimates are listed over 67 contour lengths at four angular orders, with extrapolated values and numerical sensitivity diagnostics.',
    'table_high_order_spot.tex': 'Withheld high-angular-order span estimates are close to the extrapolated reference values at representative contour lengths.',
    'table_fv_comparison.tex': 'Independently extrapolated finite-volume span estimates agree closely with the spectral reference at three contour lengths.',
    'table_fv_extended.tex': 'Extended finite-volume grid refinements at two contour lengths support the primary finite-volume extrapolations.',
    'table_path_validation.tex': 'Finite-step polygonal chain estimates are compared with spectral references; sampling uncertainties and step-size effects are reported separately.',
    'table_milne_B.tex': 'Sequences of angular-order estimates approach the reported Milne displacement and proposed next flexible-limit coefficient.',
    'table_B_sensitivity.tex': 'Changing fit windows, angular orders, and fitting assumptions produces a modest spread in the proposed flexible-limit coefficient.',
    'table_milne_threshold.tex': 'The calculated Milne displacement is stable over the range of thresholds that excludes the stationary near-zero sector.',
    'table_fit_metrics.tex': 'The rational correlation has small relative discrepancies from the spectral reference on calibration and withheld nodes.',
    'table_sign_summary.tex': 'Coefficient-sign audits establish the signs of derivative polynomials for the printed rational span correlation.',
    'table_heat_path_validation.tex': 'At four contour lengths, heat-kernel path estimates and paired grid differences are shown with sampling standard errors.',
}

def close_brace(text, pos):
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == '{' and (i == 0 or text[i-1] != '\\'):
            depth += 1
        elif text[i] == '}' and (i == 0 or text[i-1] != '\\'):
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('Unbalanced caption')

for name, alt in ALTS.items():
    path = ROOT / name
    if not path.exists():
        continue
    text = path.read_text()
    if r'\textbf{Alt text:}' in text:
        continue
    cap = text.index(r'\caption{')
    end = close_brace(text, cap + len(r'\caption')) + 1
    if name == 'table_reference_all_orders.tex':
        end = text.index('\n', end) + 1
        insert = r'\multicolumn{7}{p{0.90\linewidth}}{\textbf{Alt text:} ' + alt + r'}\\' + '\n'
    else:
        insert = '\n' + r'\par\raggedright\textbf{Alt text:} ' + alt + r'\par\centering' + '\n'
    path.write_text(text[:end] + insert + text[end:])
    print(f'Added supplemental alt text: {name}')
