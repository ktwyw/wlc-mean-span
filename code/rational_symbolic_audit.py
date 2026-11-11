#!/usr/bin/env python3
"""Exact sign audits of the printed and full-precision rational correlations.

The primary coefficient files certify the eight-decimal main-text version.
Additional files certify the ten-decimal SI version and the full-precision
JSON version separately.  Decimal coefficients are exact SymPy rationals
within each specified version; no finite-difference scan is used as a proof.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'


def rational_expressions(coeff:dict[str,float], decimals:int|None=8) -> tuple[sp.Symbol,dict[str,sp.Expr]]:
    w=sp.symbols('w',positive=True)
    def number(key:str)->sp.Rational:
        value=coeff[key]
        return sp.Rational(str(value) if decimals is None else f'{value:.{decimals}f}')
    a,b,c,d,e,f=[number(k) for k in ('a','b','c','d','e','f')]
    numerator=1+a*w+b*w**2+c*w**3
    denominator=1+a*w+d*w**2+e*w**3+f*w**4
    F=sp.Rational(1,2)*w**2*numerator/denominator
    first=sp.diff(F,w)/(2*w)
    return w,{
        'F_prime':sp.cancel(first),
        'minus_F_second':sp.cancel(-sp.diff(first,w)/(2*w)),
        'minus_d_F_over_lambda':sp.cancel(-sp.diff(F/w**2,w)/(2*w)),
        'rod_gap':sp.cancel(w**2/2-F),
    }


def audit_variant(coeff:dict[str,float],decimals:int|None)->tuple[list[dict],list[dict]]:
    w,expressions=rational_expressions(coeff,decimals)
    rows=[];summary=[]
    for name,expr in expressions.items():
        num,den=sp.fraction(expr)
        pnum,pden=sp.Poly(num,w),sp.Poly(den,w)
        signs=[]
        for power,co in enumerate(reversed(pnum.all_coeffs())):
            if co!=0:
                signs.append(int(sp.sign(co)))
                rows.append({'quantity':name,'power_w':power,
                             'coefficient_exact':str(co),'coefficient_float':float(co)})
        den_signs=[int(sp.sign(c)) for c in pden.all_coeffs() if c!=0]
        if min(signs)!=1 or min(den_signs)!=1:
            raise AssertionError(f'{name}: loss of positive numerator/denominator coefficients')
        summary.append({'quantity':name,'degree':pnum.degree(),'nonzero_coefficients':len(signs),
                        'minimum_sign':min(signs),'maximum_sign':max(signs),
                        'denominator':str(sp.factor(den))})
    return rows,summary


def write_csv(path:Path,rows:list[dict])->None:
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main()->None:
    coeff=json.loads((DATA/'rational_fit_coefficients.json').read_text())['central']
    counts={}
    for suffix,decimals in (('',8),('_si',10),('_full_precision',None)):
        rows,summary=audit_variant(coeff,decimals)
        write_csv(DATA/f'rational_sign_polynomial_coefficients{suffix}.csv',rows)
        write_csv(DATA/f'rational_sign_polynomial_summary{suffix}.csv',summary)
        counts[suffix or 'main']={'decimal_places':decimals,'positive_quantities':len(summary)}
    print(json.dumps({'status':'PASS','variants':counts},indent=2))


if __name__=='__main__':
    main()
