#!/usr/bin/env python3
"""Exact moments used by the samplers and audits.

radial_moments: second and fourth moments of the end-to-end distance of the 3D wormlike
chain, evaluated without short-chain cancellation.
endpoint_moments_ode: the same moments from an independent 25-state moment equation.
clock_variance: variance of the martingale clock Q_t = int_0^t (1 - U_s^2) ds
(supplementary material).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import expm

def radial_moments(lam):
    """E[(R/p)^2], E[(R/p)^4], avoiding short-lambda cancellation."""
    x=np.asarray(lam,dtype=float)
    if np.any(x<=0) or np.any(~np.isfinite(x)):raise ValueError('lambda must be finite and positive')
    r2=np.empty_like(x);r4=np.empty_like(x); small=x<1.
    z=x[small]
    a=np.array([2.*(-1.)**k/math.factorial(k+2) for k in range(20)])
    # Coefficients of x^(k+4) in the exact fourth-moment formula.
    b=[]
    for k in range(20):
        n=k+4
        b.append((4/27)*(-54*((-1.)**(n-1)/math.factorial(n-1)+4*(-1.)**n/math.factorial(n))+2*(-3.)**n/math.factorial(n)))
    r2[small]=z*z*np.polynomial.polynomial.polyval(z,a)
    r4[small]=z**4*np.polynomial.polynomial.polyval(z,b)
    z=x[~small];r2[~small]=2*(z+np.expm1(-z))
    r4[~small]=(4/27)*(45*z*z-156*z+214-54*(z+4)*np.exp(-z)+2*np.exp(-3*z))
    if np.any(r2<=0) or np.any(r4<=0):raise ArithmeticError('nonpositive moment')
    return r2,r4

def endpoint_moments_ode(lam):
    """Independent 25-state moment equation for E[Z^n P_l(U)]."""
    size=5;A=np.zeros((size*size,size*size))
    index=lambda n,l:n*size+l
    for n in range(size):
        for ell in range(size):
            A[index(n,ell),index(n,ell)]=-ell*(ell+1)/2
            if n>0 and ell<size-1:A[index(n,ell),index(n-1,ell+1)]+=n*(ell+1)/(2*ell+1)
            if n>0 and ell>0:A[index(n,ell),index(n-1,ell-1)]+=n*ell/(2*ell+1)
    z=expm(lam*A)[:,0]
    return 3*z[index(2,0)],5*z[index(4,0)]

def clock_variance(lam):
    t=np.asarray(lam,dtype=float)
    # expm1 avoids cancellation in 3t - 1 + exp(-3t).
    return 8*(3*t+np.expm1(-3*t))/405
