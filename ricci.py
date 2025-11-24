#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy",
#                 "sympy"]
# ///

from sympy import *
import numpy as np


""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia,
    C Bordeianu, Univ Bucharest, 2024.
    Please respect copyright & acknowledge our work."""

# Ricci.py: Riemann & Ricci tensors, Ricci scalar, uses Sympy

from sympy import * #symbolic python
import numpy as np

t,r,th, fi, rg =  symbols('t r th fi rg')        # Schwarzchild metric
print("contravariant")                                 # Upper indices

# Inverse matrix
gT = Matrix([[1/(-1 + rg/r),0,0,0], [0, 1-rg/r, 0,0],
		[0, 0, r**(-2), 0],[0, 0, 0, 1/(r**2*sin(th)**2)]])
# 4-D array for alpha, beta, mu, nu
Ri  =  [[[[[]for n in range(4)] for a in range(4)] for b in range(4)] for c in range(4)]
RT = [[[] for m in range(4)]for p in range(4)]	           # Ricci tensor

# Christoffel symbols, upper t, r, theta, and phi
Cht = Matrix([[0, 0.5*rg/(r*(r - rg)), 0, 0],
		[0.5*rg/(r*(r - rg)), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
Chr = Matrix([[0.5*rg*(r - rg)/r**3,0,0,0], [0, -0.5*rg/(r*(r - rg)),0,0],
        [0,0, -1.0*r + 1.0*rg, 0], [0,0,0, (-1.0*r + rg)*sin(th)**2]])
Chth = Matrix([[0, 0, 0, 0], [0, 0, 1.0/r, 0], [0, 1.0/r, 0, 0],
             [0, 0, 0, -0.5*sin(2*th)]])
Chfi = Matrix([[0, 0, 0, 0], [0, 0, 0, 1.0/r], [0, 0, 0, 1.0/tan(th)],
             [0, 1.0/r, 1.0/tan(th), 0]])
for alpha in range(0,4):                   # Upper index in Christoffel
   if alpha == 0:   Chalp = Cht
   elif alpha == 1: Chalp = Chr
   elif alpha == 2: Chalp = Chth
   else:  Chalp = Chfi
   for be in range(0,4):                   # Beta
    for mu in range(0,4):
       if mu == 0:      der2 = t           # Derivative
       elif mu == 1:    der2 = r
       elif mu == 2:    der2 = th
       elif mu == 3:    der2 = fi
       for nu in range(0,4):
         if nu == 0:   der1 = t            # Other derivative
         elif nu == 1: der1 = r
         elif nu == 2: der1 = th
         elif nu == 3: der1 = fi
         a1 = diff(Chalp[be,nu],der2)      # Christoffel symbol
         a2 = diff(Chalp[be,mu],der1)      # And derivative
         sump = 0
         sumn = 0
         for gam in [t,r,th,fi]:
            if gam == t:
               Chgam = Cht
               gama = 0
            elif gam == r:
               Chgam = Chr
               gama = 1
            elif gam ==  th:
              Chgam = Chth
              gama = 2
            elif gam == fi:
              Chgam = Chfi
              gama = 3
            sump = sump+Chalp[mu,gama]*Chgam[be,nu]
            sumn = sumn+Chalp[nu,gama]*Chgam[be,mu]
         R = simplify(a1-a2+sump-sumn) # Riemann tensor
         if R == 0:  # Print nonzero components
              Ri[alpha][be][mu][nu] = 0
         else :
           Ri[alpha][be][mu][nu] = R
           print("Ri[",alpha,"][",be,"][",mu,"][",nu,"] = ", Ri[alpha][be][mu][nu])
print("\n")
print("Ricci Tensor\n")
for ro in range(0,4):
     for de in range (0,4):
         sum = 0
         for alp in range (0,4):
             sum = sum+Ri[alp][ro][alp][de]
         RT[ro][de] = simplify(sum)
         print("RT[",ro,"][",de,"] = ",RT[ro][de])    # Ricci's tensor
sumR = 0  #  Ricci Scalar
for be in range(0,4):
    for nu in range(0,4): sumR = sumR+gT[be,nu]*RT[be][nu]
    print(sumR)
RS = (sumR)
print("RS",RS)                                        # Ricci Scalar R
