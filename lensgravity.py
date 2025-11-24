#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy",
#                 "matplotlib"]
# ///

""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia,
    C Bordeianu, Univ Bucharest, 2024.
    Please respect copyright & acknowledge our work."""

# LensGravity.py:  Deflection of light by the sun wi Matplotlib

import numpy as np
import matplotlib.pyplot as plt

y = np.zeros((2),float)
ph = np.zeros((181),float)                          # Time
yy = np.zeros((181),float)
xx = np.zeros((181),float)
rx = np.zeros((181),float)
ry = np.zeros((181),float)
Gsun = 4477.1                                    # Meters,  sum massxG
GM = 28.*Gsun
y[0] = 1.e-6; y[1] = 1e-6                       # Initial condition u=1/r

def f(t,y):                                     # RHS, can modify
    rhs = np.zeros((2),float)
    rhs[0] = y[1]
    rhs[1] = 3*GM*(y[0]**2)-y[0]
    return rhs
def rk4Algor(t, h, N, y, f):                    # Do not modify
    k1=np.zeros(N); k2=np.zeros(N); k3=np.zeros(N); k4=np.zeros(N)
    k1 = h*f(t,y)
    k2 = h*f(t+h/2.,y+k1/2.)
    k3= h*f(t+h/2.,y+k2/2.)
    k4= h*f(t+h,y+k3)
    y = y+(k1+2*(k2+k3)+k4)/6.
    return y

f(0,y)                                  # Initial conditions
dphi = np.pi/180.                      # 180 phi values
i = 0                       # counter
for phi  in np.arange(0,np.pi+dphi,dphi):
    ph[i] = phi
    y = rk4Algor(phi,dphi,2,y,f)    # Call rk4
    xx[i] = np.cos(phi)/y[0]/1000000 # Scale for graph
    yy[i] = np.sin(phi)/y[0]/1000000
    i = i + 1
m = (yy[180] - yy[165])/(xx[180]-xx[165])       # Slope
b = yy[180]-m*xx[180]                           # Intercept
j = 0
for phi  in np.arange(0,np.pi+dphi,dphi):
    ry[j] = m*xx[j]+b                 # Straight line eqtn
    j = j+1
plt.figure(figsize=(12,6))
plt.plot(xx,yy)     		# Light trajectory
plt.plot(xx,-yy)    		# Symmetric for negative y
plt.plot(0,0,'ro')          # Mass at origin
plt.plot(0.98,0,'bo')       # Source
plt.plot(0.98,1.91,'go')    # Position source seen from O
plt.plot(0.98,-1.91,'go')
plt.text(1,0,'S')
plt.text(-1.04,-0.02,'O')
plt.text(1.02, 1.91,"S' ")
plt.text(1.02,-2,"S''")
plt.plot([0],[3.])   # Invisible poin
plt.plot([0],[-3.])  # Invisible point at -y
plt.plot(xx,ry)      # Upper straight
plt.plot(xx,-ry)     # Lower straight line
plt.xlabel('x')
plt.ylabel('y')
plt.show()
