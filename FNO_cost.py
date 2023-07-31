#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:32:43 2023

@author: t_karmakar
"""

#Cost function calculations for 1d equation

import os
import qutip as qt
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.integrate import simps as intg
from matplotlib import rc
from pylab import rcParams
from FNO_structure import *
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('text',usetex=True)

def costF(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1,nu1,da1,kmax1,dv1,dx1,al1): #Cost function calculations
    u=OutputNN(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1,nu1,da1,kmax1,dv1)
    u1=OutputNN(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1+dx1,nu1,da1,kmax1,dv1)
    du=(u1-u)/dx1
    return intg(du**2,xs1)+al1*u[0]**2

dv=64
da=1
kmax=16
NT=20
dx=0.01
xs=np.linspace(0.01,0.99,kmax)
ts=np.linspace(0,1,NT)
Lx=xs[1]-xs[0]
ks=np.arange(0,kmax)
Nvars=4*(dv**2)*(1+kmax)+dv*(da+2)+1
nuval=0.001
W0=np.random.rand(dv,da)
b0=np.random.rand(dv)
W1=np.random.rand(dv,dv)
kappa1=np.random.rand(dv,dv,kmax)
W2=np.random.rand(dv,dv)
kappa2=np.random.rand(dv,dv,kmax)
W3=np.random.rand(dv,dv)
kappa3=np.random.rand(dv,dv,kmax)
W4=np.random.rand(dv,dv)
kappa4=np.random.rand(dv,dv,kmax)
Wf=np.random.rand(1,dv)
bf=np.random.rand(1)

tstart=time.time()
#u=OutputNN(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs,nuval,da,kmax,dv)
u=costF(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs,nuval,da,kmax,dv,dx,0.1)
tfinish=time.time()
trun=tfinish-tstart

print(trun)

