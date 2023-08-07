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
from FNO_structure_jax import *
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('text',usetex=True)


dv=64
da=1
kmax=12
#NT=20
dx=0.01
xs=np.linspace(0.01,0.99,kmax)
ts=np.linspace(0,1,kmax)
Lx=xs[1]-xs[0]
ks=np.arange(0,kmax)

#Nvars=4*(dv**2)*(1+kmax)+dv*(da+2)+1
'''
tstart=time.time()
#u=OutputNN(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs,nuval,da,kmax,dv)
lmbds=np.linspace(0.1,0.9,20)
u=TotalCost(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs,nuval,da,kmax,dv,dx,lmbds)
tfinish=time.time()
trun=tfinish-tstart

print(trun,u)
'''
