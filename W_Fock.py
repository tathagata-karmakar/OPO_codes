#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 21:36:24 2023

@author: t_karmakar
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from scipy.integrate import simps as intg
#from google.colab import files
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)


def f(m,x,f1,f2):
    #arg = 2*x**2
    #prefactor = ((-1)**m/np.pi)
    if m==0:
        return np.ones(len(x))
    elif m ==1:
        return (1-arg)
    else:
        return ((2*m-1-arg)*f1-(m-1)*f2)*(1.0/m)

def Hermite(m,x,f1):
    #arg = 2*x**2
    #prefactor = ((-1)**m/np.pi)
    return 2*x*f1 - np.gradient(f1,x)
mode = 100
Nx = 1000
ls = np.zeros((mode,Nx))
ws = np.zeros((mode,Nx))
hs = np.zeros((mode,Nx))
psis = np.zeros((mode,Nx))
xs = np.linspace(0,5,Nx)
arg = 2*xs**2
prefac = np.exp(-xs**2)
ls[0,:] = 1
ls[1,:] = (1-arg)
for n in range(2,mode):
    ls[n,:] = f(n, arg, ls[n-1], ls[n-2])
for n in range(mode):
    ws[n] = ((-1)**n)*prefac*ls[n]/np.pi
    
    
expfac = np.exp(-xs**2/2)/np.pi**0.25
hs[0,:] = np.ones(Nx)
for n in range(1,mode):
    hs[n] = Hermite(n,xs,hs[n-1])
    
for n in range(mode):
    psis[n] = expfac*hs[n]/np.sqrt((2.**n)*math.factorial(n))

plt.plot(xs, np.exp(-(xs-np.sqrt(2)*0.5*5)**2)/np.sqrt(np.pi))