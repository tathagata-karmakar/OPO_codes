#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:18:34 2023

@author: t_karmakar
"""

import os
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.integrate import simps as intg
from matplotlib import rc
from pylab import rcParams
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('text',usetex=True)


def CNapb_tmp(Natmp1,pb1):
    Deltat1=150
    gtildet1=1
    w1=0.25
    return np.absolute(np.exp(-1j*Deltat1*Natmp1)*np.exp(-(pb1-gtildet1*(Natmp1+0.5))**2/(4*w1**2))/(((2*np.pi)**0.25)*(w1**0.5)))**2
def g(p,pl,ph):
    return ((p>=pl)**2)*(1.0/(ph-pl))*((p<=ph)**2)

pl,ph=-5,5
pb1=np.linspace(pl,ph,500)
c=20
Na=1

vals=np.zeros(40000)
n=0
while n<len(vals):
    x=np.random.uniform(pl,ph)
    u=np.random.uniform(0,1)
    if (u<=CNapb_tmp(Na,x)/(20*g(x,pl,ph))):
        vals[n]=x
        n+=1


plt.plot(pb1,CNapb_tmp(Na,pb1))
plt.plot(pb1,c*g(pb1,pl,ph))
plt.hist(vals,bins=100,density=True)