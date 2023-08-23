#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 21:13:51 2023

@author: t_karmakar
"""

import os
#import qutip as qt
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.integrate import simps as intg
from matplotlib import rc
from pylab import rcParams
from FNO_structure_jax import *
from FNO_structure3D import *
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)
#rcParams['figure.figsize']=6,6
from jaxopt import OptaxSolver
import optax
from matplotlib import colors
import pickle

fname='/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/NTTResearch/OPO_codes/data1'
npzfile=np.load(fname+'.npz')
with open(fname+'.pickle','rb') as file:
    params=pickle.load(file)
    
dv=int(npzfile['dv'])
s1=int(npzfile['s1'])
s2=int(npzfile['s2'])
s3=int(npzfile['s3'])
s1p=int(npzfile['s1p'])
s2p=int(npzfile['s2p'])
s3p=int(npzfile['s3p'])
xs=npzfile['xs']
ps=npzfile['ps']
ts=npzfile['ts']
xv,pv,tv=np.meshgrid(xs,ps,ts,indexing='ij')
alist=npzfile['alist']
tempindex=250
tind=5

u=OutputNN3D(params,alist[tempindex])

fig, axs = plt.subplots(2,1,sharex='all')
lwd=3

maxu=np.max(u[s1p:-s1p,s2p:-s2p,tind])
minu=np.min(u[s1p:-s1p,s2p:-s2p,tind])
divnorm=colors.TwoSlopeNorm(vmin=minu, vcenter=(maxu+minu)/2, vmax=maxu)

im1=axs[0].contourf(xv[s1p:-s1p,s2p:-s2p,tind],pv[s1p:-s1p,s2p:-s2p,tind],u[s1p:-s1p,s2p:-s2p,tind],levels=8,cmap='RdBu', norm=divnorm)
cb1=plt.colorbar(im1,ax=axs[0])
cb1.ax.tick_params(labelsize=12)

axs[1].tick_params(labelsize=18)
axs[0].tick_params(labelsize=18)
axs[1].set_xlabel('$x$',fontsize=20)
axs[0].set_ylabel('$p$',fontsize=20)
axs[0].yaxis.set_label_coords(-.15, -.15)
#axs[0].legend(loc=1,fontsize=15)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
#fig.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/NTTResearch/Plots/Wigner3D.png',bbox_inches='tight')
