#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:32:43 2023

@author: t_karmakar
"""

#Cost function calculations for 1d equation

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
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)
rcParams['figure.figsize']=6,6
from jaxopt import OptaxSolver
import optax


dv=4
da=2
kmax=6
s1=43
s2=27
'''Padding Lengths'''
s1p=6
s2p=6
'''---------------'''
Nvars=1+dv*(da+2)+4*s1*s2*dv+4*kmax*kmax*dv*dv
#NT=20
dx=0.01
xi=-2
xf=2
ti=0
tf=1
xs=np.linspace(xi,xf,s1-s1p)
ts=np.linspace(ti,tf,s2-s2p)
dx=xs[1]-xs[0]
dt=ts[1]-ts[0]
ts=np.linspace(ti,ti+dt*(s2-1),s2)
xs=np.linspace(xi,xi+dx*(s1-1),s1)

xv,tv=np.meshgrid(xs,ts,indexing='ij')
i_seed=np.random.randint(0,1000)
i_seed=483
ks=np.arange(0,kmax)
params=init_params(kmax,kmax,da,dv,random.PRNGKey(i_seed))
#params=params3
avalue=0.2
avs=np.linspace(0.01,1,5)
mu=0.5
sigma=0.08
sigmas=np.linspace(0.08,1,300)
alist=[]
padmatrix=np.zeros((s1,s2))
fpadmat=np.zeros((s1,s2,dv,dv))
padmatrix[:-s1p,:-s2p]=np.ones((s1-s1p,s2-s2p))
fpadmat[:kmax,:kmax,:,:]=np.ones((kmax,kmax,dv,dv))
for av in avs:
    for sigmav in sigmas:
        #print(sigmav)
        atmp=np.zeros((s1,s2,da))
        atmp[:-s1p,:-s2p,0]=aMatrix(av,(s1,s2))[:-s1p,:-s2p]
        atmp[:-s1p,:-s2p,1]=gauss(xv,mu,sigmav)[:-s1p,:-s2p]
        alist.append(atmp)

alist=jnp.array(alist)

tstart=time.time()
cost=TotalCost(params,alist,dx,dt,padmatrix)

tfinish=time.time()
trun=tfinish-tstart

print(trun,cost)
l0=1e-3
step_size=l0
num_epochs=1
fig, axs = plt.subplots(2,1,sharex='all')
paramsA=params_toAdam(params)
tempindex=250

opt=optax.adam(step_size)
stime=time.time()
solver=OptaxSolver(opt=opt, fun=TotalCostAdam,maxiter=num_epochs,tol=1e-4)
res=solver.run(paramsA,alist,dx,dt,padmatrix)
ftime=time.time()
print("Adam time :", ftime-stime)
paramsf,state=res

u=OutputNNAdam(paramsf,alist[0])
u1=OutputNNAdam(paramsf,alist[tempindex])
'''
costlist=np.zeros(num_epochs)
Full_stime=time.time()
for epoch in  range(num_epochs):
    stime=time.time()
    if(epoch<300):
      step_size=l0/10
    #elif(epoch>800):
    #  step_size=l0*10.0
    #step_size=(10*(epoch//500)+1)*l0
    #if (epoch>600):
     #  step_size=10.*l0
    #elif (epoch>400):
     #  step_size=1e4*l0
    params=update(params,alist,dx,dt,step_size,padmatrix)
    epoch_time=time.time()-stime
    #train_acc = TotalCost(params,alist,dx,dt,padmatrix)
    #costlist[epoch]=train_acc
    #print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    #print("Total cost {}".format(train_acc))
print("Manual time : ", time.time()-Full_stime)
u=OutputNN(params,alist[0])
u1=OutputNN(params,alist[tempindex])
'''
lwd=3

axs[0].plot(xs[:-s1p],u[:-s1p,0],'r',label='$u(t=0)$',linewidth=lwd)
axs[0].plot(xs[:-s1p],u[:-s1p,-s2p-1],'b--',label='$u(t=1)$',linewidth=lwd)
axs[0].plot(xs[:-s1p],alist[0][:-s1p,0,1],'g',label='Initial',linewidth=lwd)




axs[1].plot(xs[:-s1p],u1[:-s1p,0],'r',label='$u(t=0)$',linewidth=lwd)
axs[1].plot(xs[:-s1p],u1[:-s1p,-s2p-1],'b--',label='$u(t=1)$',linewidth=lwd)
axs[1].plot(xs[:-s1p],alist[tempindex][:-s1p,0,1],'g',label='Initial',linewidth=lwd)

axs[1].tick_params(labelsize=18)
axs[0].tick_params(labelsize=18)
axs[1].set_xlabel('$x$',fontsize=20)
axs[0].set_ylabel('$u$',fontsize=20)
axs[0].yaxis.set_label_coords(-.15, -.15)
axs[0].legend(loc=1,fontsize=15)
plt.subplots_adjust(wspace=0.05, hspace=0.1)
fig.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/NTTResearch/Plots/Transport.png',bbox_inches='tight')
print(i_seed)
