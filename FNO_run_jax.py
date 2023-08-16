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
rc('text',usetex=True)
from jaxopt import OptaxSolver
import optax


dv=5
da=2
kmax=12
s1=33
s2=27
'''Padding Lengths'''
s1p=6
s2p=6
'''---------------''' 
Nvars=1+dv*(da+2)+4*s1*s2*dv+4*kmax*kmax*dv*dv
#NT=20
dx=0.01
xs=np.linspace(-2,2,s1)
ts=np.linspace(0,1,s2-s2p)
dx=xs[1]-xs[0]
dt=ts[1]-ts[0]
ts=np.linspace(0,dt*(s2-1),s2)

xv,tv=np.meshgrid(xs,ts,indexing='ij')
i_seed=np.random.randint(0,1000)
i_seed=628
ks=np.arange(0,kmax)
params=init_params(s1,s2,kmax,kmax,da,dv,random.PRNGKey(i_seed))
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
l0=1e-4
step_size=l0
num_epochs=200

'''
opt=optax.adam(step_size)
stime=time.time()
solver=OptaxSolver(opt=opt, fun=TotalCost,maxiter=num_epochs)
res=solver.run(params,alist,dx,dt,padmatrix)
ftime=time.time()
print("Adam time :", ftime-stime)
paramsf,state=res

u=OutputNN(paramsf,alist[0])
plt.plot(xs[:-s1p],u[:-s1p,0],'k')
'''
costlist=np.zeros(num_epochs)
Full_stime=time.time()
for epoch in  range(num_epochs):
    stime=time.time()
    #if (epoch>100):
     #   step_size=10.*l0
    params=update(params,alist,dx,dt,step_size,padmatrix)
    epoch_time=time.time()-stime
    train_acc = TotalCost(params,alist,dx,dt,padmatrix)
    costlist[epoch]=train_acc
    #test_acc = accuracy(params, test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Total cost {}".format(train_acc))
    #print("Test set accuracy {}".format(test_acc))
print("Manual time : ", time.time()-Full_stime) 
u=OutputNN(params,alist[0])
plt.plot(xs[:-s1p],u[:-s1p,0])
print(i_seed)
