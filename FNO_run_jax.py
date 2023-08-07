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


dv=16
da=1
kmax=21
s1=kmax
s2=kmax+1
#NT=20
dx=0.01
xs=np.linspace(0.01,0.99,s1)
ts=np.linspace(0,1,s2)
dx=xs[1]-xs[0]
dt=ts[1]-ts[0]
xv,tv=np.meshgrid(xs,ts,indexing='ij')

ks=np.arange(0,kmax)
params=init_params(s1,s2,da,dv,random.PRNGKey(9))
avalue=0.2
avs=np.linspace(0.2,0.8,9)
mu=0.5
sigma=0.08
alist=[]
for av in avs:    
    atmp=np.zeros((s1,s2,da))
    atmp[:,:,0]=aMatrix(av,(s1,s2))
    #atmp[:,:,1]=gauss(xv,mu,sigma)
    alist.append(atmp)
    


tstart=time.time()
cost=TotalCost(params, alist, xs,ts,dx,dt)
tfinish=time.time()
trun=tfinish-tstart

print(trun,cost)

step_size=0.1
num_epochs=300

for epoch in  range(num_epochs):
    stime=time.time()
    params=update(params,alist,xs,ts,dx,dt,step_size)
    epoch_time=time.time()-stime
    train_acc = TotalCost(params,alist,xs,ts,dx,dt)
    #test_acc = accuracy(params, test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Total cost {}".format(train_acc))
    #print("Test set accuracy {}".format(test_acc))

u=OutputNN(params,alist[0])

plt.plot(xs,u[:,-1])
