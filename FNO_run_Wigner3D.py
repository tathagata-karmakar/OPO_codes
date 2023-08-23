#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:16:10 2023

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


dv=4
da=1
kmax=6
kmax1=kmax
kmax2=kmax
kmax3=kmax
s1=29
s2=29
s3=21
'''Padding Lengths in each direction'''
'''Padding is symmetric wrt x and p'''
s1p=3
s2p=3
s3p=4
'''---------------'''
Nvars=1+dv*(da+2)+4*s1*s2*s3*dv+4*kmax*kmax*kmax*dv*dv
#NT=20
dx=0.01
xi=-10
xf=10
pi=-10
pf=10
ti=0
tf=1
xs=np.linspace(xi,xf,s1-2*s1p)
ps=np.linspace(pi,pf,s2-2*s2p)
ts=np.linspace(ti,tf,s3-s3p)
dx=xs[1]-xs[0]
dp=ps[1]-ps[0]
dt=ts[1]-ts[0]

xs=np.linspace(xi-dx*(s1p),xf+dx*(s1p),s1)
ps=np.linspace(pi-dp*(s2p),pf+dp*(s2p),s2)
ts=np.linspace(ti,ti+dt*(s3-1),s3)

xv,pv,tv=np.meshgrid(xs,ps,ts,indexing='ij')

i_seed=np.random.randint(0,1000)
i_seed=483
ks=np.arange(0,kmax)
params=init_params3D(kmax,kmax,kmax,da,dv,random.PRNGKey(i_seed))
#params=params3

alrs=np.linspace(-6,6,30)
alis=np.linspace(-6,6,30)
alist=[]
padmatrix=np.zeros((s1,s2,s3))
#fpadmat=np.zeros((s1,s2,dv,dv))
padmatrix[s1p:-s1p,s2p:-s2p,:-s3p]=np.ones((s1-2*s1p,s2-2*s2p,s3-s3p))


xv1=xv*np.cos(tv)-pv*np.sin(tv)
pv1=pv*np.cos(tv)+xv*np.sin(tv)
for alr in alrs:
    for ali in alis:
        #print(sigmav)
        atmp=np.zeros((s1,s2,s3,da))
        atmp[s1p:-s1p,s2p:-s2p,:-s3p,0]=GWigner(xv1-alr, pv1-ali)[s1p:-s1p,s2p:-s2p,:-s3p]
        #atmp[:-s1p,:-s2p,1]=gauss(xv,mu,sigmav)[:-s1p,:-s2p]
        alist.append(atmp)

alist=jnp.array(alist)

tstart=time.time()
cost=TotalCost3D(params,alist,dx,dp,dt,xv,pv,padmatrix)

tfinish=time.time()
trun=tfinish-tstart

print(trun,cost)

l0=1e-4
step_size=l0
num_epochs=1
fig, axs = plt.subplots(2,1,sharex='all')
paramsA=params_toAdam(params)
tempindex=250
'''
opt=optax.adam(step_size)
stime=time.time()
solver=OptaxSolver(opt=opt, fun=TotalCostAdam3D,maxiter=num_epochs,tol=1e-4)
res=solver.run(paramsA,alist,dx,dp,dt,xv,pv,padmatrix)
ftime=time.time()
print("Adam time :", ftime-stime)
paramsf,state=res
params=params_fromAdam(paramsf)
u=OutputNNAdam3D(paramsf,alist[0])
u1=OutputNNAdam3D(paramsf,alist[tempindex])
'''
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
    params=update3D(params,alist,dx,dp,dt,xv,pv,step_size,padmatrix)
    epoch_time=time.time()-stime
    train_acc = TotalCost3D(params,alist,dx,dp,dt,xv,pv,padmatrix)
    #costlist[epoch]=train_acc
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Total cost {}".format(train_acc))
print("Manual time : ", time.time()-Full_stime)
u=OutputNN3D(params,alist[0])
u1=OutputNN3D(params,alist[tempindex])
'''
'''
lwd=3
tind=5
maxu=np.max(u[s1p:-s1p,s2p:-s2p,tind])
minu=np.min(u[s1p:-s1p,s2p:-s2p,tind])
divnorm=colors.TwoSlopeNorm(vmin=minu, vcenter=(maxu+minu)/2, vmax=maxu)
#pcolormesh(your_data, cmap="coolwarm",)
im1=axs[0].contourf(xv[s1p:-s1p,s2p:-s2p,tind],pv[s1p:-s1p,s2p:-s2p,tind],u[s1p:-s1p,s2p:-s2p,tind],levels=8,cmap='RdBu', norm=divnorm)
cb1=plt.colorbar(im1,ax=axs[0])
cb1.ax.tick_params(labelsize=12)

maxu1=np.max(u1[s1p:-s1p,s2p:-s2p,tind])
minu1=np.min(u1[s1p:-s1p,s2p:-s2p,tind])
divnorm=colors.TwoSlopeNorm(vmin=minu1, vcenter=(maxu1+minu1)/2, vmax=maxu1)
im2=axs[1].contourf(xv[s1p:-s1p,s2p:-s2p,tind],pv[s1p:-s1p,s2p:-s2p,tind],u1[s1p:-s1p,s2p:-s2p,tind],levels=8,cmap='RdBu', norm=divnorm)
cb2=plt.colorbar(im2,ax=axs[1])
cb2.ax.tick_params(labelsize=12)


axs[1].tick_params(labelsize=18)
axs[0].tick_params(labelsize=18)
axs[1].set_xlabel('$x$',fontsize=20)
axs[0].set_ylabel('$p$',fontsize=20)
axs[0].yaxis.set_label_coords(-.15, -.15)
#axs[0].legend(loc=1,fontsize=15)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
#fig.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/NTTResearch/Plots/Wigner3D.png',bbox_inches='tight')
print(i_seed)
'''

#outfile=TemporaryFile()
fname='/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/NTTResearch/OPO_codes/data1'
np.savez(fname+'.npz',dv=dv,da=da,kmax1=kmax1,kmax2=kmax2,kmax3=kmax3,s1=s1,s2=s2,s3=s3,s1p=s1p,s2p=s2p,s3p=s3p,xs=xs,ps=ps,ts=ts,dx=dx,dp=dp,dt=dt,alist=alist,i_seed=i_seed,padmatrix=padmatrix,num_epochs=num_epochs,step_size=step_size)
with open(fname+'.pickle','wb') as file:
    pickle.dump(params,file)

#_=outfile.seek(0)

