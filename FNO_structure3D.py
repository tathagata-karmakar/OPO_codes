#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:55:49 2023

@author: t_karmakar
"""

#Neural operator structure from the references below
#[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations, Journal of Computational Physics 378, 686 (2019).
#[2] Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, and A. Anandkumar, Physics-Informed Neural Operator for Learning Partial Differential Equations, arXiv:2111.03794.

import os
#import qutip as qt
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.integrate import simps as intg
from matplotlib import rc
from pylab import rcParams
#from FNO_structure import *
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('text',usetex=True)

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from jax import device_put
from jax import make_jaxpr
from jax.scipy.special import logsumexp
from jax._src.nn.functions import relu,gelu
from functools import partial
import collections 
from typing import Iterable


#Input in d dimensions x1, x2, ... , xd
#The dimensions have discretization s1, s2, ... , sd respectively  
#s1 x s2 x s3 x .. x sd = n

'''
Equation parameters
'''
def aMatrix(a1,shape1): #shape1 is (s1, s2, ..., sd, da)
    #Equation parameter
    return a1*np.ones(shape1) #dimension s1 x s2 x s3 x ... x sd x da

'''
Initialization
'''
def shallow_initial(da1,dv1,key,scale=1e-2):
    #initialize shallow NN parameters
    w_key,b_key=random.split(key)
    return scale*random.normal(w_key,(da1,dv1)), scale*random.normal(b_key,(dv1,))

'''For 3d Domain -----'''
def Fourier_initial(k1,k2,k3,dv1,key,scale=1e1/7):
    ''' ------------------'''
    #initialize Fourier layer parameters
    kappa_key,w_key=random.split(key)
    return scale*random.normal(w_key,(dv1,dv1)), scale*(random.normal(kappa_key,(k1,k2,k3,dv1,dv1))+1j*random.normal(kappa_key,(k1,k2,k3,dv1,dv1)))

def project_initial(dv1,key,scale=1e-2):
    #initialize Projection NN parameters
    w_key,b_key=random.split(key)
    return scale*random.normal(w_key,(dv1,)), scale*random.normal(b_key)

'''For 2d Domain -----'''
def init_params(s1,s2,k1,k2,da1,dv1,key,scale=1e0):
    ''' ------------------'''
    keys=random.split(key,3)
    params=[]
    params.append(shallow_initial(da1,dv1,keys[0],scale))
    for i in  range(4):
        params.append(Fourier_initial(k1,k2,dv1,keys[i+1],scale))
    params.append(project_initial(dv1,keys[5],scale))
    return params

def params_toAdam(params1):
    parout=[]
    #W0,b0=params[0]
    parout.append(params1[0])
    for w,Rphi in params1[1:-1]:
        Rphir,Rphii=jnp.real(Rphi),jnp.imag(Rphi)
        parout.append((w,Rphir,Rphii))
    parout.append(params1[-1])
    return parout

def params_fromAdam(params1):
    parout=[]
    #W0,b0=params[0]
    parout.append(params1[0])
    for w,Rphir,Rphii in params1[1:-1]:
        Rphi = Rphir+1j*Rphii
        parout.append((w,Rphi))
    parout.append(params1[-1])
    return parout

'''
Neural operator evaluation
'''
def actv(x1): #Activation function
    return gelu(x1)

def shallowNN(avs1,W1,b1):# Initial shallow NN
#avs1 is a matrix of dimension s1 x s2 x s3 x ... x sd x da
# W1 is a matrix with dimension da x dv, b1 is a vector of dimension dv
    return actv(jnp.dot(avs1,W1)+b1) #dimension s1 x s2 x s3 x ... x sd x dv

def ProjectNN(vt1,W1,b1): #NN to project the outputs to Fourier layers to the solution
#vt1 is of dimension s1 x s2 x s3 x ... x sd x dv
#W1 is a vector of length dv (for one dependent variable)
#b1 is a constant
    return jnp.dot(vt1,W1)+b1 #shape s1 x s2 x s3 x ... x sd

def FastFT(vt1):#Fast Fourier transform
#vt1 is of dimensions s1 x s2 x s3 x ... x sd x dv
    '''For 2d Domain -----'''
    f=jnp.fft.fftn(vt1,axes=(0,1))
    f1=f[:6,:6]
    ''' ------------------'''
    return f1 #each with dimensions kmax1 x kmax2 x kmax3 x ... x kmaxd x dv
def InvFastFT(Fvt1):#Inverse Fast Fourier transform
#pointwise evaluation
#Fvt1 is of dimensions kmax1 x kmax2 x kmax3 x ... x kmaxd x dv

    '''For 2d Domain -----'''
    f=jnp.fft.ifftn(Fvt1,s=(43,27),axes=(0,1))
    ''' ------------------'''
    return f #dimension s1 x s2 x s3 x ... x sd x dv

def FourierLayer(vt1,W1,kappa1):
#W1 is of size dv x dv
#vt1 is of size  s1 x s2 x s3 x ... x sd x dv
#kappa1 is of size s1 x s2 x s3 x ... x sd x dv x dv
    ftemp=FastFT(vt1)
    '''For 2d Domain -----'''
    Rphi=FastFT(kappa1)
    RF = jnp.einsum('abc,abcd->abd',ftemp,Rphi)
    ''' ------------------'''
    kernelpart=jnp.real(InvFastFT(RF))
    act_arg=jnp.dot(vt1,W1)+kernelpart
    return actv(act_arg) #dimension s1 x s2 x s3 x ... x sd x dv

def FourierLayerR(vt1,W1,Rphi1):
#W1 is of size dv x dv
#vt1 is of size  s1 x s2 x s3 x ... x sd x dv
#Rphi1 is of size kmax1 x kmax2 x kmax3 x ... x kmaxd x dv x dv
    Rphi2=(Rphi1+jnp.conjugate(jnp.flip(Rphi1,axis=(0,1))))/2.0
    ftemp=FastFT(vt1)
    '''For 2d Domain -----'''
    #Rphi=FastFT(kappa1)
    RF = jnp.einsum('abc,abcd->abd',ftemp,Rphi2)
    ''' ------------------'''
    kernelpart=jnp.real(InvFastFT(RF))
    act_arg=jnp.dot(vt1,W1)+kernelpart
    return actv(act_arg) #dimension s1 x s2 x s3 x ... x sd x dv

def FourierLayerAdam(vt1,W1,Rphir,Rphii):
#W1 is of size dv x dv
#vt1 is of size  s1 x s2 x s3 x ... x sd x dv
#Rphir is of size kmax1 x kmax2 x kmax3 x ... x kmaxd x dv x dv
#Rphii is of size kmax1 x kmax2 x kmax3 x ... x kmaxd x dv x dv
    Rphir=(Rphir+jnp.flip(Rphir,axis=(0,1)))/2.0
    Rphii=(Rphii-jnp.flip(Rphii,axis=(0,1)))/2.0
    ftemp=FastFT(vt1)
    '''For 2d Domain -----'''
    #Rphi=FastFT(kappa1)
    RF = jnp.einsum('abc,abcd->abd',ftemp,Rphir+1j*Rphii)
    ''' ------------------'''
    kernelpart=jnp.real(InvFastFT(RF))
    act_arg=jnp.dot(vt1,W1)+kernelpart
    return actv(act_arg) #dimension s1 x s2 x s3 x ... x sd x dv

def OutputNN(params1,avs1): #NO output given input
    W0,b0=params1[0]
    vt=shallowNN(avs1,W0,b0)
    for w,kapv in params1[1:-1]:
        vt=FourierLayerR(vt,w,kapv)
    w_last,b_last=params1[-1]
    u=ProjectNN(vt,w_last,b_last)
    return u #dimension s1 x s2 x s3 x ... x sd

def OutputNNAdam(params1,avs1): #NO output given adam input
    W0,b0=params1[0]
    vt=shallowNN(avs1,W0,b0)
    for w,Rphir,Rphii in params1[1:-1]:
        vt=FourierLayerAdam(vt,w,Rphir,Rphii)
    w_last,b_last=params1[-1]
    u=ProjectNN(vt,w_last,b_last)
    return u   

'''
Cost function calculation
'''
def gauss(x,mu,sigma):
    return jnp.exp(-(x-mu)**2/(2*sigma**2))/(sigma*jnp.sqrt(2*jnp.pi))

'''For 2d Domain -----'''
def CostF(u,avs1,dx1,dt1,padM):
    ''' ------------------'''
    dudx=jnp.gradient(u,dx1,axis=0)
    dudt=jnp.gradient(u,dt1,axis=1)
    cf=(10.*jnp.sum((abs(avs1[:,:,0]*dudx+dudt))*padM)*dx1*dt1+1000*jnp.sum(((u[:,0]-avs1[:,0,1])**2)*padM[:,0])*dx1)
    return cf
def TotalCost1(params1,avlist,dx1,dt1):
    cost=0
    for avs1 in avlist:
        u=OutputNN(params1, avs1)
        cost=cost+CostF(u,avs1,dx1,dt1)
    return cost/len(avlist)

def CostCal(params1,avs1,dx1,dt1,padM):
    u=OutputNN(params1, avs1)
    cost=CostF(u,avs1,dx1,dt1,padM)
    return cost
batch_cost=vmap(CostCal,in_axes=[None,0,None,None,None])

def TotalCost(params1,avlist,dx1,dt1,padM):
    costs=batch_cost(params1,avlist,dx1,dt1,padM)
    return jnp.sum(costs)/len(avlist)

def CostCalAdam(params1,avs1,dx1,dt1,padM):
    u=OutputNNAdam(params1, avs1)
    cost=CostF(u,avs1,dx1,dt1,padM)
    return cost
batch_costAdam=vmap(CostCalAdam,in_axes=[None,0,None,None,None])

def TotalCostAdam(params1,avlist,dx1,dt1,padM):
    costs=batch_costAdam(params1,avlist,dx1,dt1,padM)
    return jnp.sum(costs)/len(avlist)

@jit
def update(params1,alist1,dx1,dt1,step_size,padM):
    grads=grad(TotalCost)(params1,alist1,dx1,dt1,padM)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params1, grads)]

