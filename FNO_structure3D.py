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
from FNO_structure_jax import *


#Input in d dimensions x1, x2, ... , xd
#The dimensions have discretization s1, s2, ... , sd respectively  
#s1 x s2 x s3 x .. x sd = n




def GWigner(x,p):
    return jnp.exp(-x**2-p**2)/jnp.pi

'''
Initialization
'''

'''For 3d Domain -----'''
def Fourier_initial3D(k1,k2,k3,dv1,key,scale=1e-2):
    ''' ------------------'''
    #initialize Fourier layer parameters
    kappa_key,w_key=random.split(key)
    return scale*random.normal(w_key,(dv1,dv1)), scale*(random.normal(kappa_key,(k1,k2,k3,dv1,dv1))+1j*random.normal(kappa_key,(k1,k2,k3,dv1,dv1)))

'''For 3d Domain -----'''
def init_params3D(k1,k2,k3,da1,dv1,key,scale=1e0):
    ''' ------------------'''
    keys=random.split(key,3)
    params=[]
    params.append(shallow_initial(da1,dv1,keys[0],scale))
    for i in  range(4):
        params.append(Fourier_initial3D(k1,k2,k3,dv1,keys[i+1],scale))
    params.append(project_initial(dv1,keys[5],scale))
    return params

'''
Neural operator evaluation
'''

def FastFT3D(vt1):#Fast Fourier transform
#vt1 is of dimensions s1 x s2 x s3 x ... x sd x dv
    '''For 3d Domain -----'''
    f=jnp.fft.fftn(vt1,axes=(0,1,2))
    f1=f[:6,:6,:6]
    ''' ------------------'''
    return f1 #each with dimensions kmax1 x kmax2 x kmax3 x ... x kmaxd x dv
def InvFastFT3D(Fvt1):#Inverse Fast Fourier transform
#pointwise evaluation
#Fvt1 is of dimensions kmax1 x kmax2 x kmax3 x ... x kmaxd x dv

    '''For 3d Domain -----'''
    f=jnp.fft.ifftn(Fvt1,s=(29,29,21),axes=(0,1,2))
    ''' ------------------'''
    return f #dimension s1 x s2 x s3 x ... x sd x dv


def FourierLayerR3D(vt1,W1,Rphi1):
#W1 is of size dv x dv
#vt1 is of size  s1 x s2 x s3 x ... x sd x dv
#Rphi1 is of size kmax1 x kmax2 x kmax3 x ... x kmaxd x dv x dv
    Rphi2=(Rphi1+jnp.conjugate(jnp.flip(Rphi1,axis=(0,1,2))))/2.0
    ftemp=FastFT3D(vt1)
    '''For 3d Domain -----'''
    #Rphi=FastFT(kappa1)
    RF = jnp.einsum('abcd,abcde->abce',ftemp,Rphi2)
    ''' ------------------'''
    kernelpart=jnp.real(InvFastFT3D(RF))
    act_arg=jnp.dot(vt1,W1)+kernelpart
    return actv(act_arg) #dimension s1 x s2 x s3 x ... x sd x dv

def FourierLayerAdam3D(vt1,W1,Rphir,Rphii):
#W1 is of size dv x dv
#vt1 is of size  s1 x s2 x s3 x ... x sd x dv
#Rphir is of size kmax1 x kmax2 x kmax3 x ... x kmaxd x dv x dv
#Rphii is of size kmax1 x kmax2 x kmax3 x ... x kmaxd x dv x dv
    Rphir=(Rphir+jnp.flip(Rphir,axis=(0,1)))/2.0
    Rphii=(Rphii-jnp.flip(Rphii,axis=(0,1)))/2.0
    ftemp=FastFT3D(vt1)
    '''For 3d Domain -----'''
    #Rphi=FastFT(kappa1)
    RF = jnp.einsum('abcd,abcde->abce',ftemp,Rphir+1j*Rphii)
    ''' ------------------'''
    kernelpart=jnp.real(InvFastFT3D(RF))
    act_arg=jnp.dot(vt1,W1)+kernelpart
    return actv(act_arg) #dimension s1 x s2 x s3 x ... x sd x dv

def OutputNN3D(params1,avs1): #NO output given input
    W0,b0=params1[0]
    vt=shallowNN(avs1,W0,b0)
    for w,kapv in params1[1:-1]:
        vt=FourierLayerR3D(vt,w,kapv)
    w_last,b_last=params1[-1]
    u=ProjectNN(vt,w_last,b_last)
    return u #dimension s1 x s2 x s3 x ... x sd

def OutputNNAdam3D(params1,avs1): #NO output given adam input
    W0,b0=params1[0]
    vt=shallowNN(avs1,W0,b0)
    for w,Rphir,Rphii in params1[1:-1]:
        vt=FourierLayerAdam3D(vt,w,Rphir,Rphii)
    w_last,b_last=params1[-1]
    u=ProjectNN(vt,w_last,b_last)
    return u

'''
Cost function calculation
'''

'''For 3d Domain -----'''
def CostF3D(u,avs1,dx1,dp1,dt1,xv1,pv1,padM,padT):
    ''' ------------------'''
    dudx=jnp.gradient(u,dx1,axis=0)
    dudp=jnp.gradient(u,dp1,axis=1)
    dudt=jnp.gradient(u,dt1,axis=2)
    #Evolution + I.C. + Normalization
    cf=1*jnp.sum((abs(dudt-xv1*dudp+pv1*dudx))*padM)*dx1*dp1*dt1+1*jnp.sum(abs(u[:,:,0]-avs1[:,:,0,0])*padM[:,:,0])*dx1*dp1+1*jnp.sum(abs(jnp.sum(u,axis=(0,1))*dx1*dp1-1)*padT)
    return cf

def CostCal3D(params1,avs1,dx1,dp1,dt1,xv1,pv1,padM,padT):
    u=OutputNN3D(params1, avs1)
    cost=CostF3D(u,avs1,dx1,dp1,dt1,xv1,pv1,padM,padT)
    return cost
batch_cost3D=vmap(CostCal3D,in_axes=[None,0,None,None,None,None,None,None,None])

def TotalCost3D(params1,avlist,dx1,dp1,dt1,xv1,pv1,padM,padT):
    costs=batch_cost3D(params1,avlist,dx1,dp1,dt1,xv1,pv1,padM,padT)
    return jnp.sum(costs)/len(avlist)

def CostCalAdam3D(params1,avs1,dx1,dp1,dt1,xv1,pv1,padM,padT):
    u=OutputNNAdam3D(params1, avs1)
    cost=CostF3D(u,avs1,dx1,dp1,dt1,xv1,pv1,padM,padT)
    return cost
batch_costAdam3D=vmap(CostCalAdam3D,in_axes=[None,0,None,None,None,None,None,None,None])

def TotalCostAdam3D(params1,avlist,dx1,dp1,dt1,xv1,pv1,padM,padT):
    costs=batch_costAdam3D(params1,avlist,dx1,dp1,dt1,xv1,pv1,padM,padT)
    return jnp.sum(costs)/len(avlist)

@jit
def update3D(params1,alist1,dx1,dp1,dt1,xv1,pv1,step_size,padM,padT):
    grads=grad(TotalCost3D)(params1,alist1,dx1,dp1,dt1,xv1,pv1,padM,padT)
    return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params1, grads)]

