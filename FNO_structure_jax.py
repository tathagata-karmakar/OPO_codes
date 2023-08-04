#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:02:54 2023

@author: t_karmakar
"""

#Neural operator structure from the references below
#[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations, Journal of Computational Physics 378, 686 (2019).
#[2] Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, and A. Anandkumar, Physics-Informed Neural Operator for Learning Partial Differential Equations, arXiv:2111.03794.

import os
import qutip as qt
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
from functools import partial
import collections 
from typing import Iterable


#Input in d dimensions x1, x2, ... , xd
#The dimensions have discretization s1, s2, ... , sd respectively  

'''
Equation parameters
'''

def nu(nu1,shape1): #shape1 is (s1, s2, ..., sd, da)
    #Equation parameter
    return nu1*np.ones(shape1) #dimension s1 x s2 x s3 x ... x sd x da

'''
Neural operator evaluation
'''
def relu(x): #Activation function
    return jnp.maximum(0,x)


def shallowNN(x,W1,b1,nu1,da1,kmax1,dv1):# Initial shallow NN
# W1 is a matrix with dimension dvxda, b1 is a vector of dimension dv  
    nuv=nu(nu1,da1,kmax1) #dimension da x kmax x NT1
    b1v=np.reshape(np.tile(b1,kmax1),(kmax1,dv1)).T
    #print (np.shape(b1v))
    return relu(np.matmul(W1,nuv)+b1v) #dimension dv x kmax
def ProjectNN(vt1,W1,b1): #NN to project the outputs to Fourier layers to the solution
#vt1 is of dimension dv x kmax
#W1 is of size 1 x dv (for one dependent variable) 
#b1 is a constant
    return np.matmul(W1,vt1)+b1 #shape 1 x kmax

def FastFT(vt1):#Fast Fourier transform
#vt1 is of dimensions dv x kmax
    f=np.fft.fft(vt1,axis=-1)
    return f #each with dimensions dv x kmax
def InvFastFT(Fvt1):#Inverse Fast Fourier transform
#pointwise evaluation
#Fvt1 is of dimensions dv x kmax
    f=np.fft.ifft(Fvt1,axis=-1)
    return f

def FourierLayer(vt1,dv1,W1,kmax1,kappa1): 
#W1 is of size dv x dv
#vt1 is of size dv x kmax
#kappa1 is of size dv x dv x kmax
    Rtensor=np.swapaxes(np.fft.fft(kappa1,axis=-1),1,2) #Rtensor is of size dv x kmax x dv
    f=FastFT(vt1) #shape dv x kmax
    RF=np.zeros((dv1,kmax1),dtype=complex)
    for i in range(kmax1):
        RF[:,i]=np.dot(Rtensor[:,i,:],f[:,i])
    kernelpart=np.real(InvFastFT(RF))
    act_arg=np.matmul(W1,vt1)+kernelpart 
    return relu(act_arg) #dimension dv x kmax

def OutputNN(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1,nu1,da1,kmax1,dv1): #NN output given input 
#W0 is of dimension dv x da, b0 is a vector of dimension dv
#W1 is of dimension dv x dv, kappa1 is of dimension dv x dv x kmax
#W2 is of dimension dv x dv, kappa2 is of dimension dv x dv x kmax
#W3 is of dimension dv x dv, kappa3 is of dimension dv x dv x kmax
#W4 is of dimension dv x dv, kappa4 is of dimension dv x dv x kmax   
#Wf is of dimension 1 x dv, bf is a scalar
    v0=shallowNN(xs1,W0,b0,nu1,da1,kmax1,dv1)
    v1=FourierLayer(v0,dv1,W1,kmax1,kappa1)
    v2=FourierLayer(v1,dv1,W2,kmax1,kappa2)
    v3=FourierLayer(v2,dv1,W3,kmax1,kappa3)
    v4=FourierLayer(v3,dv1,W4,kmax1,kappa4)
    u=ProjectNN(v4,Wf,bf)
    return u #dimension kmax



'''
xvs,kvs=np.meshgrid(xs,ks,indexing='ij')

argv=2*np.pi*xvs*kvs/Lx
Cvs=np.cos(argv)
Svs=np.sin(argv)
'''


