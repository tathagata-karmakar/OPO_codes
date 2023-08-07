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

'''For 2d Domain -----'''
def Fourier_initial(s1,s2,dv1,key,scale=1e-2):
    ''' ------------------'''
    #initialize Fourier layer parameters
    kappa_key,w_key=random.split(key)
    return scale*random.normal(w_key,(dv1,dv1)), scale*random.normal(kappa_key,(s1,s2,dv1,dv1))

def project_initial(dv1,key,scale=1e-2):
    #initialize Projection NN parameters
    w_key,b_key=random.split(key)
    return scale*random.normal(w_key,(dv1,)), scale*random.normal(b_key)

def init_params(s1,s2,da1,dv1,key):
    keys=random.split(key,6)
    params=[]
    params.append(shallow_initial(da1,dv1,keys[0]))
    for i in  range(4):
        params.append(Fourier_initial(s1,s2,dv1,keys[i+1]))
    params.append(project_initial(dv1,keys[5]))
    return params

'''
Neural operator evaluation
'''
def relu(x1): #Activation function
    return jnp.maximum(0,x1)

def shallowNN(avs1,W1,b1):# Initial shallow NN
#avs1 is a matrix of dimension s1 x s2 x s3 x ... x sd x da
# W1 is a matrix with dimension da x dv, b1 is a vector of dimension dv  
    return relu(jnp.dot(avs1,W1)+b1) #dimension s1 x s2 x s3 x ... x sd x dv

def ProjectNN(vt1,W1,b1): #NN to project the outputs to Fourier layers to the solution
#vt1 is of dimension s1 x s2 x s3 x ... x sd x dv
#W1 is a vector of length dv (for one dependent variable) 
#b1 is a constant
    return jnp.dot(vt1,W1)+b1 #shape s1 x s2 x s3 x ... x sd

def FastFT(vt1):#Fast Fourier transform
#vt1 is of dimensions s1 x s2 x s3 x ... x sd x dv
    '''For 2d Domain -----'''
    f=jnp.fft.fftn(vt1,axes=(0,1))
    ''' ------------------'''
    return f #each with dimensions s1 x s2 x s3 x ... x sd x dv
def InvFastFT(Fvt1):#Inverse Fast Fourier transform
#pointwise evaluation
#Fvt1 is of dimensions s1 x s2 x s3 x ... x sd x dv
    '''For 2d Domain -----'''
    f=jnp.fft.ifftn(Fvt1,axes=(0,1))
    ''' ------------------'''
    return f #dimension s1 x s2 x s3 x ... x sd x dv

def FourierLayer(vt1,W1,kappa1): 
#W1 is of size dv x dv
#vt1 is of size  s1 x s2 x s3 x ... x sd x dv
#kappa1 is of size s1 x s2 x s3 x ... x sd x dv x dv
    ftemp=FastFT(vt1) 
    '''For 2d Domain -----'''
    RF = jnp.einsum('abc,abcd->abd',ftemp,kappa1) 
    ''' ------------------'''
    kernelpart=jnp.real(InvFastFT(RF))
    act_arg=jnp.dot(vt1,W1)+kernelpart 
    return relu(act_arg) #dimension s1 x s2 x s3 x ... x sd x dv

def OutputNN(params,avs1): #NN output given input 
    W0,b0=params[0]
    vt=shallowNN(avs1,W0,b0)
    for w,kapv in params[1:-1]:
        vt=FourierLayer(vt,w,kapv)
    w_last,b_last=params[-1]
    u=ProjectNN(vt,w_last,b_last)
    return u #dimension s1 x s2 x s3 x ... x sd
