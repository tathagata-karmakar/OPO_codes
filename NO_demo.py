#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:02:54 2023

@author: t_karmakar
"""

#Example from neural operator paper
#[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations, Journal of Computational Physics 378, 686 (2019).
#[2] Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, and A. Anandkumar, Physics-Informed Neural Operator for Learning Partial Differential Equations, arXiv:2111.03794.

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


#1-d Burgers' equation

def nu(x,nu1,da1): #Equation parameter
    return nu1*np.ones(da1)

def sig(x): #Activation function
    return 1.0/(1.0+np.exp(-x))
def shallowNN(x,W1,b1,nu1,da1):# Initial shallow NN
# W1 is a matrix with dimension dvxda, b1 is a vector of dimension dv
    nuv=nu(x,nu1,da1)
    return sig(np.matmul(W1,nuv)+b1)
def FT(kmax1,Cv1,Sv1,vt1r,vt1i,dv1):#Fourier transform
#vt1r and vt1i are of dimensions dv x Nx
    fr,fi=np.zeros(dv1,kmax1),np.zeros(dv1,kmax1)
    for j in range(kmax1):
        for i in range(dv1):
            fr[i,j]=np.sum(vt1r[i]*Cv1[:,j]+vt1i[i]*Sv1[:,j])
            fi[i,j]=np.sum(vt1i[i]*Cv1[:,j]-vt1r[i]*Sv1[:,j])
    return fr,fi
def InverseFT(x1,kmax1,ks1,f1r,f1i,Nx1,dv1):#Inverse Fourier transform
#pointwise evaluation
#f1r and f1i are of dimensions dv x kmax
    vr,vi=np.zeros(dv1),np.zeros(dv1)
    argv=2*np.pi*x1*ks1/Lx
    Cv1=np.cos(argv)
    Sv1=np.sin(argv)
    for i in range(dv1):
        vr[i]=np.sum(f1r[i]*Cv1-f1i[i]*Sv1)
        vi[i]=np.sum(f1i[i]*Cv1+f1r[i]*Sv1)
    return vr,vi
    
#def FourierLayer(x,xs1,Nx1,vt1,W1,R1,kmax1): 
    
    
   

Nx=10
dv=64
da=1
kmax=16
xs=np.linspace(0.01,0.99,Nx)
Lx=xs[1]-xs[0]
ks=np.arange(0,kmax)
xvs,kvs=np.meshgrid(xs,ks,indexing='ij')
argv=2*np.pi*xvs*kvs/Lx
Cvs=np.cos(argv)
Svs=np.sin(argv)


