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
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('text',usetex=True)


#1-d Burgers' equation

'''
Equation parameters
'''

def nu(nu1,da1,kmax1): #Equation parameter
    return nu1*np.ones((da1,kmax1)) #dimension da x kmax x NT

'''
Neural operator evaluation
'''
def sig(x): #Activation function
    return 1.0/(1.0+np.exp(-x)) #dimension same as input
'''
def shallowNN(x,W1,b1,nu1,da1):# Initial shallow NN
# W1 is a matrix with dimension dvxda, b1 is a vector of dimension dv
    nuv=nu(x,nu1,da1)
    return sig(np.matmul(W1,nuv)+b1) #dimension dv
def FT(kmax1,Cv1,Sv1,vt1r,vt1i,dv1):#Fourier transform
#vt1r and vt1i are of dimensions dv x Nx
#Cv1 and Sv1 are of dimensions Nx x kmax
    fr,fi=np.zeros(dv1,kmax1),np.zeros(dv1,kmax1)
    for j in range(kmax1):
        for i in range(dv1):
            fr[i,j]=np.sum(vt1r[i]*Cv1[:,j]+vt1i[i]*Sv1[:,j])
            fi[i,j]=np.sum(vt1i[i]*Cv1[:,j]-vt1r[i]*Sv1[:,j])
    return fr,fi #each with dimensions dv x kmax
def InverseFT(x,kmax1,ks1,f1r,f1i,Nx1,dv1):#Inverse Fourier transform
#pointwise evaluation
#f1r and f1i are of dimensions dv x kmax
    vr,vi=np.zeros(dv1),np.zeros(dv1)
    argv=2*np.pi*x*ks1/Lx
    Cv1=np.cos(argv)
    Sv1=np.sin(argv)
    for i in range(dv1):
        vr[i]=np.sum(f1r[i]*Cv1-f1i[i]*Sv1)
        vi[i]=np.sum(f1i[i]*Cv1+f1r[i]*Sv1)
    return vr,vi #Dimension dv
'''

def shallowNN(x,W1,b1,nu1,da1,kmax1,dv1):# Initial shallow NN
# W1 is a matrix with dimension dvxda, b1 is a vector of dimension dv  
    nuv=nu(nu1,da1,kmax1) #dimension da x kmax x NT1
    b1v=np.reshape(np.tile(b1,kmax1),(kmax1,dv1)).T
    #print (np.shape(b1v))
    return sig(np.matmul(W1,nuv)+b1v) #dimension dv x kmax
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
    sig_arg=np.matmul(W1,vt1)+kernelpart 
    return sig(sig_arg) #dimension dv x kmax

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
    return u[0] #dimension kmax

def costF(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1,nu1,da1,kmax1,dv1,dx1,al1): 
    #Cost function calculations for a single parameter value
    u=OutputNN(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1,nu1,da1,kmax1,dv1)
    u1=OutputNN(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1+dx1,nu1,da1,kmax1,dv1)
    du=(u1-u)/dx1
    cf=intg((du-al1*xs1)**2,xs1)+1.0*u[0]**2
    #print (np.shape(u[0]),u[0])
    return cf

def TotalCost(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1,nu1,da1,kmax1,dv1,dx1,als1): 
    #Total cost over all parameter values
    l=len(als1)
    cfs=0
    for i in range(l):
        cfs+=costF(W0,b0,W1,kappa1,W2,kappa2,W3,kappa3,W4,kappa4,Wf,bf,xs1,nu1,da1,kmax1,dv1,dx1,als1[i])
    return cfs/l

        



'''
xvs,kvs=np.meshgrid(xs,ks,indexing='ij')

argv=2*np.pi*xvs*kvs/Lx
Cvs=np.cos(argv)
Svs=np.sin(argv)
'''


