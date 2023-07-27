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
    return nu1*np.ones(())

def sig(x): #Activation function
    return 1.0/(1.0+np.exp(-x))
def shallowNN(x,W1,b1,nu1,da1):# Initial shallow NN
# W1 is a matrix with dimension dvxda, b1 is a vector of dimension dv
    nuv=nu(x,nu1,da1)
    return sig(np.matmul(W1,nuv)+b1)
def FourierLayer(x,vt1,W1,R1,kmax1): 
    
    
    
    
    
    


Nx=20
xs=np.linspace(0,1,Nx)
dv=64
da=1
kmax=16
