#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:57:45 2023

@author: t_karmakar
"""

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


def f(n):
    return np.ones(n)
def g(x):
    return np.sum(f(len(x)))
fjt=jit(f)
gjt=jit(g)