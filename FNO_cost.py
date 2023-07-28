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
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('text',usetex=True)



