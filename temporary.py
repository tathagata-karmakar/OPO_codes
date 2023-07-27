#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:13:16 2023

@author: t_karmakar
"""
import os,sys
import time
import qutip as qt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps as intg
from matplotlib import rc
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('text',usetex=True)

from qutip.measurement import measure, measurement_statistics
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import controlled_gate, hadamard_transform
from qutip import tensor, basis

def controlled_hadamard():
    return controlled_gate(hadamard_transform(1),2,control=0,target=1,control_value=1)

qc=QubitCircuit(N=3,num_cbits=3)
qc.user_gates={"cH": controlled_hadamard}
qc.add_gate("QASMU",targets=[0],arg_value=[1.91063,0,0])
qc.add_gate("cH",targets=[0,1])
qc.add_gate("TOFFOLI",targets=[2],controls=[0,1])
qc.add_gate("X",targets=[0])
qc.add_gate("X",targets=[1])
qc.add_gate("CNOT",targets=[1],controls=0)

zero_state=tensor(basis(2,0),basis(2,0),basis(2,0))
result=qc.run(state=zero_state)
wstate=result

qc.add_measurement("M0",targets=[0],classical_store=0)
qc.add_measurement("M1",targets=[1],classical_store=1)
qc.add_measurement("M2",targets=[2],classical_store=2)

result=qc.run_statistics(state=zero_state)
states=result.get_final_states()
probabilities=result.get_probabilities()

for state,probability in zip(states,probabilities):
    print("State:\n{}\nwith probability {}".format(state,probability))
#qc.png