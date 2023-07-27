#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:06:10 2023

@author: t_karmakar
"""

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

from qutip.measurement import measure, measurement_statistics,measurement_statistics_observable,measurement_statistics_povm


Da,Db=40,35  #Number of Fock states
psi_a=qt.coherent(Da,0.7)
w=0.25
gt=1
Delta_by_g=150
gtilde_by_g=1
Deltat=Delta_by_g*gt
gtildet=gtilde_by_g*gt
psi_b=(qt.squeeze(Db,np.log(2*w))*qt.fock(Db,0))
figa,axa=qt.plot_wigner(psi_a,alpha_max=4.5,figsize=(6,6))
figb,axb=qt.plot_wigner(psi_b,alpha_max=4.5,figsize=(6,6))

c1=(np.sqrt(np.sqrt(2)+1)+np.sqrt(np.sqrt(2)-1))/2
c2=(np.sqrt(np.sqrt(2)+1)-np.sqrt(np.sqrt(2)-1))/2
a=qt.destroy(Da)
b=qt.destroy(Db)
A=c1*a+c2*a.dag()
Na=A.dag()*A
xb=(b+b.dag())/2
HD=-2*gtilde_by_g*qt.tensor((Na+1/2),xb)+Delta_by_g*qt.tensor(Na,qt.identity(Db))
r_by_g=Delta_by_g
del_by_g=np.sqrt(2)*Delta_by_g
beta=r_by_g/2.0
HNL=qt.tensor(a.dag()*a.dag(),b)+qt.tensor(a*a,b.dag())
HQ=qt.tensor(del_by_g*a.dag()*a,qt.identity(Db))+qt.tensor(r_by_g*(a.dag()*a.dag()+a*a)/2,qt.identity(Db))
#Dispb=qt.tensor(qt.identity(Da),qt.exp(beta*b.dag()-beta*b))
HD=HNL+HQ
psi0=qt.tensor(psi_a,psi_b)

Nts=500
Nxs,Nys=500,490
gts=np.linspace(0,1*gt,Nts)
result=qt.sesolve(HD,psi0,gts,[])
psif=result.states[Nts-1]
psi_af=psif.ptrace(0)
psi_bf=psif.ptrace(1)
xs=np.linspace(-4.5,4.5,Nxs)
#w=qt.wigner(psi_af,xs,xs)
#wmap=qt.wigner_cmap(w)
#nrm=mpl.colors.Normalize(-w.max(),w.max())
figc,axc=qt.plot_wigner(psi_af,alpha_max=4.5,figsize=(6,6))
figd,axd=qt.plot_wigner(psi_bf,alpha_max=7,figsize=(6,6))


xs=np.linspace(-6,6,Nxs)
ys=np.linspace(-0.5,7,Nys)
ws=qt.wigner(psi_bf,xs,ys)
Ppb=intg(ws,xs,axis=-1)
figc1,axc1=plt.subplots()
axc1.plot(ys,Ppb)

#psif=psif/psif.tr()
pbM=qt.tensor(qt.identity(Da),(b-b.dag())/(2j))
eigv,eigs,probs=measurement_statistics_observable(psif,pbM)
Nav=3
lvl,hvl=gtildet*(Nav),gtildet*(Nav+1)

def CNapb(Deltat1,Natmp1,pb1,gtildet1,w1):
    return np.exp(-1j*Deltat1*Natmp1)*np.exp(-(pb1-gtildet1*(Natmp1+0.5))**2/(4*w1**2))/(((2*np.pi)**0.25)*(w1**0.5))

def F(pb1,lNs1,vNs1,inprods1):
    S=0
    for j in range(len(lNs1)):
        Natmp=lNs1[j]
        ketNatmp=vNs1[j]
        CNa=CNapb(Deltat,Natmp,pb1,gtildet,w)
        S=S+(np.absolute(CNa)**2)*(inprods1[j])
    return S
def g(p,pl1,ph1):
    return ((p>=pl1)**2)*(1.0/(ph1-pl1))*((p<=ph1)**2)

lNs,vNs=Na.eigenstates()


inprods=np.zeros(len(lNs))
for j in range(len(lNs)):
    ketNatmp=vNs[j]
    inprods[j]=np.absolute((ketNatmp.dag()*psi_a.data)[0][0])**2

pbarray=np.zeros(5000)
pl,ph=-2,15
parr=np.linspace(pl,ph,2000)
c=20.0
n=0
while n<len(pbarray):
    x=np.random.uniform(pl,ph)
    u=np.random.uniform(0,1)
    if (u<=F(x,lNs,vNs,inprods)/(c*g(x,pl,ph))):
        pbarray[n]=x
        n+=1
        #print (n)


psi_aM=0
cnt=0
for pbv in pbarray:
    if (pbv>=lvl)and(pbv<=hvl):
        cnt+=1
        M=0#*qt.ket2dm(vNs[0])
        for j in range(len(lNs)):
            Natmp=lNs[j]
            ketNatmp=vNs[j]
            CNa=CNapb(Deltat,Natmp,pbv,gtildet,w)
            M=M+CNa*qt.ket2dm(ketNatmp)
        psi_aM+=qt.ket2dm((M*psi_a).unit())
psi_aM=psi_aM/cnt
fige,axe=qt.plot_wigner(psi_aM,alpha_max=4.5,figsize=(6,6))

plt.figure()
plt.plot(parr,F(parr,lNs,vNs,inprods))
#plt.hist(pbarray,bins=100,density=True)
