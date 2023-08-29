#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:13:36 2023

@author: t_karmakar
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from scipy.integrate import simps as intg
#from google.colab import files
from matplotlib import rc
from pylab import rcParams
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)


import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
#from torchquad import Simpson


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

class FNO_2D(nn.Module):
    def __init__(self, da, dv, dout,s1,s2,kmax1,kmax2):
        super(FNO_2D, self).__init__()
        #self.initial_linear = nn.Linear(da,dv)
        #self.actv = nn.GELU()
        self.kmax1 = kmax1
        self.kmax2 = kmax2
        self.s1 = s1
        self.s2 = s2
        self.actv = nn.GELU()
        self.Shallow = nn.Sequential(
            nn.Linear(da,dv),
            nn.GELU(),    
            )
        
        self.Rphir1 = nn.Parameter(torch.ones(kmax1,kmax2,dv,dv))
        self.Rphii1 = nn.Parameter(torch.ones(kmax1,kmax2,dv,dv))
        self.w1 = nn.Parameter(torch.zeros(dv,dv))
        
        self.Rphir2 = nn.Parameter(torch.ones(kmax1,kmax2,dv,dv))
        self.Rphii2 = nn.Parameter(torch.ones(kmax1,kmax2,dv,dv))
        self.w2 = nn.Parameter(torch.zeros(dv,dv))
        
        self.Rphir3 = nn.Parameter(torch.ones(kmax1,kmax2,dv,dv))
        self.Rphii3 = nn.Parameter(torch.ones(kmax1,kmax2,dv,dv))
        self.w3 = nn.Parameter(torch.zeros(dv,dv))
        
        self.Rphir4 = nn.Parameter(torch.ones(kmax1,kmax2,dv,dv))
        self.Rphii4 = nn.Parameter(torch.ones(kmax1,kmax2,dv,dv))
        self.w4 = nn.Parameter(torch.zeros(dv,dv))
        
        self.Project = nn.Linear(dv, dout)
        
    def FLayer(self,Rr1, Ri1, ws, input):
        #f = torch.fft.fftn(input,dim =(0,1))[:self.kmax1,:self.kmax2]
        f = torch.fft.rfftn(input,dim =(0,1))[:self.kmax1,:self.kmax2]
        #Rr1 = (self.Rphir1 + torch.flip(self.Rphir1, [0,1]))/2
        #Ri1 = (self.Rphii1 - torch.flip(self.Rphii1, [0,1]))/2
        #Rf = torch.einsum('abc,abcd->abd',f,Rr1+1j*Ri1)
        Rf = torch.einsum('abc,abcd->abd',f,Rr1+1j*Ri1)
        #print (f1,f)
        kernelpart = torch.fft.irfftn(Rf,s=(self.s1,self.s2),dim=(0,1))
        outs = kernelpart + torch.matmul(input, ws)
        return self.actv(outs)
        
    def forward(self, input):
        #lin = self.initial_linear(input)
        vt0 = self.Shallow(input)
        vt1 = self.FLayer(self.Rphir1, self.Rphii1, self.w1, vt0)
        vt2 = self.FLayer(self.Rphir2, self.Rphii2, self.w2, vt1)
        vt3 = self.FLayer(self.Rphir3, self.Rphii3, self.w3, vt2)
        vt4 = self.FLayer(self.Rphir4, self.Rphii4, self.w4, vt3) 
        u = self.Project(vt4)[:,:,0]
        
        return u
        #vt1 = 

class TotalCost(nn.Module):
    def __init__(self, xs, ts, dx, dt, padM):
        super(TotalCost, self).__init__()
        self.dx = dx
        self.dt = dt
        self.xs = xs
        self.ts = ts
        self.padmatrix = padM
    
    def CostCal(self,u,avs):
        dudx,dudt = torch.gradient(u,spacing = (self.xs,self.ts),dim=(0,1))
        #print (u.size(),avs.size())
        #dudt = torch.gradient(u, spacing = self.ts, dim =1)
        cf  = 10.0*torch.trapezoid(torch.trapezoid((abs(avs[:,:,0]*dudx+dudt))*self.padmatrix,dx=self.dt,dim=1),dx=self.dx,dim=0)+1000*torch.trapezoid(((u[:,0]-avs[:,0,1])**2)*self.padmatrix[:,0],dx=self.dx,dim=0)
        return cf
    
    def forward(self,ulist,alist):
        costs = torch.vmap(self.CostCal)(ulist,alist)
        return costs.mean()
    
    
def trainingloop(model, alist, costf, optimizer):
    
    model.train()
    pred = torch.vmap(model)(alist)
    costval = costf(pred, alist)
    
    costval.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(costval)
    
        

dv=6
da=2
kmax=6
kmax1=kmax
kmax2=kmax
s1=43
s2=27
'''Padding Lengths'''
s1p=6
s2p=6
'''---------------'''
Nvars=1+dv*(da+2)+4*dv*dv+8*kmax*kmax*dv*dv
#NT=20
dx=0.01
xi=-2
xf=2
ti=0
tf=1

xs_temp=torch.linspace(xi,xf,s1-s1p)
ts_temp=torch.linspace(ti,tf,s2-s2p)
dx=xs_temp[1]-xs_temp[0]
dt=ts_temp[1]-ts_temp[0]
ts=torch.linspace(ti,ti+dt*(s2-1),s2)
xs=torch.linspace(xi,xi+dx*(s1-1),s1)
xv,tv=torch.meshgrid(xs,ts,indexing='ij')

Na=5
Ns=300
avs=torch.linspace(0.01,1,Na)
mu=0.5
sigmas=torch.linspace(0.08,1,Ns)
alist=torch.zeros((Na*Ns,s1,s2,da))

padmatrix=torch.zeros((s1,s2))
fpadmat=torch.zeros((s1,s2,dv,dv))
padmatrix[:-s1p,:-s2p]=torch.ones((s1-s1p,s2-s2p))
fpadmat[:kmax,:kmax,:,:]=torch.ones((kmax,kmax,dv,dv))
n=0
for i in range(len(avs)):
    av=avs[i]
    for j in range(len(sigmas)):
        sigmav=sigmas[j]
        alist[n,:-s1p,:-s2p,0]=av*torch.ones(s1,s2)[:-s1p,:-s2p]
        alist[n,:-s1p,:-s2p,1]=(torch.exp(-(xv-mu)**2/(2*sigmav**2))[:-s1p,:-s2p])/(sigmav*np.sqrt(2*np.pi))
        #alist.append(atmp)
        n+=1

model = FNO_2D(da,dv,1,s1,s2,kmax1,kmax2).to(device)
total_params = sum(p.numel() for p in model.parameters())

print(model)
        
X = torch.rand(s1,s2, 2, device=device)
logits = torch.vmap(model)(alist)

costf = TotalCost(xs, ts, dx,dt,padmatrix)
costv = costf(logits,alist)

optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

epochs = 10

for t in range(epochs):
    stime = time.time()
    trainingloop(model, alist, costf, optimizer)
    dtime = time.time()-stime
    
    print (dtime)


#var1=torch.linspace(0,1,100)
#var2=torch.linspace(10,30,300)
#v1,v2=torch.meshgrid(var1,var2,indexing='ij')
#f = v1**2+v2

#pred_probab = nn.Softmax(dim=1)(logits)
#y_pred = pred_probab.argmax(1)
#print(f"Predicted class: {y_pred}")