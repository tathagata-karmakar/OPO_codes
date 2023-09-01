#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:08:16 2023

@author: t_karmakar
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from scipy.integrate import simps as intg
from google.colab import files
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)


import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )


class FNO_3D(nn.Module):
    def __init__(self, da, dv, dout,s1,s2,s3,kmax1,kmax2,kmax3):
        super(FNO_3D, self).__init__()
        #self.initial_linear = nn.Linear(da,dv)
        #self.actv = nn.GELU()
        self.kmax1 = kmax1
        self.kmax2 = kmax2
        self.kmax3 = kmax3
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.actv = nn.GELU()
        self.Shallow = nn.Sequential(
            nn.Linear(da,dv),
            nn.GELU(),
            )

        #self.Rphir1 = nn.Parameter(2*np.sqrt(2/dv)*(torch.rand((kmax1,kmax2,kmax3,dv,dv))-0.5))
        self.Rphir1 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = np.sqrt(2/dv)*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.Rphii1 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = np.sqrt(2/dv)*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.w1 = nn.Parameter(torch.normal(mean = torch.zeros(dv,dv), std = np.sqrt(2/dv)*torch.ones(dv,dv)))

        self.Rphir2 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = np.sqrt(2/dv)*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.Rphii2 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = np.sqrt(2/dv)*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.w2 = nn.Parameter(torch.normal(mean = torch.zeros(dv,dv), std = np.sqrt(2/dv)*torch.ones(dv,dv)))

        self.Rphir3 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = np.sqrt(2/dv)*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.Rphii3 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = np.sqrt(2/dv)*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.w3 = nn.Parameter(torch.normal(mean = torch.zeros(dv,dv), std = np.sqrt(2/dv)*torch.ones(dv,dv)))
        #self.w3 = nn.Parameter(torch.normal(torch.zeros(dv,dv)))

        self.Rphir4 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = np.sqrt(2/dv)*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.Rphii4 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = np.sqrt(2/dv)*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.w4 = nn.Parameter(torch.normal(mean = torch.zeros(dv,dv), std = np.sqrt(2/dv)*torch.ones(dv,dv)))

        self.Project = nn.Linear(dv, dout)

    def FLayer(self,Rr1, Ri1, ws, input):
        f = torch.fft.rfftn(input,dim =(0,1,2))[:self.kmax1,:self.kmax2,:self.kmax3]
        #Rr1 = (self.Rphir1 + torch.flip(self.Rphir1, [0,1]))/2
        #Ri1 = (self.Rphii1 - torch.flip(self.Rphii1, [0,1]))/2
        #Rf = torch.einsum('abc,abcd->abd',f,Rr1+1j*Ri1)
        Rf = torch.einsum('abcd,abcde->abce',f,Rr1+1j*Ri1)
        #print (input.size())
        kernelpart = torch.fft.irfftn(Rf,s=(self.s1,self.s2,self.s3),dim=(0,1,2))
        outs = kernelpart + torch.matmul(input, ws)
        return self.actv(outs)

    def forward(self, input):
        #lin = self.initial_linear(input)
        vt0 = self.Shallow(input)
        vt1 = self.FLayer(self.Rphir1, self.Rphii1, self.w1, vt0)
        vt2 = self.FLayer(self.Rphir2, self.Rphii2, self.w2, vt1)
        vt3 = self.FLayer(self.Rphir3, self.Rphii3, self.w3, vt2)
        vt4 = self.FLayer(self.Rphir4, self.Rphii4, self.w4, vt3)
        u = self.Project(vt4)[:,:,:,0]

        return u
        #vt1 =

class TotalCost(nn.Module):
    def __init__(self, xs, ps, ts, xv, pv, tv, dx, dp, dt, padM, padT):
        super(TotalCost, self).__init__()
        self.dx = dx
        self.dp = dp
        self.dt = dt
        self.xs = xs
        self.ps = ps
        self.ts = ts
        self.xv = xv
        self.pv = pv
        self.tv = tv
        self.padmatrix = padM
        self.padt = padT

    def CostCal(self,u,avs):
        W = u*self.tv/(2*np.pi) + (1-self.tv/(2*np.pi))*avs[:,:,:,0]
        dudx, dudp, dudt = torch.gradient(W,spacing = (self.xs, self.ps, self.ts),dim=(0,1,2))
        #print (u.size(),avs.size())
        #dudt = torch.gradient(u, spacing = self.ts, dim =1)
        #cf  = 10*torch.trapezoid(torch.trapezoid((abs(avs[:,:,0]*dudx+dudt))*self.padmatrix,dx=self.dt,dim=1),dx=self.dx,dim=0)+1000*torch.trapezoid(((u[:,0]-avs[:,0,1])**2)*self.padmatrix[:,0],dx=self.dx,dim=0)
        #cf = 1e3*(1*torch.trapezoid(torch.trapezoid(torch.trapezoid((abs(dudt-self.xv*dudp+self.pv*dudx))*self.padmatrix, dx = self.dt, dim =2), dx = self.dp, dim = 1), dx = self.dx, dim = 0)+1*torch.trapezoid(abs(torch.trapezoid(torch.trapezoid(W, dx = self.dp, dim = 1), dx = self.dx, dim = 0)-1)*self.padt, dx = self.dt,  dim = 0)+1*torch.trapezoid(torch.trapezoid(abs(W[:,:,0]-avs[:,:,0,0])*self.padmatrix[:,:,0], dx = self.dp, dim = 1), dx = self.dx, dim = 0))
        cf = 1e3*(1*torch.trapezoid(torch.trapezoid(torch.trapezoid((abs(dudt-self.xv*dudp+self.pv*dudx))*self.padmatrix, dx = self.dt, dim =2), dx = self.dp, dim = 1), dx = self.dx, dim = 0)+1*torch.trapezoid(abs(torch.trapezoid(torch.trapezoid(W, dx = self.dp, dim = 1), dx = self.dx, dim = 0)-1)*self.padt, dx = self.dt,  dim = 0))
        return cf

    def forward(self,ulist,alist):
        costs = torch.vmap(self.CostCal)(ulist,alist)
        return costs.mean()



def trainingloop(model, datlist, costf, optimizer):

    model.train()
    for batch, alist in enumerate(datlist):
      pred = torch.vmap(model)(alist)
      costval = costf(pred, alist)

      costval.backward()
      optimizer.step()
      optimizer.zero_grad()

    print(costval)
    
dv=8
da=1
kmax=10

s1=39
s2=39
s3=21

kmax1=int(1.5*kmax)
kmax2=int(1.5*kmax)
kmax3=kmax   #kmax should be lesser than the half of number of grid points along the particular dimension
'''Padding Lengths'''
s1p=3
s2p=3
s3p=4
'''---------------'''
Nvars=1+dv*(da+2)+4*dv*dv+8*kmax1*kmax2*kmax3*dv*dv
#NT=20
dx=0.01
xi=-10
xf=10
pi=-10
pf=10
ti=0
tf=2*np.pi

xs_temp=torch.linspace(xi,xf,s1-s1p,device = device)
ps_temp = torch.linspace(pi, pf, s2-s2p, device = device)
ts_temp=torch.linspace(ti,tf,s3-s3p,device = device)
dx=xs_temp[1]-xs_temp[0]
dp=ps_temp[1]-ps_temp[0]
dt=ts_temp[1]-ts_temp[0]

xs=torch.linspace(xi-dx*(s1p),xf+dx*(s1p),s1, device = device)
ps=torch.linspace(pi-dp*(s2p),pf+dp*(s2p),s2, device = device)
ts=torch.linspace(ti,ti+dt*(s3-1),s3, device = device)

xv, pv, tv=torch.meshgrid(xs,ps,ts,indexing='ij')


Nalr = 12
Nali = 12
batch_size = 16
alrs=torch.linspace(xi*0.7,xf*0.7, Nalr, device = device)
alis=torch.linspace(pi*0.7,pf*.7, Nali, device = device)


alist=torch.zeros((Nalr*Nali, s1, s2, s3, da),device = device)
chcklist=torch.zeros((Nalr*Nali, s1, s2, s3, da),device = device)

padmatrix=torch.zeros((s1, s2, s3),device = device)
padmatrix[s1p:-s1p,s2p:-s2p,:-s3p]=torch.ones((s1-2*s1p,s2-2*s2p,s3-s3p))
padt=torch.zeros(s3, device = device)
padt[:-s3p]=torch.ones(s3-s3p)

xv1=xv*torch.cos(tv)-pv*torch.sin(tv)
pv1=pv*torch.cos(tv)+xv*torch.sin(tv)

n=0
for i in range(Nalr):
  alr = alrs[i]
  for j in range(Nali):
    ali = alis[j]
    #print(sigmav)
    chcklist[n,s1p:-s1p,s2p:-s2p,:-s3p,0]=torch.exp(-(xv1-alr)**2-(pv1-ali)**2)[s1p:-s1p,s2p:-s2p,:-s3p]/np.pi
    alist[n,s1p:-s1p,s2p:-s2p,:-s3p,0]=torch.exp(-(xv-alr)**2-(pv-ali)**2)[s1p:-s1p,s2p:-s2p,:-s3p]/np.pi
    #atmp[:-s1p,:-s2p,1]=gauss(xv,mu,sigmav)[:-s1p,:-s2p]
    n+=1

model = FNO_3D(da,dv,1,s1,s2,s3,kmax1,kmax2,kmax3).to(device)
total_params = sum(p.numel() for p in model.parameters())

print(model)

X = torch.rand(s1,s2,s3, 2, device=device)
#logits = torch.vmap(model)(alist)


costf = TotalCost(xs, ps,  ts, xv, pv, tv, dx, dp, dt, padmatrix, padt).to(device)
##costv = costf(logits,alist)

l0=1e-4
optimizer = torch.optim.Adam(model.parameters(),lr = l0)

epochs = 8000


datlist = DataLoader(alist, batch_size = batch_size, shuffle =True)
Ttime = time.time()
for t in range(1,epochs+1):
    stime = time.time()
    #if t % 400 == 0:
     # l0 = l0/2
      #optimizer = torch.optim.Adam(model.parameters(),lr = l0)
    trainingloop(model, datlist, costf, optimizer)
    dtime = time.time()-stime
    print (t+1,round(dtime,3))
Ttime = time.time()-Ttime
print('Total: ', round(Ttime,3))


fig, axs = plt.subplots(2,1,figsize=(9,9),sharex='all')
model.eval()

tempindex=100
xvnp = xv.cpu().numpy()
pvnp = pv.cpu().numpy()
u = model(alist[tempindex])*tv/(2*np.pi)+(1-tv/(2*np.pi))*alist[tempindex][:,:,:,0]
unp = u.detach().cpu().numpy()
#avalnp = alist[tempindex].cpu().numpy()

lwd=3
tind=6
maxu=np.max(unp[s1p:-s1p,s2p:-s2p,tind])
minu=np.min(unp[s1p:-s1p,s2p:-s2p,tind])
divnorm=colors.TwoSlopeNorm(vmin=minu, vcenter=(minu+maxu)/2, vmax=maxu)
im1=axs[0].contourf(xvnp[s1p:-s1p,s2p:-s2p,tind],pvnp[s1p:-s1p,s2p:-s2p,tind],unp[s1p:-s1p,s2p:-s2p,tind],levels=10,cmap='RdBu', norm=divnorm)
cb1=plt.colorbar(im1,ax=axs[0])
cb1.ax.tick_params(labelsize=12)


tind=10
u1=chcklist[tempindex][:,:,:,0].cpu().numpy()
maxu1=np.max(u1[s1p:-s1p,s2p:-s2p,tind])
minu1=np.min(u1[s1p:-s1p,s2p:-s2p,tind])
divnorm=colors.TwoSlopeNorm(vmin=minu1, vcenter=(minu1+maxu1)/2, vmax=maxu1)
im2=axs[1].contourf(xvnp[s1p:-s1p,s2p:-s2p,tind],pvnp[s1p:-s1p,s2p:-s2p,tind],u1[s1p:-s1p,s2p:-s2p,tind],levels=10,cmap='RdBu', norm=divnorm)
cb2=plt.colorbar(im2,ax=axs[1])
cb2.ax.tick_params(labelsize=12)

axs[1].tick_params(labelsize=18)
axs[0].tick_params(labelsize=18)
#axs[1].set_xlabel('$x$',fontsize=20)
#axs[0].set_ylabel('$p$',fontsize=20)
axs[0].yaxis.set_label_coords(-.15, -.15)
#axs[0].legend(loc=1,fontsize=15)
axs[0].set_box_aspect(1)
axs[1].set_box_aspect(1)

plt.subplots_adjust(wspace=0.05, hspace=0.1)
