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
        #self.Shallow = nn.Sequential(
         #   nn.Linear(da,dv),
          #  nn.GELU(),
          #  )
        #self.scale1 = 1.0/(da*dv)
        #self.scale2 = 1.0/(dv*dv)
        #self.scale3 = 1.0/(dv*dout)
        self.scale1 = 2/da#np.sqrt(2/da)
        self.scale2 = 2/(1*dv)#np.sqrt(2/(1*dv))
        self.scale3 = 2/dv#np.sqrt(2/dv)
        self.win = nn.Parameter(torch.normal(mean = torch.zeros(da,dv), std = self.scale1*torch.ones(da,dv)))
        self.bin1 = nn.Parameter(torch.normal(mean = torch.zeros(dv), std = self.scale1*torch.ones(dv)))
        self.bin = self.bin1.expand(self.s1,self.s2,self.s3,dv).to(device)

        #self.Rphir1 = nn.Parameter(2*np.sqrt(2/dv)*(torch.rand((kmax1,kmax2,kmax3,dv,dv))-0.5))
        self.Rphir1 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = self.scale2*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.Rphii1 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = self.scale2*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.w1 = nn.Parameter(torch.normal(mean = torch.zeros(dv,dv), std = self.scale2*torch.ones(dv,dv)))

        self.Rphir2 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = self.scale2*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.Rphii2 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = self.scale2*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.w2 = nn.Parameter(torch.normal(mean = torch.zeros(dv,dv), std = self.scale2*torch.ones(dv,dv)))

        self.Rphir3 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = self.scale2*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.Rphii3 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = self.scale2*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.w3 = nn.Parameter(torch.normal(mean = torch.zeros(dv,dv), std = self.scale2*torch.ones(dv,dv)))
        #self.w3 = nn.Parameter(torch.normal(torch.zeros(dv,dv)))

        self.Rphir4 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = self.scale2*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.Rphii4 = nn.Parameter(torch.normal(mean = torch.zeros(kmax1,kmax2,kmax3,dv,dv), std = self.scale2*torch.ones(kmax1,kmax2,kmax3,dv,dv)))
        self.w4 = nn.Parameter(torch.normal(mean = torch.zeros(dv,dv), std = self.scale2*torch.ones(dv,dv)))
        #self.w4 = torch.zeros(dv,dv ,device = device)

        #self.Project = nn.Linear(dv, dout)
        self.wout = nn.Parameter(torch.normal(mean = torch.zeros(dv,dout), std = self.scale3*torch.ones(dv,dout)))
        self.bout = nn.Parameter(torch.normal(mean = torch.zeros(dout), std = self.scale3*torch.ones(dout)))

    def FLayer(self,Rr1, Ri1, ws, input):
        f = torch.fft.rfftn(input,dim =(0,1,2))[:self.kmax1,:self.kmax2,:self.kmax3]
        #del input
        #Rr1 = (self.Rphir1 + torch.flip(self.Rphir1, [0,1]))/2
        #Ri1 = (self.Rphii1 - torch.flip(self.Rphii1, [0,1]))/2
        #Rf = torch.einsum('abc,abcd->abd',f,Rr1+1j*Ri1)
        Rf = torch.einsum('abcd,abcde->abce',f,Rr1+1j*Ri1)
        del f, Rr1, Ri1
        #torch.cuda.empty_cache()
        #print (input.size())
        kernelpart = torch.fft.irfftn(Rf,s=(self.s1,self.s2,self.s3),dim=(0,1,2))
        del Rf
        outs = kernelpart + torch.matmul(input, ws)
        return self.actv(outs)
    '''
    def FLayer(self,Rr1, Ri1, ws, input):
        #Rr1 = (self.Rphir1 + torch.flip(self.Rphir1, [0,1]))/2
        #Ri1 = (self.Rphii1 - torch.flip(self.Rphii1, [0,1]))/2
        #Rf = torch.einsum('abc,abcd->abd',f,Rr1+1j*Ri1)
        return self.actv(torch.fft.irfftn(torch.einsum('abcd,abcde->abce',torch.fft.rfftn(input,dim =(0,1,2))[:self.kmax1,:self.kmax2,:self.kmax3],Rr1+1j*Ri1),s=(self.s1,self.s2,self.s3),dim=(0,1,2)) + torch.matmul(input, ws))
    '''

    def forward(self, input):
        #lin = self.initial_linear(input)
        #vt0 = self.Shallow(input)
        vt0 = self.actv(torch.matmul(input, self.win) + self.bin1)
        del input
        #torch.cuda.empty_cache()
        vt1 = self.FLayer(self.Rphir1, self.Rphii1, self.w1, vt0)
        del vt0
        #torch.cuda.empty_cache()
        vt2 = self.FLayer(self.Rphir2, self.Rphii2, self.w2, vt1)
        del vt1
        #torch.cuda.empty_cache()
        vt3 = self.FLayer(self.Rphir3, self.Rphii3, self.w3, vt2)
        del vt2
        #torch.cuda.empty_cache()
        vt4 = self.FLayer(self.Rphir4, self.Rphii4, self.w4, vt3)
        del vt3
        #torch.cuda.empty_cache()
        #u = self.Project(vt4)[:,:,:,0]
        u = torch.matmul(vt4, self.wout) + self.bout
        del vt4
        #torch.cuda.empty_cache()
        return u
        #vt1 =

class TotalCost(nn.Module):
    def __init__(self, xs, ps, ts, dx, dp, dt, s1p, s2p, s3p):
        super(TotalCost, self).__init__()
        self.dx = dx
        self.dp = dp
        self.dt = dt
        self.s1p = s1p
        self.s2p = s2p
        self.s3p = s3p
        self.xs = xs[s1p:-s1p]
        self.ps = ps[s2p:-s2p]
        self.ts = ts[:-s3p]
        #self.xv = xv[s1p:-s1p,s2p:-s2p,:-s3p]
        #self.pv = pv[s1p:-s1p,s2p:-s2p,:-s3p]
        #self.tv = tv[s1p:-s1p,s2p:-s2p,:-s3p]

    def CostCal(self,u,avs):
        #print(u.requires_grad)
        W = u[self.s1p:-self.s1p,self.s2p:-self.s2p,:-self.s3p,0]#*self.tv/(np.pi) + (1-self.tv/(np.pi))*avs[self.s1p:-self.s1p,self.s2p:-self.s2p,:-s3p,0]
        del u
        dudx, dudp, dudt = torch.gradient(W,spacing = (self.xs, self.ps, self.ts),dim=(0,1,2),edge_order =2)

        cf = 1e3*(1*torch.trapezoid(torch.trapezoid(torch.trapezoid(((dudt-avs[s1p:-s1p,s2p:-s2p,:-s3p,1]*dudp+avs[s1p:-s1p,s2p:-s2p,:-s3p,2]*dudx)**2), dx = self.dt, dim =2), dx = self.dp, dim = 1), dx = self.dx, dim = 0)+1*torch.trapezoid(torch.trapezoid((W[:,:,0]-avs[self.s1p:-self.s1p,self.s2p:-self.s2p,0,0])**2, dx = self.dp, dim = 1), dx = self.dx, dim = 0)+10*torch.trapezoid((torch.trapezoid(torch.trapezoid(W, dx = self.dp, dim = 1), dx = self.dx, dim = 0)-1)**2, dx = self.dt,    dim = 0))
        #cf = 1e3*(1*torch.sum(abs(dudt-self.xv*dudp+self.pv*dudx)**2)+1*torch.sum(abs(torch.trapezoid(torch.trapezoid(W, dx = self.dp, dim = 1), dx = self.dx, dim = 0)-1)**2)+ 10*torch.sum(abs(W[:,:,0]-avs[self.s1p:-self.s1p,self.s2p:-self.s2p,0,0])**2))
        #print(cf)
        return cf

    def forward(self,ulist,alist):
        costs = torch.vmap(self.CostCal)(ulist,alist)
        return costs.mean()

def trainingloop(model, datlist, costf, optimizer, scheduler, varlist):

    model.train()
    for batch, alist in enumerate(datlist):
      #print (alist.size())
      blist = torch.cat((alist, varlist[:len(alist)]), dim = -1)
      optimizer.zero_grad()
      pred = torch.vmap(model)(blist)
      costval = costf(pred, blist)
      costval.backward()
      optimizer.step()
      scheduler.step()
    torch.cuda.empty_cache()
    print(costval)

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


    
def Fock(m, x, f1, f2):
  #prefactor = ((-1)**m)/np.pi
  if m == 0:
    return torch.ones_like(x)
  elif m == 1:
    return (1-x)
  else:
    return ((2*m-1-x)*f1-(m-1)*f2)*(1.0/m)

dv=32
da=4
dout = 1
kmax= 16

s1=59
s2=59
s3=59

kmax1=int(1*kmax)
kmax2=int(1*kmax)
kmax3=kmax   #kmax should be lesser than the half of number of grid points along the particular dimension
'''Padding Lengths'''
s1p=4
s2p=4
s3p=4
'''---------------'''
Nvars=1+dv*(da+2)+4*dv*dv+8*kmax1*kmax2*kmax3*dv*dv
#NT=20

xi=-5
xf=5
pi=-5
pf=5
ti=0
tf=2*np.pi

xs_temp=torch.linspace(xi,xf,s1-2*s1p,device = device)
ps_temp = torch.linspace(pi, pf, s2-2*s2p, device = device)
ts_temp=torch.linspace(ti,tf,s3-s3p,device = device)
dx=xs_temp[1]-xs_temp[0]
dp=ps_temp[1]-ps_temp[0]
dt=ts_temp[1]-ts_temp[0]

xs=torch.linspace(xi-dx*(s1p),xi+dx*(s1-s1p-1),s1, device = device)
ps=torch.linspace(pi-dp*(s2p),pi+dx*(s2-s2p-1),s2, device = device)
ts=torch.linspace(ti,ti+dt*(s3-1),s3, device = device)

xv, pv, tv=torch.meshgrid(xs,ps,ts,indexing='ij')

Nalr = 6
Nali = 6
Nfock = 6
batch_size = 6#int(np.sqrt(Nalr*Nali))
alrs=torch.linspace(xi*0.35,xf*0.35, Nalr, device = device)
alis=torch.linspace(pi*0.35,pf*0.35, Nali, device = device)


alist=torch.zeros((Nalr*Nali + Nfock, s1, s2, s3, da-3),device = device)
varlist=torch.zeros((batch_size, s1, s2, s3, 3),device = device)
#Sollist=torch.zeros((Nalr*Nali+Nfock, s1, s2, s3, da),device = device)

xv1=xv*torch.cos(tv)-pv*torch.sin(tv)
pv1=pv*torch.cos(tv)+xv*torch.sin(tv)

n=0
for i in range(Nalr):
  alr = alrs[i]
  for j in range(Nali):
    ali = alis[j]
    #print(sigmav)
    #chcklist[n,s1p:-s1p,s2p:-s2p,:-s3p,0]=torch.exp(-(xv1-alr)**2-(pv1-ali)**2)[s1p:-s1p,s2p:-s2p,:-s3p]/np.pi
    alist[n,s1p:-s1p,s2p:-s2p,:-s3p,0]=torch.exp(-(xv-np.sqrt(2)*alr)**2-(pv-np.sqrt(2)*ali)**2)[s1p:-s1p,s2p:-s2p,:-s3p]/np.pi
    #alist[n,s1p:-s1p,s2p:-s2p,:-s3p,1]=xv[s1p:-s1p,s2p:-s2p,:-s3p]
    #alist[n,s1p:-s1p,s2p:-s2p,:-s3p,2]=pv[s1p:-s1p,s2p:-s2p,:-s3p]
    #alist[n,s1p:-s1p,s2p:-s2p,:-s3p,3]=tv[s1p:-s1p,s2p:-s2p,:-s3p]
    #Sollist[n,s1p:-s1p,s2p:-s2p,:-s3p,0]=torch.exp(-(xv1-alr)**2-(pv1-ali)**2)[s1p:-s1p,s2p:-s2p,:-s3p]/np.pi
    #Sollist[n,s1p:-s1p,s2p:-s2p,:-s3p,1]=xv[s1p:-s1p,s2p:-s2p,:-s3p]
    #Sollist[n,s1p:-s1p,s2p:-s2p,:-s3p,2]=pv[s1p:-s1p,s2p:-s2p,:-s3p]
    #Sollist[n,s1p:-s1p,s2p:-s2p,:-s3p,3]=tv[s1p:-s1p,s2p:-s2p,:-s3p]
    #atmp[:-s1p,:-s2p,1]=gauss(xv,mu,sigmav)[:-s1p,:-s2p]
    n+=1

for i in range(batch_size):
  varlist[i,s1p:-s1p,s2p:-s2p,:-s3p,0] = xv[s1p:-s1p,s2p:-s2p,:-s3p]
  varlist[i,s1p:-s1p,s2p:-s2p,:-s3p,1] = pv[s1p:-s1p,s2p:-s2p,:-s3p]
  varlist[i,s1p:-s1p,s2p:-s2p,:-s3p,2] = tv[s1p:-s1p,s2p:-s2p,:-s3p]

expfac = torch.exp(-xv**2-pv**2)
arg = 2*(xv**2+pv**2)
lslist = torch.zeros((Nfock, s1, s2, s3),device = device)
if (Nfock >0):
  lslist[0,:,:,:] = torch.ones_like(xv)
  if (Nfock>1):
    lslist[1,:,:,:] = 1-arg

for mode in range(2,Nfock):
  lslist[mode,:,:,:] = Fock(mode, arg, lslist[mode-1,:,:,:],lslist[mode-2,:,:,:])

if (Nfock>0):
  for mode in range(Nfock):
    alist[Nalr*Nali+mode,s1p:-s1p,s2p:-s2p,:-s3p,0]= (((-1)**mode)*expfac*lslist)[mode,s1p:-s1p,s2p:-s2p,:-s3p]/np.pi
    #alist[Nalr*Nali+mode,s1p:-s1p,s2p:-s2p,:-s3p,1]=xv[s1p:-s1p,s2p:-s2p,:-s3p]
    #alist[Nalr*Nali+mode,s1p:-s1p,s2p:-s2p,:-s3p,2]=pv[s1p:-s1p,s2p:-s2p,:-s3p]
    #alist[Nalr*Nali+mode,s1p:-s1p,s2p:-s2p,:-s3p,3]=tv[s1p:-s1p,s2p:-s2p,:-s3p]

#alist.requires_grad(requires_grad = True)
model = FNO_3D(da,dv,dout,s1,s2,s3,kmax1,kmax2,kmax3).to(device)
total_params = sum(p.numel() for p in model.parameters())

print(model)

#X = torch.rand(s1,s2,s3, 2, device=device)
#logits = torch.vmap(model)(alist)


costf = TotalCost(xs, ps,  ts, dx, dp, dt, s1p, s2p, s3p).to(device)
#costv = costf(logits,alist)

l0=1e-2
epochs = 2000
optimizer = torch.optim.Adam(model.parameters(),lr = l0)
iterations = epochs*(Nalr*Nali//batch_size)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

datlist = DataLoader(alist, batch_size = batch_size, shuffle = True)

Ttime = time.time()

for t in range(1,epochs+1):
    stime = time.time()
      #if t % 100 == 0:
      #l0 = l0/2
      #optimizer = torch.optim.Adam(model.parameters(),lr = l0)
    trainingloop(model, datlist, costf, optimizer, scheduler, varlist)
    dtime = time.time()-stime
    print (t,round(dtime,3))
Ttime = time.time()-Ttime
print('Total: ', round(Ttime,3))

fig, axs = plt.subplots(2,1,figsize=(9,9),sharex='all')
model.eval()

xvnp = xv.cpu().numpy()
pvnp = pv.cpu().numpy()
tvnp = tv.cpu().numpy()

xsnp = xs.cpu().numpy()
psnp = ps.cpu().numpy()
tsnp = ts.cpu().numpy()

tempindex=35
tind= 30
chcklist=np.zeros((Nalr*Nali+Nfock, s1-2*s1p, s2-2*s2p, s3-s3p))
with torch.no_grad():
  blist = torch.cat((alist[tempindex],varlist[0]),dim=-1)
  u =model(blist)[:,:,:,0]#*tv/(tf)+(1-tv/(tf))*alist[tempindex][:,:,:,0]
unp = u.detach().cpu().numpy()
#avalnp = alist[tempindex].cpu().numpy()

lwd=3

maxu=np.max(unp[s1p:-s1p,s2p:-s2p,tind])
minu=np.min(unp[s1p:-s1p,s2p:-s2p,tind])
divnorm=colors.TwoSlopeNorm(vmin=minu, vcenter=(minu+maxu)/2, vmax=maxu)
im1=axs[0].contourf(xvnp[s1p:-s1p,s2p:-s2p,tind],pvnp[s1p:-s1p,s2p:-s2p,tind],unp[s1p:-s1p,s2p:-s2p,tind],levels=10,cmap='RdBu', norm=divnorm)
cb1=plt.colorbar(im1,ax=axs[0])
cb1.ax.tick_params(labelsize=12)

xv1=xvnp*np.cos(tvnp)-pvnp*np.sin(tvnp)
pv1=pvnp*np.cos(tvnp)+xvnp*np.sin(tvnp)
n=0
for i in range(Nalr):
  alr = alrs[i].item()
  for j in range(Nali):
    ali = alis[j].item()
    #print(sigmav)
    #print (alr, ali)
    chcklist[n,:,:,:]=np.exp(-(xv1-np.sqrt(2)*alr)**2-(pv1-np.sqrt(2)*ali)**2)[s1p:-s1p,s2p:-s2p,:-s3p]/np.pi
    n+=1

if (Nfock>0):
  for mode in range(Nfock):
    chcklist[Nalr*Nali+mode,:,:,:] = alist[Nalr*Nali+mode,s1p:-s1p,s2p:-s2p,:-s3p,0].cpu().numpy()

#tind=0
u1=chcklist[tempindex]
maxu1=np.max(u1[:,:,tind])
minu1=np.min(u1[:,:,tind])
if (maxu1>minu1):
  divnorm=colors.TwoSlopeNorm(vmin=minu1, vcenter=(minu1+maxu1)/2, vmax=maxu1)
im2=axs[1].contourf(xvnp[s1p:-s1p,s2p:-s2p,tind],pvnp[s1p:-s1p,s2p:-s2p,tind],u1[:,:,tind],levels=10,cmap='RdBu', norm=divnorm)
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
