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


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )



class NeuralOperator2D(nn.Module):
    def __init__(self, da, dv, dout,s1,s2,kmax1,kmax2):
        super(NeuralOperator2D, self).__init__()
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
        u = self.Project(vt4)
        
        return u
        #vt1 = 
        
model = NeuralOperator2D(2,64,1,28,24,6,6).to(device)
total_params = sum(p.numel() for p in model.parameters())

print(model)
        
X = torch.rand(28, 24, 2, device=device)
logits = model(X)
#pred_probab = nn.Softmax(dim=1)(logits)
#y_pred = pred_probab.argmax(1)
#print(f"Predicted class: {y_pred}")