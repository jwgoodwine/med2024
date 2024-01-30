import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchmetrics import MeanSquaredError
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
from enum import Enum 
import copy
from pytorch_lightning import LightningModule, Trainer 
import control as ct
import numfracpy as nfr
import scipy.special as special
import matplotlib.pyplot as plt

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
plt.ion()
# create data set
print(f"creating dataset")
N = 10
t = np.linspace(0,50,101)
solns = np.zeros(shape=(2*N,101))
orders = np.zeros(shape=(2*N,1))

fig = plt.figure(figsize=(8,5))
for n in range(2*N):
    if n%1000 == 0:
        print(n)
    if np.random.uniform(0,1,1) < 0.5:
        den = np.random.uniform(0.01, 4, 3)
        den[1] = 0
        sys = ct.TransferFunction(den[2],den)
        t, yout = ct.step_response(sys,t)
        plt.plot(t,yout)
    else:
        den = np.random.uniform(0.1, 4, 2)
        sys = ct.TransferFunction(den[1],den)
        y, yout = ct.step_response(sys,t)
        plt.plot(t,yout)

plt.show()

Initial1 = [0]
Initial2 = [0,0]
Interv = [0,50]
dx = 0.05

print(f"Doing fractional")
fig = plt.figure(figsize=(8,5))
for gain in range(5,6):
    def f(t,u):
        return (1 - u)*gain

    for alpha in [0.5, 0.75, 1.01, 1.25, 1.5, 1.75, 1.99]:
        print(alpha)
        if alpha < 1:
          NumSol = nfr.FODE(f,Initial1,Interv,dx,alpha)
        else:
          NumSol = nfr.FODE(f,Initial2,Interv,dx,alpha)
        plt.plot(NumSol[0], NumSol[1], label=r'$\alpha='+str(round(alpha,2))+'$',linewidth=1.5)

plt.legend(loc=(1,0.2),shadow=False,ncol=1,prop={'size':16},frameon=False)
plt.grid(True)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel(r'$t$',size=23)
plt.ylabel(r'$u(t)$',size=23)
plt.show()

