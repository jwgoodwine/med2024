import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MeanSquaredError
from sklearn import preprocessing
from enum import Enum 
import copy
from pytorch_lightning import LightningModule, Trainer 
from lightning.pytorch.loggers import TensorBoardLogger
import control as ct
import numfracpy as nfr
import scipy.special as special
import matplotlib as mpl

mpl.use('pgf')

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from torchviz import make_dot
import collections

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# create data set
print(f"creating dataset")
N = 50000
tfinish = 10
t = np.linspace(0,tfinish,101)
solns = np.zeros(shape=(2*N,101))
orders = np.zeros(shape=(2*N,1))
for n in range(2*N):
    if n%1000 == 0:
        print(n)
    if np.random.uniform(0,1,1) < 0.5:
        den = np.random.uniform(0.01, 4, 3)
        den[1] = 0
        sys = ct.TransferFunction(den[2],den)
        t, yout = ct.step_response(sys,t)
        solns[n] = yout
        orders[n] = 2
    else:
        den = np.random.uniform(0.01, 4, 2)
        sys = ct.TransferFunction(den[1],den)
        y, yout = ct.step_response(sys,t)
        solns[n] = yout
        orders[n] = 1

# print how many first and second order responses were generated
print(f"number of first order transfer functions")
print((orders < 1.5).sum())
print(f"number of second order transfer functions")
print((orders > 1.5).sum())

# convert matricies to the right data type for nnet input and output
solns = torch.from_numpy(solns).to(torch.float32)
orders = torch.from_numpy(orders).to(torch.float32)
orders = torch.transpose(orders,1,0).reshape(-1)

# Make simple Enum for code clarity
class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3

# Again create a Dataset but this time, do the split in train test val
class CustomStarDataset(Dataset):
    def __init__(self):
        # load data and shuffle, befor splitting
        self.df = solns
        train_split=0.6
        val_split = 0.8
        self.df_labels = orders
        # split pointf.df
        self.train, self.val, self.test = np.split(self.df, [int(train_split*len(self.df)), int(val_split*len(self.df))])
        self.train_labels, self.val_labels, self.test_labels = np.split(self.df_labels, [int(train_split*len(self.df)), int(val_split*len(self.df))])
        # do the feature scaling only on the train set!
        self.scaler=preprocessing.StandardScaler().fit(self.train)
        for data_split in [ self.train, self.val, self.test]:
            data_split=self.scaler.transform(data_split)
        # convet labels to 1 hot
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx],self.labels[idx]
    
    def set_fold(self,set_type):
        # Make sure to call this befor using the dataset
        if set_type==DatasetType.TRAIN:
            self.dataset,self.labels=self.train,self.train_labels
        if set_type==DatasetType.TEST:
            self.dataset,self.labels=self.test,self.test_labels
        if set_type==DatasetType.VAL:
            self.dataset,self.labels=self.val,self.val_labels
        # Convert the datasets and the labels to pytorch format
        # Also use the StdScaler on the training set
        self.dataset = torch.tensor(self.scaler.transform(self.dataset)).float()

        return self
        
dataset = CustomStarDataset()
train = copy.deepcopy(dataset).set_fold(DatasetType.TRAIN)
test = copy.deepcopy(dataset).set_fold(DatasetType.TEST)
val = copy.deepcopy(dataset).set_fold(DatasetType.VAL)

# Define Batch Size
BATCH_SIZE=len(train)

# Defin a SimpleLightning Model
class SimpleModel(LightningModule):
    def __init__(self,train,test,val):
        super().__init__()
        self.train_ds=train
        self.val_ds=val
        self.test_ds=test
        # Define PyTorch model
        classes=1
        features=101
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0,1),
            nn.Linear(16, classes),
        )
        #self.accuracy = MeanSquaredError()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted_order = self(x).reshape(-1)
        return F.mse_loss(predicted_order, y)
    
    # Make use of the validation set
    def validation_step(self, batch, batch_idx, print_str="val"):
        x, y = batch
        predicted_order = self(x).reshape(-1)
        loss = F.mse_loss(predicted_order, y)
        #preds = logits.reshape(-1)
        #self.accuracy(predicted_order, y)

        # Calling self.log will surface up scalers for you in TensorBoard
        self.log(f"{print_str}_loss", loss, prog_bar=True)
        #self.log(f"{print_str}_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx,print_str='test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    # HERE: We define the 3 Dataloaders, only train needs to be shuffled
    # This will then directly be usable with Pytorch Lightning to make a super quick model
    def train_dataloader(self):
        return DataLoader(self.train_ds,batch_size=BATCH_SIZE,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size=BATCH_SIZE,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=BATCH_SIZE,shuffle=False)

logger = TensorBoardLogger("tb_logs", name="my_model")
# Start the Trainer
trainer = Trainer(
    max_epochs=1000,
    logger=logger
)
# Define the Model
model=SimpleModel(train,test,val).to(device)
# Train the Model
trainer.fit(model)
# Test on the Test SET, it will print validation
trainer.test()

# hand validation section: first check second and first order inputs
scaler=preprocessing.StandardScaler().fit(solns)
for n in range(5):
    thissoln = np.zeros(shape=(1,101))
    den = np.random.uniform(0.01, 4, 3)
    den[1] = 0
    t, yout = ct.step_response(ct.TransferFunction(den[2],den),t)
    thissoln[0] = yout
    thissoln=scaler.transform(thissoln)
    thissoln = torch.from_numpy(thissoln).to(torch.float32)
    print(model.forward(thissoln))

for n in range(5):
    thissoln = np.zeros(shape=(1,101))
    den = np.random.uniform(0.01, 4, 2)
    t, yout = ct.step_response(ct.TransferFunction(den[1],den),t)
    thissoln[0] = yout
    thissoln=scaler.transform(thissoln)
    thissoln = torch.from_numpy(thissoln).to(torch.float32)
    print(model.forward(thissoln))

## make the plot for the paper
#yhat = model.forward(thissoln)
#dot = make_dot(yhat,params=dict(list(model.named_parameters())))
#print(dot)
#dot.format='png'
#dot.render("viz.png")

# now see if fractional is identified properly
Initial = [0,0]
Interv = [0,tfinish]
dx = 0.01

NN = 1000
predictions = np.zeros(shape=(NN,2))
fig = plt.figure(figsize=(8,5))

for n in range(NN):
  print(n)
  gain = np.random.uniform(5,9)
  alpha = np.random.uniform(1.01,2)
  def f(t,u):
    return (1 - u)*gain

  print(alpha)
  thissoln = np.zeros(shape=(1,101))
  NumSol = nfr.FODE(f,Initial,Interv,dx,alpha)
  thissoln[0] = np.interp(t,NumSol[0],NumSol[1])
  plt.plot(NumSol[0], NumSol[1], label=r'$\alpha='+str(round(alpha,2))+'$',linewidth=1.5)
  thissoln=scaler.transform(thissoln)
  thissoln = torch.from_numpy(thissoln).to(torch.float32)
  out = model.forward(thissoln).detach().numpy()
  out = out[0,0]
  print(out)
  predictions[n,:] = [alpha,out]

plt.legend(loc=(1,0.2),shadow=False,ncol=1,prop={'size':16},frameon=False)
plt.grid(True)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel(r'$t$',size=23)
plt.ylabel(r'$u(t)$',size=23)
plt.show()

width = 3.5
height = width*(5**.5-1)/2
height = 1.1*height
fit = plt.figure(figsize=(width,height))

plt.scatter(predictions[:,0],predictions[:,1],marker=".")
plt.grid(True)
regr = linear_model.LinearRegression()
regr.fit(predictions[:,0].reshape(-1,1),predictions[:,1].reshape(-1,1))
linpredict = regr.predict(predictions[:,0].reshape(-1,1))
print(mean_squared_error(predictions[:,1].reshape(-1,1),linpredict))
print(r2_score(predictions[:,1].reshape(-1,1),linpredict))
plt.plot(predictions[:,0],linpredict,color="k")
plt.xlabel("Input order, $\\alpha$")
plt.ylabel("Predicted order")
plt.tight_layout()
plt.savefig("predicted.pgf",format="pgf")

