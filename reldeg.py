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

# create data set
print(f"creating dataset")
N = 50000
t = np.linspace(0,50,101)
solns = np.zeros(shape=(2*N,101))
orders = np.zeros(shape=(2*N,1))
for n in range(2*N):
    if n%1000 == 0:
        print(n)
    if np.random.uniform(0,1,1) < 0.5:
        den = np.random.uniform(0.01, 4, 3)
        sys = ct.TransferFunction(den[2],den)
        t, yout = ct.step_response(sys,t)
        solns[n] = yout
        orders[n] = 2
    else:
        den = np.random.uniform(0.1, 4, 2)
        sys = ct.TransferFunction(den[1],den)
        y, yout = ct.step_response(sys,t)
        solns[n] = yout
        orders[n] = 1

# convert matricies to the right data type for nnet input and output
solns = torch.from_numpy(solns)
solns = solns.to(torch.float32)
orders = torch.from_numpy(orders)
orders = orders.to(torch.float32)
#orders = orders.to(torch.long)
orders = torch.transpose(orders,1,0).reshape(-1)
#orders = torch.transpose(orders,1,0)

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
        #self.labels = torch.tensor(self.scaler.transform(self.labels)).float()
        #self.labels=torch.tensor(self.labels.reshape(-1)).long()

        return self
        
dataset = CustomStarDataset()

train = copy.deepcopy(dataset).set_fold(DatasetType.TRAIN)
test = copy.deepcopy(dataset).set_fold(DatasetType.TEST)
val = copy.deepcopy(dataset).set_fold(DatasetType.VAL)

# Define Batch Size
BATCH_SIZE=2*N

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
        #self.accuracy = Accuracy(task="binary")
        self.accuracy = MeanSquaredError()
    # Same as above
    def forward(self, x):
        x = self.model(x)
        #return F.log_softmax(x, dim=1)
        return x
    
    # Same as above
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).reshape(-1)
        #loss = F.nll_loss(logits, y)
        loss = F.mse_loss(logits, y)
        
        return loss
    
    # Make use of the validation set
    def validation_step(self, batch, batch_idx, print_str="val"):
        x, y = batch
        logits = self(x).reshape(-1)
        #loss = F.nll_loss(logits, y)
        loss = F.mse_loss(logits, y)
        #preds = torch.argmax(logits, dim=1)
        preds = logits.reshape(-1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{print_str}_loss", loss, prog_bar=True)
        self.log(f"{print_str}_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx,print_str='test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    #
    # HERE: We define the 3 Dataloaders, only train needs to be shuffled
    # This will then directly be usable with Pytorch Lightning to make a super quick model
    def train_dataloader(self):
#        return DataLoader(self.train_ds,batch_size=BATCH_SIZE,num_workers=4,persistent_workers=True,shuffle=True)
        return DataLoader(self.train_ds,batch_size=BATCH_SIZE,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size=BATCH_SIZE,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=BATCH_SIZE,shuffle=False)

# Start the Trainer
trainer = Trainer(
    max_epochs=1000,
)
# Define the Model
model=SimpleModel(train,test,val).to(device)
# Train the Model
trainer.fit(model)
# Test on the Test SET, it will print validation
trainer.test()


# hand validation section
for n in range(5):
    thissoln = np.zeros(shape=(1,101))
    den = np.random.uniform(0.01, 4, 3)
    t, yout = ct.step_response(ct.TransferFunction(den[2],den),t)
    thissoln[0] = yout
    thissoln = torch.from_numpy(thissoln)
    thissoln = thissoln.to(torch.float32) 
    scaler=preprocessing.StandardScaler().fit(solns)
    thissoln=scaler.transform(thissoln)
    thissoln = torch.from_numpy(thissoln)
    thissoln = thissoln.to(torch.float32) 
    print(model.forward(thissoln))

for n in range(5):
    thissoln = np.zeros(shape=(1,101))
    den = np.random.uniform(0.01, 4, 2)
    t, yout = ct.step_response(ct.TransferFunction(den[1],den),t)
    thissoln[0] = yout
    thissoln = torch.from_numpy(thissoln)
    thissoln = thissoln.to(torch.float32)
    scaler=preprocessing.StandardScaler().fit(solns)
    thissoln=scaler.transform(thissoln)
    thissoln = torch.from_numpy(thissoln)
    thissoln = thissoln.to(torch.float32) 
    print(model.forward(thissoln))

Initial = [0,0]
Interv = [0,50]
dx = 0.01

fig = plt.figure(figsize=(8,5))
for gain in range(3,7):
    def f(t,u):
        return (1 - u)/gain

    for alpha in [1.01, 1.25, 1.5, 1.75, 1.99]:
        print(alpha)
        thissoln = np.zeros(shape=(1,101))
        NumSol = nfr.FODE(f,Initial,Interv,dx,alpha)
        thissoln[0] = np.interp(t,NumSol[0],NumSol[1])
        plt.plot(NumSol[0], NumSol[1], label=r'$\alpha='+str(round(alpha,2))+'$',linewidth=1.5)
        thissoln = torch.from_numpy(thissoln)
        thissoln = thissoln.to(torch.float32)
        scaler=preprocessing.StandardScaler().fit(solns)
        thissoln=scaler.transform(thissoln)
        thissoln = torch.from_numpy(thissoln)
        thissoln = thissoln.to(torch.float32) 
        print(model.forward(thissoln))

plt.legend(loc=(1,0.2),shadow=False,ncol=1,prop={'size':16},frameon=False)
plt.grid(True)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel(r'$t$',size=23)
plt.ylabel(r'$u(t)$',size=23)
plt.show()

