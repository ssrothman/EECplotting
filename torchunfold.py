import torch
import torch.nn as nn
import lightning as L
import numpy as np

def multinorm_likelihood(fw, reco, invcov):
    diff = fw - reco
    #print("diff = ", diff)

    #print("Shapes")
    #print(diff.T.shape)
    #print(invcov.shape)
    #print(diff.shape)
    #print((diff.T @ invcov).shape)
    #print((diff.T @ invcov @ diff).shape)
    #print()

    #like = torch.sum(torch.square(diff))
    like = 0.5 * torch.dot(diff,  invcov @ diff)

    #print("like = ", like)

    return like

class TorchUnfolding(L.LightningModule):
    def __init__(self, invcov, transfer, recopure):
        super(TorchUnfolding, self).__init__()

        self.invcov = torch.Tensor(invcov)
        self.transfer = torch.Tensor(transfer)
        self.recopure = torch.sum(self.transfer, axis=0)

        self.unfolded = torch.nn.Parameter(torch.zeros_like(self.recopure))
        self.unfolded.data = torch.Tensor(recopure)

    def forward(self, x):
        #print()
        #print("FORWARD")
        #print("x = ", x)
        #print("self.unfolded = ", self.unfolded)
        #print("self.transfer = ", self.transfer)

        fw = torch.matmul(self.transfer, self.unfolded)
        #print("fw = ", fw)

        return multinorm_likelihood(fw, x, self.invcov)
    
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        self.log("train_loss", loss)

        return -loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1)

'''
size = 5

transfer = torch.Tensor(np.eye(size))
recopure = torch.Tensor(np.random.random((size)))
cov = torch.Tensor(np.eye(size))

mod = TorchUnfolding(cov, transfer, recopure)
opt = torch.optim.Adam(mod.parameters(), lr=1e-2)

from tqdm import tqdm
t = tqdm(range(1000))

for epoch in t:
    loss = mod.forward(recopure)
    loss.backward()
    opt.step()
    opt.zero_grad()
    t.set_description(f"Loss: {loss.item()}")
    
print(mod.unfolded.data.numpy())
print(recopure.data.numpy())

'''
