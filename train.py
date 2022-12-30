import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import MultiStepLR
from metrics import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
from data_prepare import dataset, trainloader, testloader
from model import GCNN

def train(model, device, trainloader, optimizer, epoch):
    print(f'Training on {len(trainloader)} samples.....')
    model.train()
    loss_func = nn.CrossEntropyLoss()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    scheduler = MultiStepLR(optimizer, milestones=[1,5], gamma=0.5)
    torch.autograd.set_detect_anomaly(True)
    for count, (prot, label) in enumerate(trainloader):
        prot = prot.to(device)
        optimizer.zero_grad()
        output = model(prot)

        predictions = torch.cat((predictions,output.cpu()), 0)
        labels = torch.cat((labels, label.cpu()), 0)

        print(output)
        print(label)
        # exit()

        loss = loss_func(output, label.float().to(device))
        loss.backward()
        optimizer.step()
    scheduler.step()
    labels = labels.detach().numpy()
    predictions = predictions.detach().numpy()
    acc = get_accuracy(labels, predictions)
    print(f"Epoch: {epoch} Loss: {loss} Accuracy: {acc}")

model = GCNN()
model.to(device)
num_epochs = 10
optimizer =  torch.optim.Adam(model.parameters(), lr= 0.1)

for epoch in tqdm(range(num_epochs)):
    train(model, device, trainloader, optimizer, epoch)
    torch.save(model.state_dict(), "/Users/benjamin/Documents/Classification/Graph-redo/GCN.pth")
    