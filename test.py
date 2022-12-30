import torch 
from metrics import *
from data_prepare import testloader
import torch.nn as nn
from model import GCNN

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = GCNN()
model.load_state_dict(torch.load("/Users/benjamin/Documents/Classification/Graph-redo/GCN.pth"))
model.to(device)
model.eval()
predictions = torch.Tensor()
labels = torch.Tensor()

with torch.no_grad():
    for prot, label in testloader:
        prot_1 = prot.to(device)
        output = model(prot)
        print(label)
        print(output)
        predictions = torch.cat((predictions, output.cpu()), 0)
        labels = torch.cat((labels, label.cpu()), 0)
labels = torch.tensor(labels.numpy())
predictions = torch.tensor(predictions.numpy())



loss_func = nn.CrossEntropyLoss()
loss = loss_func(labels, predictions)
acc = get_accuracy(labels, predictions)

print(f'loss : {loss}')
print(f'Accuracy : {acc}')