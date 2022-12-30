import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool as gep, dense
import numpy as np

class GCNN(nn.Module):
    def __init__(self, output = 5, num_features = 21, output_dim = 128, dropout = 0.3):
        super(GCNN, self).__init__()

        print("GCNN Loaded")
        self.batch = None
        self.output = output

        self.conv1 = GCNConv(num_features, num_features)
        self.conv2 = GCNConv(num_features, num_features)
        self.conv3 = GCNConv(num_features, num_features)
        self.pro1_fc1 = dense.Linear(num_features, output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

        self.fc1 = nn.Linear(output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, self.output)

    def forward(self, prot):
        prot_x, prot_edge_index, prot_dist, prot_batch = prot.x, prot.edge_index, prot.dist, prot.batch

        self.batch = prot_batch
        x = self.conv1(prot_x, prot_edge_index, prot_dist)
        x = x.to(torch.float32)

        x = self.conv2(x, prot_edge_index, prot_dist)
        x = x.to(torch.float32)

        x = self.conv3(x, prot_edge_index, prot_dist)
        x = x.to(torch.float32)

        x = F.relu(x)

        x = gep(x, prot_batch)   
        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        out = self.out(x)
        out = self.softmax(out)
        # print(out)
       
        # max_index = out.argmax(axis=1)
        # out[np.arange(out.shape[0]), max_index] = 1
        # out[out != 1] = 0
        return out

net = GCNN()
print(net)