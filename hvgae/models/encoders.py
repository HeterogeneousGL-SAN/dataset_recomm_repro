from torch.nn import Linear, Module, ReLU, Dropout

from torch_geometric.nn import SAGEConv, Sequential, GCNConv
from torch_geometric.nn.models import GraphSAGE
import torch_geometric.transforms as T
import torch.nn.init as init
import random
import numpy as np
import torch
MAX_LOGSTD = 10


# sigir
class Encoder_Citation_Network(Module):
    def __init__(self, in_dims: int, hid1_dims: int, hid2_dims: int, out_dims: int):
        super(Encoder_Citation_Network, self).__init__()

        self.conv_sage = GraphSAGE(in_dims, hid1_dims, 2, hid2_dims)

        self.conv_gcn = Sequential("x, edge_idx", [
            (GCNConv(in_dims, hid1_dims), "x, edge_idx -> x1a"),
            (GCNConv(hid1_dims, hid2_dims), "x1a, edge_idx -> x2"),
        ])
        self.conv_linear = Sequential("x", [
            (Linear(in_dims, hid1_dims), "x -> x1a"),
            (Linear(hid1_dims, hid2_dims), "x1a -> x2"),
        ])

        self.conv_mu = Sequential("x", [
            (Linear(hid2_dims, out_dims), "x -> x1"),
            # (ReLU(), "x1 -> x2"),
        ])

        for layer in self.conv_mu.children():
            if isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)


        self.conv_logvar = Sequential("x", [
            (Linear(hid2_dims, out_dims), "x -> x1"),
            # (ReLU(), "x1 -> x2"),
        ])


        for layer in self.conv_logvar.children():
            if isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)




    def forward(self, x,edge_index,type='sage'):
        if type == 'linear':
            x_hat = self.conv_linear(x)
        elif type == 'gcn':
            x_hat = self.conv_gcn(x,edge_index)
        else:
            x_hat = self.conv_sage(x,edge_index)


        return self.conv_mu(x_hat), self.conv_logvar(x_hat)




class Encoder_Datasets_Publications_GCN(Module):
    def __init__(self, in_dims: int, hid1_dims: int, hid2_dims: int, out_dims: int):
        super(Encoder_Datasets_Publications_GCN, self).__init__()

        # self.conv = GraphSAGE(-1,hid1_dims,3,hid2_dims)
        self.conv = Sequential("x,edge_index", [
            (GCNConv(in_dims, hid1_dims), "x, edge_index -> x1a"),
            # (ReLU(), "x1 -> x1a"),
            (GCNConv(hid1_dims, hid2_dims), "x1a, edge_index -> x2"),
            # (ReLU(), "x2 -> x2a"),
        ])


        self.conv_mu = Sequential("x2a", [
            (Linear(hid2_dims, out_dims), "x2a -> x3"),
            # (ReLU(), "x3 -> x4"),
        ])

        for layer in self.conv_mu.children():
            if isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)

        self.conv_log_var = Sequential("x2a", [
            (Linear(hid2_dims, out_dims), "x2a -> x5"),
            # (ReLU(), "x5 -> x6"),
        ])

        for layer in self.conv_log_var.children():
            if isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, x,edge_index):
        x_hat = self.conv(x,edge_index)
        return self.conv_mu(x_hat), self.conv_log_var(x_hat)



class Encoder_Datasets_Publications(Module):
    def __init__(self, in_dims: int, hid1_dims: int, hid2_dims: int, out_dims: int):
        super(Encoder_Datasets_Publications, self).__init__()
        self.conv = Sequential("x", [
            (Linear(in_dims, hid1_dims), "x -> x1"),
            # (ReLU(), "x1 -> x1a"),
            (Linear(hid1_dims, hid2_dims), "x1 -> x2"),
            # (ReLU(), "x2 -> x2a"),
        ])
        for layer in self.conv.children():
            if isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)


        self.conv_mu = Sequential("x2a", [
            (Linear(hid2_dims, out_dims), "x2a -> x3"),
            # (ReLU(), "x3 -> x4"),
        ])

        for layer in self.conv_mu.children():
            if isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)

        self.conv_log_var = Sequential("x2a", [
            (Linear(hid2_dims, out_dims), "x2a -> x5"),
            # (ReLU(), "x5 -> x6"),
        ])

        for layer in self.conv_log_var.children():
            if isinstance(layer, Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x_hat = self.conv(x)
        return self.conv_mu(x_hat), self.conv_log_var(x_hat)



