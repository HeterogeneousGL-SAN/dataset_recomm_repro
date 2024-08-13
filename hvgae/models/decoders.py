import torch
from torch.nn import Module
from torch import Tensor
import random
import numpy as np


class InnerProductDecoderAdj(Module):
    def __init__(self):
        super(InnerProductDecoderAdj, self).__init__()

    def forward(self, z, sigmoid: bool = True):
        logits = torch.matmul(z, z.t())
        if sigmoid:
            return torch.sigmoid(logits)
        return logits

class InnerProductDecoderTen(Module):
    def __init__(self):
        super(InnerProductDecoderTen, self).__init__()

    def forward(self, z: Tensor, edge_idx: Tensor, sigmoid: bool = True):
        adj = (z[edge_idx[0]] * z[edge_idx[1]]).sum(dim=1)
        if sigmoid:
            return torch.sigmoid(adj)
        return adj



class DatasetDecoderInnerProductDecoderAdj(Module):
    def __init__(self):
        super(DatasetDecoderInnerProductDecoderAdj,self).__init__()


    def forward(self, zd, zp, sigmoid: bool = True):
        logits = torch.matmul(zp, zd.t())
        if sigmoid:
            return torch.sigmoid(logits)

        return logits



class DatasetDecoderInnerProductDecoderTen(Module):
    def __init__(self):
        super(DatasetDecoderInnerProductDecoderTen,self).__init__()


    def forward(self, z: Tensor,zd: Tensor, edge_idx: Tensor, sigmoid: bool = True):
        logits = (z[edge_idx[0]] * zd[edge_idx[1]]).sum(dim=1)
        if sigmoid:
            return torch.sigmoid(logits)

        return logits