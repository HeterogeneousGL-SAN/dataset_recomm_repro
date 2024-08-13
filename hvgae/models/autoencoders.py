from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.utils import negative_sampling, to_dense_adj

from models.encoders import Encoder_Citation_Network,Encoder_Datasets_Publications,Encoder_Datasets_Publications_GCN
from models.decoders import DatasetDecoderInnerProductDecoderTen
from torch_geometric.nn.models import InnerProductDecoder
EPS = 1e-15
MAX_LOGSTD = 10

num_papers = 1358
num_datasets = 2344


class GVAEPaperAutoEncoder(Module):
    def __init__(self, in_dims: int, hid1_dims: int, hid2_dims: int, out_dims: int):
        super(GVAEPaperAutoEncoder, self).__init__()
        self.encoder = Encoder_Citation_Network(in_dims, hid1_dims, hid2_dims, out_dims)
        self.decoder = InnerProductDecoder()

    def encode_paper(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z,self.__mu__,self.__logstd__

    def decode(self, *args, **kwargs) -> Tensor:
        return self.decoder(*args, **kwargs)

    def reparametrize(self, mu: Tensor, logstd: Tensor): # taken from GVAE class of torch
        eps = torch.randn_like(logstd)
        rep = mu + eps * torch.exp(logstd)
        return rep

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu

        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def forward(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z, self.__mu__, self.__logstd__


class GVAEBipartiteNetAutoEncoder(Module):
    def __init__(self, in_dims: int,  hid1_dims: int,  hid2_dims: int, out_dims: int):
        super(GVAEBipartiteNetAutoEncoder, self).__init__()
        self.encoder = Encoder_Datasets_Publications(in_dims, hid1_dims, hid2_dims, out_dims)
        self.encoder_gcn = Encoder_Datasets_Publications_GCN(in_dims, hid1_dims, hid2_dims, out_dims)
        self.decoder = DatasetDecoderInnerProductDecoderTen()

    def encode_dataset(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        zd = self.reparametrize(self.__mu__, self.__logstd__)
        return zd,self.__mu__,self.__logstd__


    def encode_dataset_gcn(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder_gcn(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        zd = self.reparametrize(self.__mu__, self.__logstd__)
        return zd,self.__mu__,self.__logstd__

    def decode(self, z, zd, *args, **kwargs) -> Tensor:
        return self.decoder(z,zd*args, **kwargs)

    def reparametrize(self, mu: Tensor, logstd: Tensor):  # taken from GVAE class of torch
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def recon_loss(self, z: Tensor, zp: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        pos_loss = -torch.log(
            self.decoder(zp, z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            sources = neg_edge_index[0]
            targets = neg_edge_index[1]
            news, newt = [],[]
            for i in range(len(sources)):
                to_excl = False
                if sources[i] >= num_papers or targets[i] >= num_datasets:
                    to_excl = True

                if not to_excl:
                    news.append(sources[i])
                    newt.append(targets[i])
            neg_edge_index = torch.tensor([news, newt], dtype=torch.long)

        neg_loss = -torch.log(1 -
                              self.decoder(zp, z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu

        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def forward(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z


class GVAEMLPAutoEncoder(Module):
    def __init__(self, in_dims: int, hid1_dims: int, hid2_dims: int, out_dims: int, dropout: float = 0.5):
        super(GVAEMLPAutoEncoder, self).__init__()
        self.encoder = Encoder_Datasets_Publications(in_dims, hid1_dims, hid2_dims, out_dims)
        self.decoder = InnerProductDecoder()

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z,self.__mu__,self.__logstd__

    def decode(self, *args, **kwargs) -> Tensor:

        return self.decoder(*args, **kwargs)

    def reparametrize(self, mu: Tensor, logstd: Tensor): # taken from GVAE class of torch
        eps = torch.randn_like(logstd)
        rep = mu + eps * torch.exp(logstd)
        return rep

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu

        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def forward(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z, self.__mu__, self.__logstd__


class GVAEMLPAutoEncoderGCN(Module):
    def __init__(self, in_dims: int, hid1_dims: int, hid2_dims: int, out_dims: int):
        super(GVAEMLPAutoEncoderGCN, self).__init__()
        self.encoder = Encoder_Datasets_Publications_GCN(in_dims, hid1_dims, hid2_dims, out_dims)
        self.decoder = InnerProductDecoder()

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z,self.__mu__,self.__logstd__

    def decode(self, *args, **kwargs) -> Tensor:

        return self.decoder(*args, **kwargs)

    def reparametrize(self, mu: Tensor, logstd: Tensor): # taken from GVAE class of torch
        eps = torch.randn_like(logstd)
        rep = mu + eps * torch.exp(logstd)
        return rep

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu

        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def forward(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z, self.__mu__, self.__logstd__




class GVAEAutoEncoder(Module):
    def __init__(self, in_dims: int, in_dims_1: int, hid1_dims: int, hid1_dims_1: int, hid2_dims: int, hid2_dims_1: int, out_dims: int):
        super(GVAEAutoEncoder, self).__init__()

        self.paper_gvae = GVAEPaperAutoEncoder(in_dims,hid1_dims,hid2_dims,out_dims)
        self.bip_gvae = GVAEBipartiteNetAutoEncoder(in_dims_1,hid1_dims_1,hid2_dims_1,out_dims)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def reparametrize(self, mu: Tensor, logstd: Tensor): # taken from GVAE class of torch
        eps = torch.randn_like(logstd)
        rep = mu + eps * torch.exp(logstd)
        return rep



    def paper_encoder(self,*args,**kwargs):
         return self.paper_gvae.encode_paper(*args,**kwargs)

    def dataset_encoder(self,*args,**kwargs):
         return self.bip_gvae.encode_dataset(*args,**kwargs)

    def dataset_encoder_gcn(self,*args,**kwargs):
         return self.bip_gvae.encode_dataset_gcn(*args,**kwargs)

    def paper_decoder(self,*args, **kwargs):
        return self.paper_gvae.decode(*args, **kwargs)

    def data_decoder(self,*args, **kwargs):
        return self.bip_gvae.decode(*args, **kwargs)

    def recon_loss_paper(self, *args, **kwargs):
        return self.paper_gvae.recon_loss(*args, **kwargs)

    def recon_loss_bip(self, *args, **kwargs):
        return self.bip_gvae.recon_loss(*args, **kwargs)

    def kl_loss_paper(self,*args,**kwargs):
        return self.paper_gvae.kl_loss(*args,**kwargs)

    def kl_loss_bip(self, *args, **kwargs):
        return self.bip_gvae.kl_loss(*args, **kwargs)

    def forward(self, x_paper, paper_num_nodes, edge_idx_paper, x_dataset,dataset_num_nodes, edge_idx_dataset, type,enriched=False):
        mu1, log_var1 = self.paper_gvae.encoder(x_paper, edge_idx_paper,'sage')
        z1 = self.reparametrize(mu1, log_var1).to(self.device)
        loss_paper = self.recon_loss_paper(z1, edge_idx_paper)
        loss_paper = loss_paper + (1 / paper_num_nodes) * self.kl_loss_paper(mu1,log_var1)


        if type != 'linear':
            mu2, log_var2 = self.paper_gvae.encoder(x_dataset,edge_idx_dataset,type)
            z2 = self.reparametrize(mu2, log_var2).to(self.device)
            z2 = z2[paper_num_nodes:,:]
        else:
            if enriched:
                mask = edge_idx_dataset[0] < paper_num_nodes
                edge_idx_dataset = edge_idx_dataset[:, mask]
                mask = edge_idx_dataset[1] >= paper_num_nodes
                edge_idx_dataset = edge_idx_dataset[:, mask]

                matrix = torch.zeros((dataset_num_nodes - paper_num_nodes, paper_num_nodes), dtype=torch.float).to(
                    self.device)
                matrix[edge_idx_dataset[1] - paper_num_nodes, edge_idx_dataset[0]] = 1  # assuming undirected graph

            else:
                matrix = torch.zeros((dataset_num_nodes - paper_num_nodes, paper_num_nodes), dtype=torch.float).to(
                    self.device)
                matrix[edge_idx_dataset[1] - paper_num_nodes, edge_idx_dataset[0]] = 1  # assuming undirected graph
            mu2, log_var2 = self.bip_gvae.encoder(matrix)
            z2 = self.reparametrize(mu2, log_var2).to(self.device)
            # if enriched:
            #     z2 = z2[paper_num_nodes:, :paper_num_nodes]


        sources,new_sources = edge_idx_dataset[0],[]
        targets,new_targets = edge_idx_dataset[1],[]
        for i in range(len(sources)):
            # exclude edges with incorrect source and target indices
            if (sources[i].item() < paper_num_nodes and targets[i].item() >= dataset_num_nodes - paper_num_nodes):
                new_sources.append(sources[i].item())
                new_targets.append(targets[i].item() - paper_num_nodes)
        edge_idx = torch.tensor([new_sources, new_targets], dtype=torch.long)
        loss_bip = self.recon_loss_bip(z2, z1, edge_idx)
        loss_bip = loss_bip + (1 / dataset_num_nodes) * self.kl_loss_bip(mu2,log_var2)
        loss = loss_paper + loss_bip

        return loss, loss_paper, loss_bip


