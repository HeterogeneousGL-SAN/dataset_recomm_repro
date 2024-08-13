import argparse
import os.path as osp
import time
import tqdm
import torch
import os
import json
import numpy as np
import torch.nn.functional as F
from loader import BipartiteDataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.data import Batch,InMemoryDataset
from utils import write_qrels,write_run
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Linear, VGAE, GCNConv, SAGEConv
from torch_geometric.utils import to_dense_adj
import random
from models.autoencoders import GVAEPaperAutoEncoder,GVAEAutoEncoder
from sklearn.metrics import ndcg_score
import pandas as pd




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['mes','pubmed_kcore','pubmed'],help="The selected dataset")
parser.add_argument('--vgae',type=str,default='cit_net_vae',choices=['hvgae','mlp_vae','gcn_vae','cit_net_vae',
                                                              'sage_vae_enriched','sage_vae','sage_vae_bip'])
parser.add_argument('--seed',default=42)
parser.add_argument('--path',default=str)
parser.add_argument('--type',default='linear')

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
g = torch.Generator()
g.manual_seed(args.seed)


def evaluation(rankings, edge_index,k = 10):
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0

    def ndcg_at_k(true_labels, predictions, k):
        relevance_scores = [1 if item in true_labels else 0 for item in predictions]
        dcg = dcg_at_k(relevance_scores, k)
        idcg = dcg_at_k([1] * len(true_labels), k)  # IDCG assuming all true labels are relevant
        if not idcg:
            return 0
        return dcg / idcg

    print(len(edge_index))
    print(len(rankings))
    all_datasets = []
    values = [v for k,v in edge_index.items()]

    for v in values:
        all_datasets.extend(v)
    all_datasets = list(set(all_datasets))

    precision, recall, ndcg = 0, 0, 0
    edge_index = dict(sorted(edge_index.items(), key=lambda x: x[0]))
    for index,i in enumerate(values):
        true_vals = i
        predicted = rankings[index]
        # print(len(predicted))
        # predicted = [p for p in predicted if p in all_datasets]
        # print(len(predicted))
        predicted = predicted[:k]
        correct = list(set(true_vals) & set(predicted))
        precision += len(correct) / k
        recall += len(correct) / len(true_vals)
        ndcg += ndcg_at_k(predicted, true_vals, k)
    return precision/len(edge_index), recall/len(edge_index), ndcg/len(edge_index)


def run():
    seed = args.seed

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    transform = T.Compose([
        T.ToDevice(device),
        T.remove_isolated_nodes.RemoveIsolatedNodes()

    ])
    print('TRAINING')
    # ./ datasets / {dataset} / split_transductive / train
    data = BipartiteDataset(root=f'{args.path}')
    train_dataset = data[0]
    num_training_papers = train_dataset['publication'].num_nodes
    # print(train_dataset)
    # print(num_training_papers)
    del train_dataset['publication', 'cites', 'dataset']
    del train_dataset['dataset', 'cites', 'dataset']
    del train_dataset['dataset']
    # print(train_dataset)
    train_citation_network = train_dataset.to_homogeneous().to(device)
    # print(train_citation_network)
    data = BipartiteDataset(root=f'{args.path}')
    train_bipartite_network = data[0]
    # print(train_bipartite_network)
    del train_bipartite_network['publication', 'cites', 'publication']
    del train_bipartite_network['dataset', 'cites', 'dataset']
    train_bipartite_network['publication', 'cites', 'dataset'].edge_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_train
    train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_train
    tups_train = train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_train = [tuple(e) for e in tups_train]
    train_bipartite_network = train_bipartite_network.to_homogeneous().to(device)
    # print(train_bipartite_network)


    print('VALIDATION')
    data = BipartiteDataset(root=f'{args.path}')
    validation_dataset = data[0]
    del validation_dataset['publication', 'cites', 'dataset']
    del validation_dataset['dataset', 'cites', 'dataset']
    del validation_dataset['dataset']
    # print(validation_dataset)
    validation_citation_network = validation_dataset.to_homogeneous().to(device)
    # print(validation_citation_network)

    data = BipartiteDataset(root=f'{args.path}')
    validation_bipartite_network = data[0]
    # print(validation_bipartite_network)
    del validation_bipartite_network['publication', 'cites', 'publication']
    del validation_bipartite_network['dataset', 'cites', 'dataset']
    validation_bipartite_network['publication', 'cites', 'dataset'].edge_index = validation_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_val
    validation_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = validation_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_val

    validation_bipartite_network = validation_bipartite_network.to_homogeneous().to(device)
    # print(validation_bipartite_network)

    print('TEST')
    data = BipartiteDataset(root=f'{args.path}')
    test_dataset = data[0]
    del test_dataset['publication', 'cites', 'dataset']
    del test_dataset['dataset', 'cites', 'dataset']
    del test_dataset['dataset']
    # print(test_dataset)
    test_citation_network = test_dataset.to_homogeneous().to(device)

    # print(test_citation_network)
    test_dataset = BipartiteDataset(root=f'{args.path}')
    test_dataset = test_dataset[0]
    # print(test_dataset)
    del test_dataset['publication', 'cites', 'publication']
    del test_dataset['dataset', 'cites', 'dataset']
    test_dataset['publication', 'cites', 'dataset'].edge_index = test_dataset[
        'publication', 'cites', 'dataset'].edge_index_test
    test_dataset['publication', 'cites', 'dataset'].edge_label_index = test_dataset[
        'publication', 'cites', 'dataset'].edge_label_index_test
    tups_test = test_dataset['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_test = [tuple(e) for e in tups_test]
    print(f'INTERSECTION: {len(list(set(tups_train).intersection(tups_test)))}')
    test_bipartite_network = test_dataset.to_homogeneous().to(device)
    # print(test_bipartite_network)

    in_channels, out_channels = data[0]['publication'].num_features, 16 # 16 was not stated in the paper
    print(in_channels,train_bipartite_network.num_nodes)
    model = GVAEAutoEncoder(in_channels, train_citation_network.num_nodes, hidden1, hidden2,hidden1_mlp,hidden2_mlp, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()
        loss = 0
        loss,_,_ = model(train_citation_network.x,num_training_papers,train_citation_network.edge_index,train_bipartite_network.x,train_bipartite_network.num_nodes,train_bipartite_network.edge_index,type)
        loss.backward()
        optimizer.step()
        return float(loss)

    def validation():
        model.eval()
        with torch.no_grad():
            loss,_,_ = model(validation_citation_network.x,num_training_papers,validation_citation_network.edge_index,validation_bipartite_network.x,validation_bipartite_network.num_nodes,validation_bipartite_network.edge_index,type)

        return float(loss)

    times=[]
    early_stopping_counter = 0
    max_count = 50
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train()
        # val_loss = validation()
        #
        print(f'Epoch: {epoch:03d}, LOSS ON TRAIN: {loss:.4f}')
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        #     best_model_state_dict = model.state_dict()
        # else:
        #     early_stopping_counter += 1
        # if early_stopping_counter == max_count:
        #     print('max count reached, exit')
        #     break


        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    model.eval()
    with torch.no_grad():
        z,mu,log = model.paper_encoder(test_citation_network.x, test_citation_network.edge_index)

        matrix = torch.zeros((test_bipartite_network.num_nodes - test_citation_network.num_nodes, test_citation_network.num_nodes), dtype=torch.float).to(device)
        matrix[test_bipartite_network.edge_index[1] - test_citation_network.num_nodes, test_bipartite_network.edge_index[0]] = 1  # assuming undirected graph

        zd,mu2,log2 = model.dataset_encoder(matrix)
        if type != 'linear':
            zd,mu2,log2 = model.paper_encoder(test_bipartite_network.x,test_bipartite_network.edge_index,type)
            zd = zd[num_training_papers:,:]
        # print(z.shape,zd.shape)
        # print(sorted(list(set(test_bipartite_network.edge_index[0].tolist()))))
        recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_bipartite_network.edge_label_index[0].tolist()))),:],zd.t()))

        # sorted_recon = torch.argsort(recon, dim=1,descending=True).tolist()
        top_values, top_indices = torch.topk(recon, k=20, dim=1)
        # sorted_recon = [t[:20] for t in sorted_recon]
        sorted_recon = top_indices.tolist()

        edge_index = test_bipartite_network.edge_label_index.t().tolist()
        edge_index = sorted(edge_index, key=lambda x: x[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # print(edge_index)
        edge_index = edge_index.tolist()
        edge_index_dict = {}
        for i, pair in enumerate(edge_index):
            if edge_index_dict.get(pair[0], None) is not None:
                edge_index_dict[pair[0]].append(pair[1] - test_citation_network.num_nodes)
            else:
                edge_index_dict[pair[0]] = [pair[1] - test_citation_network.num_nodes]
            edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))
        # print(edge_index_dict)
        edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))
        # print(edge_index_dict)

        # print(len(list(edge_index_dict.keys())))
        # print(edge_index_dict)
        # Sort the tensor along dimension 1 and get the indices

        # Check the sorted indices
        json_res = {}
        string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}_{hidden1_mlp}_{hidden2_mlp}_{type}'
        for k in [1, 5, 10]:
            print('RESULTS: ',k)
            precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
            print(ndcg, precision, recall)
            json_res[f'ndcg_{str(k)}'] = str(ndcg)
            json_res[f'precision_{str(k)}'] = str(precision)
            json_res[f'recall_{str(k)}'] = str(recall)
            print('\n\n')
        f = open(f'./hgvae_final_{string}_standard.json','w')
        json.dump(json_res,f,indent=4)


def run_reduced():
    seed = args.seed

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('TRAINING')
    data = BipartiteDataset(root=f'{args.path}')
    train_dataset = data[0]
    num_training_papers = train_dataset['publication'].num_nodes
    del train_dataset['publication', 'cites', 'dataset']
    del train_dataset['dataset', 'cites', 'dataset']
    del train_dataset['dataset']

    train_citation_network = train_dataset.to_homogeneous().to(device)
    data = BipartiteDataset(root=f'{args.path}')
    train_bipartite_network = data[0]
    del train_bipartite_network['publication', 'cites', 'publication']
    del train_bipartite_network['dataset', 'cites', 'dataset']

    train_bipartite_network['publication', 'cites', 'dataset'].edge_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_train_reduced
    train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_train_reduced

    tups_train = train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_train = [tuple(e) for e in tups_train]
    train_bipartite_network = train_bipartite_network.to_homogeneous().to(device)



    data = BipartiteDataset(root=f'{args.path}')
    test_dataset = data[0]
    del test_dataset['publication', 'cites', 'dataset']
    del test_dataset['dataset', 'cites', 'dataset']
    del test_dataset['dataset']
    test_citation_network = test_dataset.to_homogeneous().to(device)

    test_dataset = BipartiteDataset(root=f'{args.path}')
    test_dataset = test_dataset[0]
    del test_dataset['publication', 'cites', 'publication']
    del test_dataset['dataset', 'cites', 'dataset']
    test_dataset['publication', 'cites', 'dataset'].edge_index = test_dataset[
        'publication', 'cites', 'dataset'].edge_index_test
    test_dataset['publication', 'cites', 'dataset'].edge_label_index = test_dataset[
        'publication', 'cites', 'dataset'].edge_label_index_test
    tups_test = test_dataset['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_test = [tuple(e) for e in tups_test]
    print(f'INTERSECTION: {len(list(set(tups_train).intersection(tups_test)))}')
    test_bipartite_network = test_dataset.to_homogeneous().to(device)

    in_channels, out_channels = data[0]['publication'].num_features, 16 # 16 was not stated in the paper
    print(in_channels,train_bipartite_network.num_nodes)
    model = GVAEAutoEncoder(in_channels, train_citation_network.num_nodes, hidden1, hidden2,hidden1_mlp,hidden2_mlp, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()
        loss,_,_ = model(train_citation_network.x,num_training_papers,train_citation_network.edge_index,train_bipartite_network.x,train_bipartite_network.num_nodes,train_bipartite_network.edge_index,type)
        loss.backward()
        optimizer.step()
        return float(loss)

    def validation():
        model.eval()
        with torch.no_grad():
            loss,_,_ = model(validation_citation_network.x,num_training_papers,validation_citation_network.edge_index,validation_bipartite_network.x,validation_bipartite_network.num_nodes,validation_bipartite_network.edge_index,type)

        return float(loss)

    times=[]
    early_stopping_counter = 0
    max_count = 50
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train()
        # val_loss = validation()
        #
        print(f'Epoch: {epoch:03d}, LOSS ON TRAIN: {loss:.4f}')
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        #     best_model_state_dict = model.state_dict()
        # else:
        #     early_stopping_counter += 1
        # if early_stopping_counter == max_count:
        #     print('max count reached, exit')
        #     break


        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    model.eval()
    with torch.no_grad():
        z,mu,log = model.paper_encoder(test_citation_network.x, test_citation_network.edge_index)

        matrix = torch.zeros((test_bipartite_network.num_nodes - test_citation_network.num_nodes, test_citation_network.num_nodes), dtype=torch.float).to(device)
        matrix[test_bipartite_network.edge_index[1] - test_citation_network.num_nodes, test_bipartite_network.edge_index[0]] = 1  # assuming undirected graph

        zd,mu2,log2 = model.dataset_encoder(matrix)
        if type != 'linear':
            zd,mu2,log2 = model.paper_encoder(test_bipartite_network.x,test_bipartite_network.edge_index,type)
            zd = zd[num_training_papers:,:]

        recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_bipartite_network.edge_label_index[0].tolist()))),:],zd.t()))

        top_values, top_indices = torch.topk(recon, k=20, dim=1)
        sorted_recon = top_indices.tolist()

        edge_index = test_bipartite_network.edge_label_index.t().tolist()
        edge_index = sorted(edge_index, key=lambda x: x[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # print(edge_index)
        edge_index = edge_index.tolist()
        edge_index_dict = {}
        for i, pair in enumerate(edge_index):
            if edge_index_dict.get(pair[0], None) is not None:
                edge_index_dict[pair[0]].append(pair[1] - test_citation_network.num_nodes)
            else:
                edge_index_dict[pair[0]] = [pair[1] - test_citation_network.num_nodes]
            edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))
        # print(edge_index_dict)
        edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))
        # print(edge_index_dict)

        # print(len(list(edge_index_dict.keys())))
        # print(edge_index_dict)
        # Sort the tensor along dimension 1 and get the indices

        # Check the sorted indices
        json_res = {}
        string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}_{hidden1_mlp}_{hidden2_mlp}_{type}'
        for k in [1, 5, 10]:
            print('RESULTS: ',k)
            precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
            print(ndcg, precision, recall)
            json_res[f'ndcg_{str(k)}'] = str(ndcg)
            json_res[f'precision_{str(k)}'] = str(precision)
            json_res[f'recall_{str(k)}'] = str(recall)
            print('\n\n')
        f = open(f'./hgvae_final_{string}_reduced.json','w')
        json.dump(json_res,f,indent=4)

def run_enriched():
    seed = args.seed

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('TRAINING')
    data = BipartiteDataset(root=f'{args.path}')
    train_dataset = data[0]
    num_training_papers = train_dataset['publication'].num_nodes
    num_training_dataset = train_dataset['dataset'].num_nodes

    print(num_training_papers)
    # del train_dataset['publication', 'cites', 'dataset']
    # del train_dataset['dataset', 'cites', 'dataset']
    # del train_dataset['dataset']

    # print(train_dataset)
    train_citation_network = train_dataset.to_homogeneous().to(device)
    print(train_citation_network)
    data = BipartiteDataset(root=f'{args.path}')
    train_bipartite_network = data[0]

    # del train_bipartite_network['publication', 'cites', 'publication']
    # del train_bipartite_network['dataset', 'cites', 'dataset']
    train_bipartite_network['publication', 'cites', 'dataset'].edge_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_train
    train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_train

    train_bipartite_network = train_bipartite_network.to_homogeneous().to(device)
    # print(train_bipartite_network)



    data = BipartiteDataset(root=f'{args.path}')
    test_dataset = data[0]
    # del test_dataset['publication', 'cites', 'dataset']
    # del test_dataset['dataset', 'cites', 'dataset']
    # del test_dataset['dataset']
    # print(test_dataset)
    test_citation_network = test_dataset.to_homogeneous().to(device)

    # print(test_citation_network)
    test_dataset = BipartiteDataset(root=f'{args.path}')
    test_dataset = test_dataset[0]
    # print(test_dataset)

    del test_dataset['publication', 'cites', 'publication']
    del test_dataset['dataset', 'cites', 'dataset']

    test_dataset['publication', 'cites', 'dataset'].edge_index = test_dataset[
        'publication', 'cites', 'dataset'].edge_index_test
    test_dataset['publication', 'cites', 'dataset'].edge_label_index = test_dataset[
        'publication', 'cites', 'dataset'].edge_label_index_test

    test_bipartite_network = test_dataset.to_homogeneous().to(device)
    # print(test_bipartite_network)

    in_channels, out_channels = data[0]['publication'].num_features, 16 # 16 was not stated in the paper
    print(in_channels,train_bipartite_network.num_nodes)
    model = GVAEAutoEncoder(in_channels, num_training_papers, hidden1, hidden2,hidden1_mlp,hidden2_mlp, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()
        loss,_,_ = model(train_citation_network.x,num_training_papers,train_citation_network.edge_index,train_bipartite_network.x,train_bipartite_network.num_nodes,train_bipartite_network.edge_index,type,enriched=True)
        loss.backward()
        optimizer.step()
        return float(loss)


    times=[]
    early_stopping_counter = 0
    max_count = 50
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train()
        # val_loss = validation()
        #
        print(f'Epoch: {epoch:03d}, LOSS ON TRAIN: {loss:.4f}')
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        #     best_model_state_dict = model.state_dict()
        # else:
        #     early_stopping_counter += 1
        # if early_stopping_counter == max_count:
        #     print('max count reached, exit')
        #     break


        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    model.eval()
    with torch.no_grad():
        z,mu,log = model.paper_encoder(test_citation_network.x, test_citation_network.edge_index)
        edge_idx_dataset = test_bipartite_network.edge_index
        mask = edge_idx_dataset[0] < num_training_papers
        edge_idx_dataset = edge_idx_dataset[:, mask]
        mask = edge_idx_dataset[1] >= num_training_papers
        edge_idx_dataset = edge_idx_dataset[:, mask]
        print(edge_idx_dataset.shape)


        matrix = torch.zeros((test_bipartite_network.num_nodes - num_training_papers, num_training_papers), dtype=torch.float).to(device)
        matrix[edge_idx_dataset[1] - num_training_papers, edge_idx_dataset[0]] = 1  # assuming undirected graph

        zd,mu2,log2 = model.dataset_encoder(matrix)
        print(zd.shape)

        if type != 'linear':
            zd,mu2,log2 = model.paper_encoder(test_bipartite_network.x,test_bipartite_network.edge_index,type)
            zd = zd[num_training_papers:,:]

        recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_bipartite_network.edge_label_index[0].tolist()))),:],zd.t()))
        print(recon.shape)
        # sorted_recon = torch.argsort(recon, dim=1,descending=True).tolist()
        top_values, top_indices = torch.topk(recon, k=20, dim=1)
        # sorted_recon = [t[:20] for t in sorted_recon]
        sorted_recon = top_indices.tolist()

        edge_index = test_bipartite_network.edge_label_index.t().tolist()
        edge_index = sorted(edge_index, key=lambda x: x[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # print(edge_index)
        edge_index = edge_index.tolist()
        edge_index_dict = {}
        for i, pair in enumerate(edge_index):
            if edge_index_dict.get(pair[0], None) is not None:
                edge_index_dict[pair[0]].append(pair[1] - num_training_papers)
            else:
                edge_index_dict[pair[0]] = [pair[1] - num_training_papers]
            edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))
        # print(edge_index_dict)
        edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))
        # print(edge_index_dict)

        # print(len(list(edge_index_dict.keys())))
        # print(edge_index_dict)
        # Sort the tensor along dimension 1 and get the indices

        # Check the sorted indices
        json_res = {}
        string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}_{hidden1_mlp}_{hidden2_mlp}_{type}'
        for k in [1, 5, 10]:
            print('RESULTS: ',k)
            precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
            print(ndcg, precision, recall)
            json_res[f'ndcg_{str(k)}'] = str(ndcg)
            json_res[f'precision_{str(k)}'] = str(precision)
            json_res[f'recall_{str(k)}'] = str(recall)
            print('\n\n')
        f = open(f'./hgvae_final_{string}_enriched.json','w')
        json.dump(json_res,f,indent=4)




if __name__ == '__main__':
    # hidden1 = 64
    # hidden2 = 32
    # hidden1_mlp = 64  # not stated in the paper
    # hidden2_mlp = 32  # not stated in the paper
    # decay = 1e-5
    # lr = 0.01
    epochs = 500
    lrs = [0.01,0.001]
    hiddens = [(64,32),(128,64),(128,32)]
    hiddens_mlp = [(64,32),(128,64),(128,32)]
    decays = [1e-5,5e-5,1e-4,1e-6]

    for dataset in ['pubmed']:
        for type in ['linear','gcn','sage']:
            if dataset == 'pubmed':
                hiddens.append((256,128))
                hiddens_mlp.append((256, 128))
            for iteration in range(10):
                lr = random.choice(lrs)
                random.shuffle(hiddens)
                random.shuffle(hiddens_mlp)
                random.shuffle(lrs)
                random.shuffle(decays)
                hids = random.choice(hiddens)
                hids_1 = random.choice(hiddens_mlp)
                print(hids,hids_1)
                hidden1 = hids[0]
                hidden2 = hids[1]
                hidden1_mlp = hids_1[0]
                hidden2_mlp = hids_1[1]
                decay = random.choice(decays)
                print(f'WORKING ON {dataset}')
                # run_enriched()
                run_reduced()
                run()
                # run()
                run_reduced()


