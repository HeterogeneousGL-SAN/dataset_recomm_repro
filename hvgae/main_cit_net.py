import argparse
import os.path as osp
import time
import tqdm
import torch
import os
import json
import numpy as np
from loader import BipartiteDataset
from torch_geometric import seed_everything

seed_everything(10)
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.data import Batch,InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Linear, VGAE, GCNConv, SAGEConv
from torch_geometric.utils import to_dense_adj,to_undirected
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

import random
from models.autoencoders import GVAEPaperAutoEncoder
from sklearn.metrics import ndcg_score
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mes',
                    choices=['mes','pubmed_kcore','pubmed'],help="The selected dataset")
parser.add_argument('--vgae',type=str,default='cit_net_vae',choices=['hvgae','mlp_vae','gcn_vae','cit_net_vae',
                                                              'sage_vae_enriched','sage_vae','sage_vae_bip'])
parser.add_argument('--seed',default=42)
parser.add_argument('--path',default=str)

args = parser.parse_args()

# seed = 42
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)
# g = torch.Generator()
# g.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
def edge_index_split(edge_index,reduced=False):
    print('SPLITTING')
    edge_index_t = edge_index.t().tolist()
    sources = [e[0] for e in edge_index_t]
    targets = [e[1] for e in edge_index_t]

    val_test = []
    validation = []
    test = []
    train = []
    train_nodes = []
    val_nodes = []
    val = []
    for source,target in zip(sources,targets):
        if source in train and target in train and len(val_nodes) < 0.1*edge_index.size(1):
            val_nodes.append([source,target])
            val.append(target)
            val.append(source)
        else:
            train_nodes.append([source,target])
            train.append(target)
            train.append(source)
    train_final = []
    for el in train_nodes:
        if el[0] in val and el[1] in val:
            train_final.append(el)
    print(len(train_nodes))
    print(len(val_nodes))
    print(len(train_final))
    train_final = torch.tensor(train_final).t()
    train_nodes = torch.tensor(train_nodes).t()
    val_nodes = torch.tensor(val_nodes).t()
    validation = val_nodes[:,:int(val_nodes.size(1)/2)]
    test = val_nodes[:,int(val_nodes.size(1)/2):]
    # print(train_nodes)
    # print(train_nodes.t())
    # print(val_nodes.t())
    if reduced:
        return train_final, validation, val_nodes

    return train_nodes,validation,val_nodes

def edge_index_split_mes(edge_index):

    print('SPLITTING')
    edge_index_t = edge_index.t().tolist()
    sources = [e[0] for e in edge_index_t]
    targets = [e[1] for e in edge_index_t]

    val_test = []
    validation = []
    test = []
    train = []
    train_nodes = []
    val_nodes = []
    val = []
    for source,target in zip(sources,targets):
        if source in train and target in train and len(val_nodes) < 0.1*edge_index.size(1):
            val_nodes.append([source,target])
            val.append(target)
            val.append(source)
        else:
            train_nodes.append([source,target])
            train.append(target)
            train.append(source)
    train_final = []
    for el in train_nodes:
        if el[0] in val or el[1] in val:
            train_final.append(el)
    print(len(train_nodes))
    print(len(val_nodes))
    print(len(train_final))
    train_final = torch.tensor(train_final).t()
    train_nodes = torch.tensor(train_nodes).t()
    val_nodes = torch.tensor(val_nodes).t()
    validation = val_nodes[:,:int(val_nodes.size(1)/2)]
    test = val_nodes[:,int(val_nodes.size(1)/2):]
    # print(train_nodes)
    # print(train_nodes.t())
    # print(val_nodes.t())
    return train_nodes,validation,val_nodes
    # for edge in edge_index_t:
    #     if ((counts_sources[edge[0]] + counts_sources[edge[1]] > 1) and (counts_targets[edge[1]] + counts_targets[edge[0]] > 1)):
    #         val_test.append(edge)


    # val_test = sorted(val_test, key=lambda x: x[0])
    # cur = None
    # for el in val_test:
    #     if el[0] != cur:
    #         train.append(el)
    #         cur = el[0]
    #     else:
    #         if len(validation) > len(test):
    #             test.append(el)
    #         else:
    #             validation.append(el)
    # train = torch.tensor(train)
    # validation = torch.tensor(validation)
    # test = torch.tensor(test)
    #
    # return train.t(),validation.t(),test.t()



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

def evaluation(rankings, edge_index,k = 10):
    edge_index = {key:list(set(value)) for key,value in edge_index.items()}
    print(len(edge_index))
    print(len(rankings))
    values = [v for k,v in edge_index.items()]

    precision, recall, ndcg = 0, 0, 0
    edge_index = dict(sorted(edge_index.items(), key=lambda x: x[0]))
    for index,i in enumerate(values):
        true_vals = i
        predicted = rankings[index]
        predicted = predicted[:k]
        correct = list(set(true_vals) & set(predicted))
        precision += len(correct) / k
        recall += len(correct) / len(true_vals)
        ndcg += ndcg_at_k(predicted, true_vals, k)
    return precision/len(edge_index), recall/len(edge_index), ndcg/len(edge_index)



def run():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print('TRAINING')

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
    )

    data = BipartiteDataset(root=f'{args.path}')
    train_data = data[0]
    print(train_data)

    del train_data['publication', 'cites', 'dataset']
    del train_data['dataset', 'cites', 'dataset']
    del train_data['dataset']
    train_data = train_data.to_homogeneous().to(device)
    train_split,val_split,test_split = edge_index_split(train_data.edge_index)
    train_data.edge_index = train_split
    train_data.edge_label_index = train_split
    train_data.to(device)
    # edge index rimosso è un terzo del train data edge insec
    # ei = train_data.edge_index[:,:int(2*train_data.edge_index.size(1)/3)]
    # train_data.edge_label_index = ei
    # train_data.edge_index = ei

    data = BipartiteDataset(root=f'{args.path}')
    validation_data = data[0]
    # setup for noraml-- uncomment
    del validation_data['publication', 'cites', 'dataset']
    del validation_data['dataset', 'cites', 'dataset']
    del validation_data['dataset']
    validation_data = validation_data.to_homogeneous().to(device)
    validation_data.edge_label_index = val_split
    validation_data.to(device)

    # eiv = validation_data.edge_index[:,int(2*validation_data.edge_index.size(1)/3):int(2*validation_data.edge_index.size(1)/3)+int(validation_data.edge_index.size(1)/6)]
    # validation_data.edge_index = ei
    # validation_data.edge_label_index = eiv
    #
    data = BipartiteDataset(root=f'{args.path}')
    test_data = data[0]

    # setup for noraml-- uncomment
    del test_data['publication', 'cites', 'dataset']
    del test_data['dataset', 'cites', 'dataset']
    del test_data['dataset']
    test_data = test_data.to_homogeneous().to(device)
    test_data.edge_index = train_split
    test_data.edge_label_index = test_split
    test_data.to(device)

    print(train_data)
    # print(validation_data)
    print(test_data)

    in_channels, out_channels = train_data.num_features, 16 # 16 was not stated in the paper

    # model = GVAEAutoEncoder(in_channels, train_citation_network.num_nodes, hidden1, hidden2,hidden1_mlp,hidden2_mlp, out_channels).to(device)
    model = GVAEPaperAutoEncoder(in_channels, hidden1, hidden2, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()

        zp, mu1, log1 = model.encode_paper(train_data.x, train_data.edge_index)
        loss_paper = model.recon_loss(zp, train_data.edge_label_index)
        loss = loss_paper + (1 / train_data.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        return float(loss)

    def validation():
        model.eval()
        with torch.no_grad():
            zp, mu1, log1 = model.encode_paper(validation_data.x, validation_data.edge_index)
            loss_paper = model.recon_loss(zp, validation_data.edge_label_index)
            loss = loss_paper + (1 / validation_data.num_nodes) * model.kl_loss()

        return float(loss)

    times=[]
    early_stopping_counter = 0
    max_count = 400
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train()
        # val_loss = validation()
        #
        # print(f'Epoch: {epoch:03d}, LOSS ON TRAIN: {loss:.4f}, LOSS ON VALIDATION: {val_loss: 4f}')
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        # else:
        #     early_stopping_counter += 1
        # if early_stopping_counter == max_count:
        #     print('max count reached, exit')
        #     break


        times.append(time.time() - start)

    model.eval()
    z,mu,log = model.encode_paper(test_data.x, test_data.edge_index)
    # print(z)
    recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_data.edge_label_index[0].tolist()))),:],z.t()))


    print(recon.shape)


    # sorted_recon = torch.argsort(recon, dim=1,descending=True).tolist()
    # sorted_recon = [t[:20] for t in sorted_recon]
    top_values, top_indices = torch.topk(recon, k=20, dim=1)
    sorted_recon = top_indices.tolist()
    # print(sorted_recon[0:10])

    edge_index = test_data.edge_label_index.t().tolist()
    edge_index = sorted(edge_index, key=lambda x: x[0])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.tolist()
    edge_index_dict = {}
    for i, pair in enumerate(edge_index):
        if edge_index_dict.get(pair[0], None) is not None:
            edge_index_dict[pair[0]].append(pair[1])
        else:
            edge_index_dict[pair[0]] = [pair[1]]
        edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))
    edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))


    # print(edge_index_dict)
    # Sort the tensor along dimension 1 and get the indices
    json_res = {}
    string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}'
    for k in [1, 5, 10]:
        print('RESULTS: ',k)
        precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
        print(ndcg, precision, recall)
        json_res[f'ndcg_{str(k)}'] = str(ndcg)
        json_res[f'precision_{str(k)}'] = str(precision)
        json_res[f'recall_{str(k)}'] = str(recall)
        print('\n\n')
    # f = open(f'baselines/hvgae/results/{dataset}/cit_net_{string}.json','w')
    f = open(f'baselines/hvgae/results/{dataset}/standard/cit_net_{string}.json','w')
    json.dump(json_res,f,indent=4)


def run_enriched():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print('TRAINING')

    data = BipartiteDataset(root=f'{args.path}')
    train_data = data[0]
    print(train_data)
    num_training_papers = train_data['publication'].num_nodes
    ei = train_data['publication','cites','publication'].edge_index
    train_data = train_data.to_homogeneous().to(device)
    train_split,val_split,test_split = edge_index_split(ei)
    train_data.edge_index = train_split
    train_data.edge_label_index = train_split
    train_data.to(device)

    data = BipartiteDataset(root=f'{args.path}')
    test_data = data[0]
    num_test_papers = test_data['publication'].num_nodes
    print(num_test_papers,num_training_papers)
    print('end')
    test_data = test_data.to_homogeneous().to(device)
    test_data.edge_index = train_split
    test_data.edge_label_index = test_split
    test_data.to(device)

    print(train_data)
    # print(validation_data)
    print(test_data)

    in_channels, out_channels = train_data.num_features, 16 # 16 was not stated in the paper

    # model = GVAEAutoEncoder(in_channels, train_citation_network.num_nodes, hidden1, hidden2,hidden1_mlp,hidden2_mlp, out_channels).to(device)
    model = GVAEPaperAutoEncoder(in_channels, hidden1, hidden2, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()

        zp, mu1, log1 = model.encode_paper(train_data.x, train_data.edge_index)
        loss_paper = model.recon_loss(zp, train_data.edge_label_index)
        loss = loss_paper + (1 / train_data.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        return float(loss)


    times=[]
    early_stopping_counter = 0
    max_count = 400
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train()
        # val_loss = validation()
        #
        # print(f'Epoch: {epoch:03d}, LOSS ON TRAIN: {loss:.4f}, LOSS ON VALIDATION: {val_loss: 4f}')
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        # else:
        #     early_stopping_counter += 1
        # if early_stopping_counter == max_count:
        #     print('max count reached, exit')
        #     break


        times.append(time.time() - start)

    model.eval()
    z,mu,log = model.encode_paper(test_data.x, test_data.edge_index)
    z = z[:num_training_papers,:]
    recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_data.edge_label_index[0].tolist()))),:],z.t()))


    print(recon.shape)


    # sorted_recon = torch.argsort(recon, dim=1,descending=True).tolist()
    # sorted_recon = [t[:20] for t in sorted_recon]
    top_values, top_indices = torch.topk(recon, k=20, dim=1)
    sorted_recon = top_indices.tolist()
    # print(sorted_recon[0:10])

    edge_index = test_data.edge_label_index.t().tolist()
    edge_index = sorted(edge_index, key=lambda x: x[0])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.tolist()
    edge_index_dict = {}
    for i, pair in enumerate(edge_index):
        if edge_index_dict.get(pair[0], None) is not None:
            edge_index_dict[pair[0]].append(pair[1])
        else:
            edge_index_dict[pair[0]] = [pair[1]]
        edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))
    edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))


    # print(edge_index_dict)
    # Sort the tensor along dimension 1 and get the indices
    json_res = {}
    string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}'
    for k in [1, 5, 10]:
        print('RESULTS: ',k)
        precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
        print(ndcg, precision, recall)
        json_res[f'ndcg_{str(k)}'] = str(ndcg)
        json_res[f'precision_{str(k)}'] = str(precision)
        json_res[f'recall_{str(k)}'] = str(recall)
        print('\n\n')
    # f = open(f'baselines/hvgae/results/{dataset}/cit_net_{string}.json','w')
    f = open(f'baselines/hvgae/results/{dataset}/enriched/cit_net_{string}.json','w')
    json.dump(json_res,f,indent=4)

def run_reduced():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print('TRAINING')

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
    )

    data = BipartiteDataset(root=f'{args.path}')
    train_data = data[0]
    print(train_data)
    num_training_papers = train_data['publication'].num_nodes

    del train_data['publication', 'cites', 'dataset']
    del train_data['dataset', 'cites', 'dataset']
    del train_data['dataset']
    train_data = train_data.to_homogeneous().to(device)
    train_split,val_split,test_split = edge_index_split(train_data.edge_index,reduced=True)
    train_data.edge_index = train_split
    train_data.edge_label_index = train_split
    train_data.to(device)
    # edge index rimosso è un terzo del train data edge insec
    # ei = train_data.edge_index[:,:int(2*train_data.edge_index.size(1)/3)]
    # train_data.edge_label_index = ei
    # train_data.edge_index = ei

    data = BipartiteDataset(root=f'{args.path}')
    validation_data = data[0]
    # setup for noraml-- uncomment
    del validation_data['publication', 'cites', 'dataset']
    del validation_data['dataset', 'cites', 'dataset']
    del validation_data['dataset']
    validation_data = validation_data.to_homogeneous().to(device)
    validation_data.edge_label_index = val_split
    validation_data.to(device)

    # eiv = validation_data.edge_index[:,int(2*validation_data.edge_index.size(1)/3):int(2*validation_data.edge_index.size(1)/3)+int(validation_data.edge_index.size(1)/6)]
    # validation_data.edge_index = ei
    # validation_data.edge_label_index = eiv
    #
    data = BipartiteDataset(root=f'{args.path}')
    test_data = data[0]

    # setup for noraml-- uncomment
    del test_data['publication', 'cites', 'dataset']
    del test_data['dataset', 'cites', 'dataset']
    del test_data['dataset']
    test_data = test_data.to_homogeneous().to(device)
    test_data.edge_index = train_split
    test_data.edge_label_index = test_split
    test_data.to(device)

    print(train_data)
    # print(validation_data)
    print(test_data)

    in_channels, out_channels = train_data.num_features, 16 # 16 was not stated in the paper

    # model = GVAEAutoEncoder(in_channels, train_citation_network.num_nodes, hidden1, hidden2,hidden1_mlp,hidden2_mlp, out_channels).to(device)
    model = GVAEPaperAutoEncoder(in_channels, hidden1, hidden2, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()

        zp, mu1, log1 = model.encode_paper(train_data.x, train_data.edge_index)
        loss_paper = model.recon_loss(zp, train_data.edge_label_index)
        loss = loss_paper + (1 / train_data.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        return float(loss)

    def validation():
        model.eval()
        with torch.no_grad():
            zp, mu1, log1 = model.encode_paper(validation_data.x, validation_data.edge_index)
            loss_paper = model.recon_loss(zp, validation_data.edge_label_index)
            loss = loss_paper + (1 / validation_data.num_nodes) * model.kl_loss()

        return float(loss)

    times=[]
    early_stopping_counter = 0
    max_count = 400
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train()
        # val_loss = validation()
        #
        # print(f'Epoch: {epoch:03d}, LOSS ON TRAIN: {loss:.4f}, LOSS ON VALIDATION: {val_loss: 4f}')
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        # else:
        #     early_stopping_counter += 1
        # if early_stopping_counter == max_count:
        #     print('max count reached, exit')
        #     break


        times.append(time.time() - start)

    model.eval()
    z,mu,log = model.encode_paper(test_data.x, test_data.edge_index)
    z = z[:num_training_papers,:]
    # print(z)
    recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_data.edge_label_index[0].tolist()))),:],z.t()))


    print(recon.shape)


    # sorted_recon = torch.argsort(recon, dim=1,descending=True).tolist()
    # sorted_recon = [t[:20] for t in sorted_recon]
    top_values, top_indices = torch.topk(recon, k=20, dim=1)
    sorted_recon = top_indices.tolist()
    # print(sorted_recon[0:10])

    edge_index = test_data.edge_label_index.t().tolist()
    edge_index = sorted(edge_index, key=lambda x: x[0])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.tolist()
    edge_index_dict = {}
    for i, pair in enumerate(edge_index):
        if edge_index_dict.get(pair[0], None) is not None:
            edge_index_dict[pair[0]].append(pair[1])
        else:
            edge_index_dict[pair[0]] = [pair[1]]
        edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))
    edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))


    # print(edge_index_dict)
    # Sort the tensor along dimension 1 and get the indices
    json_res = {}
    string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}'
    for k in [1, 5, 10]:
        print('RESULTS: ',k)
        precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
        print(ndcg, precision, recall)
        json_res[f'ndcg_{str(k)}'] = str(ndcg)
        json_res[f'precision_{str(k)}'] = str(precision)
        json_res[f'recall_{str(k)}'] = str(recall)
        print('\n\n')
    # f = open(f'baselines/hvgae/results/{dataset}/cit_net_{string}.json','w')
    f = open(f'baselines/hvgae/results/{dataset}/reduced/cit_net_{string}.json','w')
    json.dump(json_res,f,indent=4)

if __name__ == '__main__':
    epochs = 500
    lrs = [0.01,0.001]
    hiddens = [(64,32),(128,64),(128,32)]
    decays = [1e-5,5e-5,1e-4]
    #
    for dataset in ['mes','pubmed_kcore','pubmed']:
        if dataset == 'pubmed':
            lrs = [0.001,0.005]
            hiddens = [(256, 128),(128,64),(128,128),(512,256) ,(256,256), (256, 64), (256, 128)]
        for iteration in range(30):
            random.shuffle(hiddens)
            random.shuffle(lrs)
            random.shuffle(decays)
            lr = random.choice(lrs)
            hids = random.choice(hiddens)
            hidden1 = hids[0]
            hidden2 = hids[1]
            decay = random.choice(decays)
            print(f'WORKING ON {dataset}')
            run()
            run_enriched()
            run_reduced()



