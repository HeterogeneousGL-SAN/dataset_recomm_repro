import argparse
import os.path as osp
import time
import tqdm
import torch
import os
import json
import numpy as np
from loader import BipartiteDataset

import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.data import Batch,InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Linear, VGAE, GCNConv, SAGEConv
from torch_geometric.utils import to_dense_adj
import random
from models.autoencoders import GVAEPaperAutoEncoder
from sklearn.metrics import ndcg_score


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['mes','pubmed_kcore','pubmed'],help="The selected dataset")
parser.add_argument('--vgae',type=str,default='cit_net_vae',choices=['hvgae','mlp_vae','gcn_vae','cit_net_vae',
                                                              'sage_vae_enriched','sage_vae','sage_vae_bip'])
parser.add_argument('--seed',default=42)
parser.add_argument('--repro')
parser.add_argument('--path',default=str)

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
g = torch.Generator()
g.manual_seed(args.seed)

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
    all_datasets = []
    values = [v for k,v in edge_index.items()]
    keys = [k for k,v in edge_index.items()]

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
    seed = args.seed

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print('TRAINING')

    data = BipartiteDataset(root=f'{args.path}')
    train_dataset = data[0]
    num_training_papers = train_dataset['publication'].num_nodes
    print(train_dataset)
    print(num_training_papers)
    # del train_dataset['publication', 'cites', 'dataset']
    # del train_dataset['dataset', 'cites', 'dataset']
    # del train_dataset['dataset']
    # print(train_dataset)
    # train_citation_network = train_dataset.to_homogeneous().to(device)
    # print(train_citation_network)

    data = BipartiteDataset(root=f'{args.path}')
    train_bipartite_network = data[0]

    # remove comment to consider only bip net
    del train_bipartite_network['publication', 'cites', 'publication']
    del train_bipartite_network['dataset', 'cites', 'dataset']

    train_bipartite_network['publication', 'cites', 'dataset'].edge_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_train
    train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_train


    tups_train = train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_train = [tuple(e) for e in tups_train]
    tups_train_all = train_bipartite_network['publication', 'cites', 'dataset'].edge_index.t().tolist()
    tups_train_all = [tuple(e) for e in tups_train_all]
    print(train_bipartite_network)
    train_bipartite_network = train_bipartite_network.to_homogeneous().to(device)
    print(train_bipartite_network)


    # print('VALIDATION')
    # data = BipartiteDataset(root=f'{args.path}')
    # validation_dataset = data[0]
    # del validation_dataset['publication', 'cites', 'dataset']
    # del validation_dataset['dataset', 'cites', 'dataset']
    # del validation_dataset['dataset']
    # print(validation_dataset)
    # validation_citation_network = validation_dataset.to_homogeneous().to(device)
    # print(validation_citation_network)

    data = BipartiteDataset(root=f'{args.path}')
    validation_bipartite_network = data[0]

    # remove comment to consider only bip net
    del validation_bipartite_network['publication', 'cites', 'publication']
    del validation_bipartite_network['dataset', 'cites', 'dataset']

    validation_bipartite_network['publication', 'cites', 'dataset'].edge_index = validation_bipartite_network['publication', 'cites', 'dataset'].edge_index_val
    validation_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = validation_bipartite_network['publication', 'cites', 'dataset'].edge_label_index_val
    print(validation_bipartite_network)
    validation_bipartite_network = validation_bipartite_network.to_homogeneous().to(device)
    print(validation_bipartite_network)
    # validation_bipartite_network = transform(validation_bipartite_network)
    print('TEST')
    # data = BipartiteDataset(root=f'{args.path}')
    # test_dataset = data[0]
    # print(test_dataset)
    # del validation_dataset['publication', 'cites', 'dataset']
    # del validation_dataset['dataset', 'cites', 'dataset']
    # del validation_dataset['dataset']
    # test_citation_network = test_dataset.to_homogeneous().to(device)
    #
    # print(test_citation_network)
    test_dataset = BipartiteDataset(root=f'{args.path}')
    test_bipartite_network = test_dataset[0]

    # remove comment to consider only bip net
    del test_bipartite_network['publication', 'cites', 'publication']
    del test_bipartite_network['dataset', 'cites', 'dataset']

    test_bipartite_network['publication', 'cites', 'dataset'].edge_index = test_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_test
    test_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = test_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_test
    tups_test = test_bipartite_network['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_test = [tuple(e) for e in tups_test]
    tups_test_all = test_bipartite_network['publication', 'cites', 'dataset'].edge_index.t().tolist()
    tups_test_all = [tuple(e) for e in tups_test_all]
    print(f'INTERSECTION: {len(list(set(tups_train).intersection(tups_test)))}')
    print(f'INTERSECTION ALL TRAIN: {len(list(set(tups_train).intersection(tups_train_all)))}')
    print(f'INTERSECTION ALL TEST: {len(list(set(tups_test).intersection(tups_test_all)))}')

    print(test_bipartite_network)
    test_bipartite_network = test_bipartite_network.to_homogeneous().to(device)

    print(test_bipartite_network)

    in_channels, out_channels = train_bipartite_network.num_features, 16 # 16 was not stated in the paper
    model = GVAEPaperAutoEncoder(in_channels, hidden1, hidden2, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()
        loss = 0


        zp, mu1, log1 = model.encode_paper(train_bipartite_network.x, train_bipartite_network.edge_index,type)
        loss_paper = model.recon_loss(zp, train_bipartite_network.edge_index)
        loss = loss_paper + (1 / train_bipartite_network.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        return float(loss)

    def validation():
        model.eval()
        with torch.no_grad():
            optimizer.zero_grad()



            zp, mu1, log1 = model.encode_paper(validation_bipartite_network.x, validation_bipartite_network.edge_index,type)
            loss_paper = model.recon_loss(zp, validation_bipartite_network.edge_index)
            loss = loss_paper + (1 / validation_bipartite_network.num_nodes) * model.kl_loss()


            return float(loss)

    times=[]
    early_stopping_counter = 0
    max_count = 100
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
        z,mu,log = model.encode_paper(test_bipartite_network.x, test_bipartite_network.edge_index,type)
        print(len(list(set(test_bipartite_network.edge_index[0].tolist()))))
        recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_bipartite_network.edge_label_index[0].tolist()))),:],z[num_training_papers:,:].t()))

        # sorted_recon = torch.argsort(recon, dim=1,descending=True).tolist()
        # sorted_recon = [t[:20] for t in sorted_recon]
        top_values, top_indices = torch.topk(recon, k=20, dim=1)
        sorted_recon = top_indices.tolist()

        edge_index = test_bipartite_network.edge_label_index.t().tolist()

        edge_index = sorted(edge_index, key=lambda x: x[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.tolist()
        edge_index_dict = {}
        for i, pair in enumerate(edge_index):
            if pair[0] < num_training_papers and pair[1] >= num_training_papers:
                if edge_index_dict.get(pair[0], None) is not None:
                    edge_index_dict[pair[0]].append(pair[1] - num_training_papers)
                else:
                    edge_index_dict[pair[0]] = [pair[1] - num_training_papers]
                edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))

        edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))

        # Sort the tensor along dimension 1 and get the indices

        # Check the sorted indices
        json_res = {}
        string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}_{type}'
        for k in [1, 5, 10]:
            print('RESULTS: ', k)
            precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
            print(ndcg, precision, recall)
            json_res[f'ndcg_{str(k)}'] = str(ndcg)
            json_res[f'precision_{str(k)}'] = str(precision)
            json_res[f'recall_{str(k)}'] = str(recall)
            print('\n\n')
        f = open(f'./bip_net_{string}_standard.json', 'w')
        json.dump(json_res, f, indent=4)
        # for k in [1,5,10]:
        #     precision,recall,ndcg = evaluation(sorted_recon,edge_index_dict,k=k)
        #     print(ndcg,precision,recall)


def run_enriched():
    seed = args.seed

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print('TRAINING')

    data = BipartiteDataset(root=f'{args.path}')
    train_dataset = data[0]
    num_training_papers = train_dataset['publication'].num_nodes
    print(train_dataset)
    print(num_training_papers)


    data = BipartiteDataset(root=f'{args.path}')
    train_bipartite_network = data[0]
    train_bipartite_network['publication', 'cites', 'dataset'].edge_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_train
    train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = train_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_train


    tups_train = train_bipartite_network['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_train = [tuple(e) for e in tups_train]
    tups_train_all = train_bipartite_network['publication', 'cites', 'dataset'].edge_index.t().tolist()
    tups_train_all = [tuple(e) for e in tups_train_all]
    print(train_bipartite_network)
    train_bipartite_network = train_bipartite_network.to_homogeneous().to(device)
    print(train_bipartite_network)

    data = BipartiteDataset(root=f'{args.path}')
    validation_bipartite_network = data[0]
    validation_bipartite_network['publication', 'cites', 'dataset'].edge_index = validation_bipartite_network['publication', 'cites', 'dataset'].edge_index_val
    validation_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = validation_bipartite_network['publication', 'cites', 'dataset'].edge_label_index_val
    print(validation_bipartite_network)
    validation_bipartite_network = validation_bipartite_network.to_homogeneous().to(device)
    print(validation_bipartite_network)
    # validation_bipartite_network = transform(validation_bipartite_network)
    print('TEST')

    test_dataset = BipartiteDataset(root=f'{args.path}')
    test_bipartite_network = test_dataset[0]

    test_bipartite_network['publication', 'cites', 'dataset'].edge_index = test_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_test
    test_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = test_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_test
    tups_test = test_bipartite_network['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_test = [tuple(e) for e in tups_test]
    tups_test_all = test_bipartite_network['publication', 'cites', 'dataset'].edge_index.t().tolist()
    tups_test_all = [tuple(e) for e in tups_test_all]
    print(f'INTERSECTION: {len(list(set(tups_train).intersection(tups_test)))}')
    print(f'INTERSECTION ALL TRAIN: {len(list(set(tups_train).intersection(tups_train_all)))}')
    print(f'INTERSECTION ALL TEST: {len(list(set(tups_test).intersection(tups_test_all)))}')

    print(test_bipartite_network)
    test_bipartite_network = test_bipartite_network.to_homogeneous().to(device)

    print(test_bipartite_network)

    in_channels, out_channels = train_bipartite_network.num_features, 16 # 16 was not stated in the paper
    model = GVAEPaperAutoEncoder(in_channels, hidden1, hidden2, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()
        loss = 0


        zp, mu1, log1 = model.encode_paper(train_bipartite_network.x, train_bipartite_network.edge_index,type)
        loss_paper = model.recon_loss(zp, train_bipartite_network.edge_index)
        loss = loss_paper + (1 / train_bipartite_network.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        return float(loss)

    def validation():
        model.eval()
        with torch.no_grad():
            optimizer.zero_grad()



            zp, mu1, log1 = model.encode_paper(validation_bipartite_network.x, validation_bipartite_network.edge_index,type)
            loss_paper = model.recon_loss(zp, validation_bipartite_network.edge_index)
            loss = loss_paper + (1 / validation_bipartite_network.num_nodes) * model.kl_loss()


            return float(loss)

    times=[]
    early_stopping_counter = 0
    max_count = 100
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train()

        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


    model.eval()
    with torch.no_grad():
        z,mu,log = model.encode_paper(test_bipartite_network.x, test_bipartite_network.edge_index,type)
        print(len(list(set(test_bipartite_network.edge_index[0].tolist()))))
        recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_bipartite_network.edge_label_index[0].tolist()))),:],z[num_training_papers:,:].t()))

        top_values, top_indices = torch.topk(recon, k=20, dim=1)
        sorted_recon = top_indices.tolist()

        edge_index = test_bipartite_network.edge_label_index.t().tolist()

        edge_index = sorted(edge_index, key=lambda x: x[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.tolist()
        edge_index_dict = {}
        for i, pair in enumerate(edge_index):
            if pair[0] < num_training_papers and pair[1] >= num_training_papers:
                if edge_index_dict.get(pair[0], None) is not None:
                    edge_index_dict[pair[0]].append(pair[1] - num_training_papers)
                else:
                    edge_index_dict[pair[0]] = [pair[1] - num_training_papers]
                edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))

        edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))

        # Sort the tensor along dimension 1 and get the indices

        # Check the sorted indices
        json_res = {}
        string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}_{type}'
        for k in [1, 5, 10]:
            print('RESULTS: ', k)
            precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
            print(ndcg, precision, recall)
            json_res[f'ndcg_{str(k)}'] = str(ndcg)
            json_res[f'precision_{str(k)}'] = str(precision)
            json_res[f'recall_{str(k)}'] = str(recall)
            print('\n\n')
        f = open(f'./bip_net_{string}_enriched.json', 'w')
        json.dump(json_res, f, indent=4)


def run_reduced():
    seed = args.seed

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print('TRAINING')

    data = BipartiteDataset(root=f'{args.path}')
    train_dataset = data[0]
    num_training_papers = train_dataset['publication'].num_nodes
    print(train_dataset)
    print(num_training_papers)


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
    tups_train_all = train_bipartite_network['publication', 'cites', 'dataset'].edge_index.t().tolist()
    tups_train_all = [tuple(e) for e in tups_train_all]
    print(train_bipartite_network)
    train_bipartite_network = train_bipartite_network.to_homogeneous().to(device)
    print(train_bipartite_network)


    print('TEST')

    test_dataset = BipartiteDataset(root=f'{args.path}')
    test_bipartite_network = test_dataset[0]
    del test_bipartite_network['publication', 'cites', 'publication']
    del test_bipartite_network['dataset', 'cites', 'dataset']
    test_bipartite_network['publication', 'cites', 'dataset'].edge_index = test_bipartite_network[
        'publication', 'cites', 'dataset'].edge_index_test
    test_bipartite_network['publication', 'cites', 'dataset'].edge_label_index = test_bipartite_network[
        'publication', 'cites', 'dataset'].edge_label_index_test
    tups_test = test_bipartite_network['publication', 'cites', 'dataset'].edge_label_index.t().tolist()
    tups_test = [tuple(e) for e in tups_test]
    tups_test_all = test_bipartite_network['publication', 'cites', 'dataset'].edge_index.t().tolist()
    tups_test_all = [tuple(e) for e in tups_test_all]
    print(f'INTERSECTION: {len(list(set(tups_train).intersection(tups_test)))}')
    print(f'INTERSECTION ALL TRAIN: {len(list(set(tups_train).intersection(tups_train_all)))}')
    print(f'INTERSECTION ALL TEST: {len(list(set(tups_test).intersection(tups_test_all)))}')

    print(test_bipartite_network)
    test_bipartite_network = test_bipartite_network.to_homogeneous().to(device)

    print(test_bipartite_network)

    in_channels, out_channels = train_bipartite_network.num_features, 16 # 16 was not stated in the paper
    model = GVAEPaperAutoEncoder(in_channels, hidden1, hidden2, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    def train():
        model.train()
        optimizer.zero_grad()
        loss = 0


        zp, mu1, log1 = model.encode_paper(train_bipartite_network.x, train_bipartite_network.edge_index,type)
        loss_paper = model.recon_loss(zp, train_bipartite_network.edge_index)
        loss = loss_paper + (1 / train_bipartite_network.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()
        return float(loss)

    def validation():
        model.eval()
        with torch.no_grad():
            optimizer.zero_grad()



            zp, mu1, log1 = model.encode_paper(validation_bipartite_network.x, validation_bipartite_network.edge_index,type)
            loss_paper = model.recon_loss(zp, validation_bipartite_network.edge_index)
            loss = loss_paper + (1 / validation_bipartite_network.num_nodes) * model.kl_loss()


            return float(loss)

    times=[]
    early_stopping_counter = 0
    max_count = 100
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train()

        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


    model.eval()
    with torch.no_grad():
        z,mu,log = model.encode_paper(test_bipartite_network.x, test_bipartite_network.edge_index,type)
        print(len(list(set(test_bipartite_network.edge_index[0].tolist()))))
        recon = torch.sigmoid(torch.matmul(z[sorted(list(set(test_bipartite_network.edge_label_index[0].tolist()))),:],z[num_training_papers:,:].t()))

        top_values, top_indices = torch.topk(recon, k=20, dim=1)
        sorted_recon = top_indices.tolist()

        edge_index = test_bipartite_network.edge_label_index.t().tolist()

        edge_index = sorted(edge_index, key=lambda x: x[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.tolist()
        edge_index_dict = {}
        for i, pair in enumerate(edge_index):
            if pair[0] < num_training_papers and pair[1] >= num_training_papers:
                if edge_index_dict.get(pair[0], None) is not None:
                    edge_index_dict[pair[0]].append(pair[1] - num_training_papers)
                else:
                    edge_index_dict[pair[0]] = [pair[1] - num_training_papers]
                edge_index_dict[pair[0]] = list(set(edge_index_dict[pair[0]]))

        edge_index_dict = dict(sorted(edge_index_dict.items(), key=lambda x: x[0]))

        # Sort the tensor along dimension 1 and get the indices

        # Check the sorted indices
        json_res = {}
        string = f'{dataset}_{lr}_{decay}_{hidden1}_{hidden2}_{type}'
        for k in [1, 5, 10]:
            print('RESULTS: ', k)
            precision, recall, ndcg = evaluation(sorted_recon, edge_index_dict, k=k)
            print(ndcg, precision, recall)
            json_res[f'ndcg_{str(k)}'] = str(ndcg)
            json_res[f'precision_{str(k)}'] = str(precision)
            json_res[f'recall_{str(k)}'] = str(recall)
            print('\n\n')
        f = open(f'.bip_net_{string}_reduced.json', 'w')
        json.dump(json_res, f, indent=4)


if __name__ == '__main__':

    epochs = 500
    lrs = [0.01,0.001]
    hiddens = [(64,32),(128,64),(128,32)]
    decays = [1e-5,5e-5,1e-4,1e-6]

    for dataset in ['mes','pubmed_kcore','pubmed']:
        if dataset == 'pubmed':
            hiddens.append((256,128))
            hiddens.append((256, 64))
        for type in ['linear','gcn','sage']:

            for iteration in range(20):
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

    # hidden1 = 128
    # hidden2 = 64
    # hidden1_mlp = 128 # not stated in the paper
    # hidden2_mlp = 64 # not stated in the paper
    # decay = 1e-5
    # lr = 0.01
    # epochs = 200
    # import pandas as pd
    # bip_net = 'linear'
    # dataset = args.dataset
    # type = args.type
    # print(type)
    #
    # run()


