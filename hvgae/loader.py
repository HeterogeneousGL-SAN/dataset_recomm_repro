import os.path as osp
import pandas as pd
import torch_geometric.utils.convert
from sentence_transformers import SentenceTransformer
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.data import Data, InMemoryDataset
import pickle
import argparse
from gensim.models import KeyedVectors
import multiprocessing as mp
import json
import random
import torch
import numpy as np
import utils



class MiniBatchLoader:
    def __init__(self, edge_index, batch_size, shuffle=True):
        self.edge_index = edge_index.t().tolist()
        self.batch_size = batch_size


def cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_vector1 = np.linalg.norm(emb1)
    norm_vector2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    print(path, df.shape[0])
    if 'content' in list(df.columns):
        df['content'] = df['content'].astype(str)
    if 'name' in list(df.columns):
        df['name'] = df['name'].astype(str)
    if 'fullname' in list(df.columns):
        df['fullname'] = df['fullname'].astype(str)
    if 'description' in list(df.columns):
        df['description'] = df['description'].astype(str)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)


    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    print(path, df.shape[0])

    def switch_values(row):
        if row['source'].startswith('d_') and row['target'].startswith('p_'):
            return pd.Series({'source': row['target'], 'target': row['source']})
        else:
            return row

    df = df.apply(switch_values, axis=1)
    df = df.drop_duplicates(keep='first')

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class ContentEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class KeywordEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='whaleloops/phrase-bert', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class BipartiteDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if 'transductive' in self.root:
            return ['graph_baseline_hvgae_10.pt']

        # elif 'inductive_full' in self.root:
        #     return ['graph_baseline_hvgae.pt']
        #
        # elif 'inductive_light' in self.root:
        #     return ['graph_baseline_hvgae.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        dataset = self.root.split('/')[1]
        print(f'DATASET: {dataset}')
        if 'transductive' in self.root:
            data_pre = self.create_transductive_graph()
            data_list = [data_pre]
        elif 'inductive_light' in self.root:
            data_train = self.create_inductive_graph()
            # data_vali = self.create_inductive_graph()
            # data_test = self.create_inductive_graph()
            data_list = [data_train]
        elif 'inductive_full' in self.root:
            data_train = self.create_inductive_graph()
            # data_vali = self.create_inductive_graph()
            # data_test = self.create_inductive_graph()
            data_list = [data_train]

        print(self.root)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # print(self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

    def create_transductive_graph(self):
        publication_path = self.root + '/publications_all.csv'
        dataset_path = self.root + '/datasets_all.csv'


        publication_x, publication_mapping = load_node_csv(publication_path, index_col='id',
                                                                              encoders={'content': ContentEncoder()})
        dataset_x, dataset_mapping = load_node_csv(dataset_path, index_col='id',
                                                                  encoders={'content': ContentEncoder()})

        pd_edges_path_train = self.root + '/pubdataedges_kcore_train.csv'
        pd_edges_path_vali = self.root + '/pubdataedges_kcore_validation.csv'
        pd_edges_path_test = self.root + '/pubdataedges_kcore_test.csv'
        pp_edges_path = self.root + '/pubpubedges.csv'
        dd_edges_path = self.root + '/datadataedges.csv'

        pd_edge_index_train, pd_edge_label = load_edge_csv(
            pd_edges_path_train,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )
        pd_edge_index_validation, pd_edge_label = load_edge_csv(
            pd_edges_path_vali,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )
        pd_edge_index_test, pd_edge_label = load_edge_csv(
            pd_edges_path_test,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )



        pp_edge_index, pp_edge_label = load_edge_csv(
            pp_edges_path,
            src_index_col='source',
            src_mapping=publication_mapping,
            dst_index_col='target',
            dst_mapping=publication_mapping,
        )

        dd_edge_index, dd_edge_label = load_edge_csv(
            dd_edges_path,
            src_index_col='source',
            src_mapping=dataset_mapping,
            dst_index_col='target',
            dst_mapping=dataset_mapping,
        )


        data = HeteroData()
        data['publication'].x = publication_x
        data['publication'].num_nodes = len(publication_mapping)  # Users do not have any features.

        data['dataset'].num_nodes = len(dataset_mapping)  # Users do not have any features.
        data['dataset'].x = dataset_x


        # leave part of edges for message passing
        data['publication', 'cites', 'dataset'].edge_index_train = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_train = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_index_val = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_val = pd_edge_index_validation
        data['publication', 'cites', 'dataset'].edge_index_test = pd_edge_index_train
        data['publication', 'cites', 'dataset'].edge_label_index_test = pd_edge_index_test
        

        reduced = []
        test_nodes_s,test_nodes_t = pd_edge_index_test[0].tolist(),pd_edge_index_test[1].tolist()
        training_edges = pd_edge_index_train.t().tolist()
        for e in training_edges:
            if e[0] in test_nodes_s or e[0] in test_nodes_t or e[1] in test_nodes_s or e[1] in test_nodes_t:
                reduced.append(e)
        print(len(reduced))
        print(pd_edge_index_test.shape)
        reduced = torch.tensor(reduced).t()
        data['publication', 'cites', 'dataset'].edge_index_train_reduced = reduced
        data['publication', 'cites', 'dataset'].edge_label_index_train_reduced = reduced
        
        data['publication', 'cites', 'publication'].edge_index = pp_edge_index
        data['publication', 'cites', 'publication'].edge_label = pp_edge_label
        data['dataset', 'cites', 'dataset'].edge_index = dd_edge_index
        data['dataset', 'cites', 'dataset'].edge_label = dd_edge_label



        return data

