from rank_bm25 import BM25Okapi
import argparse
import os.path as osp
import time
import tqdm
import torch
import os
import json
import numpy as np
import random
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import ndcg_score
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='mes',
                    choices=['mes','pubmed_kcore','pubmed'],help="The selected dataset")
parser.add_argument('-seed',default=42)
parser.add_argument('-path',type=str,help='path to data')

# path = ''./datasets/{args.dataset}/split_transductive/test'

args = parser.parse_args()
path = args.path

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
g = torch.Generator()
g.manual_seed(args.seed)


def evaluation(rankings, true_labels,k = 10):
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

    precision, recall, ndcg = 0, 0, 0
    for key,v in rankings.items():
        true_vals = true_labels[key]
        predicted = v
        predicted = predicted[:k]
        correct = list(set(true_vals) & set(predicted))
        precision += len(correct) / k
        recall += len(correct) / len(true_vals)
        ndcg += ndcg_at_k(predicted, true_vals, k)
    return precision/len(rankings), recall/len(rankings), ndcg/len(rankings)

def get_corpus():
    dataset = args.dataset
    data_list = pd.read_csv(f'{path}/pubdataedges_all.csv')['target'].unique().tolist()
    datasets = pd.read_csv(f'{path}/datasets_all.csv')
    datasets = datasets[datasets['id'].isin(data_list)]
    datasets_ids = datasets['id'].tolist()
    datasets_corpus = datasets['description'].tolist()
    # print(len(datasets_ids))
    # print(datasets_corpus[0])

    return datasets_ids,datasets_corpus

def get_queries():
    dataset = args.dataset
    pub_list = pd.read_csv(f'{path}/pubdataedges_test.csv')['source'].unique().tolist()
    publications = pd.read_csv(f'{path}/publications_all.csv')
    publications = publications[publications['id'].isin(pub_list)]
    publications_ids = publications['id'].tolist()
    publications_corpus = publications['description'].tolist()
    # print(len(publications_ids))
    # print(publications_corpus[0])

    return publications_ids,publications_corpus




def process(text):
    porter_stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    # Stem the tokens
    tokens = [porter_stemmer.stem(token) for token in filtered_tokens]
    return tokens

def main():
    datasets_ids, datasets_corpus = get_corpus()
    publications_ids,publications_corpus = get_queries()
    tokenized_corpus = [process(doc) for doc in datasets_corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    pd_edges = pd.read_csv(f'{path}/pubdataedges_test.csv')
    pd_edges = pd_edges.groupby('source')['target'].apply(lambda x: ' '.join(x)).reset_index()
    edge_index_dict = {row['source']: row['target'].split() for i, row in pd_edges.iterrows()}
    edge_index_dict = {key: edge_index_dict[key] for key in sorted(edge_index_dict.keys())}


    rankings = {}
    true_values = {}
    for i,query in enumerate(publications_corpus):
        print(publications_ids[i])
        tokenized_query = process(query)
        doc_scores = bm25.get_scores(tokenized_query)
        doc_scores = [(i,score) for i,score in enumerate(doc_scores)]
        doc_scores = sorted(doc_scores,key=lambda x: x[1],reverse=True)
        # print(doc_scores[0:10])
        doc_scores = [(datasets_ids[tup[0]],tup[1]) for tup in doc_scores][0:20]
        # print(doc_scores[0:10])
        doc_scores_datasets = [d[0] for d in doc_scores]
        rankings[publications_ids[i]] = doc_scores_datasets
        true_values[publications_ids[i]] = edge_index_dict[publications_ids[i]]
        # print(edge_index_dict[publications_ids[i]])

    for k in [1, 5, 10]:
        print('RESULTS: ', k)
        precision, recall, ndcg = evaluation(rankings, true_values, k=k)
        print(ndcg, precision, recall)
        print('\n\n')


if __name__ == '__main__':
    main()




