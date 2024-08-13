import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='mes',choices=['mes','pubmed','pubmed_kcore'],
                    type=str)
parser.add_argument("--cutoff",default="1,5,10")
parser.add_argument("--path",default="1,5,10")
import time
import shutil

args = parser.parse_args()
path = args.path

def concatenate_targets(group):
    return ' '.join(group)


def get_data(dataset):
    pdedges_all = pd.read_csv(f'{args.path}/pubdataedges.csv', low_memory=False)
    publications_all = pd.read_csv(f'.{args.path}/publications_all.csv')
    datasets_all = pd.read_csv(f'{args.path}/datasets_all.csv', low_memory=False)
    datasets_all = datasets_all[datasets_all['id'].isin(pdedges_all['target'].unique().tolist())]
    publications_all = publications_all[publications_all['id'].isin(pdedges_all['source'].unique().tolist())]
    return publications_all,datasets_all


def create_dataset_search_collection(dataset):
    publications,datasets = get_data(dataset)
    f = open(f'./dataset_search_collection.jsonl','w')
    for i,d in datasets.iterrows():
        if d['id'] == 'd_173427':
            print('trovato')
        json_obj = {}
        json_obj['id'] = d['id']
        json_obj['contents'] = d['title']+' ' +d['description']
        json_obj['title'] = d['title']
        json_obj['variants'] = [d['id']]
        json_obj['structured_info'] = d['title'] + ' '+ d['description']
        f.write(json.dumps(json_obj) + '\n')


def create_qrels(dataset):
    f = open(f'./test_dataset_collection.qrels','w')
    df = pd.read_csv(f'{args.path}/pubdataedges_kcore_train.csv')
    print(df.shape[0])
    df = pd.read_csv(f'{args.path}/pubdataedges_kcore_validation.csv')
    print(df.shape[0])
    df = pd.read_csv(f'{args.path}/pubdataedges_kcore_test.csv')
    print(df.shape[0])
    head = 'QueryID\t0\tDocID\tRelevance\n'
    f.write(head)
    publications,datasets = get_data(dataset)

    for i,row in df.iterrows():
        pub = publications[publications['id'] == row['source']]
        query = str(pub['title'].iloc[0]) + ' ' + str(pub['description'].iloc[0])
        query = '_'.join(query.split())
        line = f"{query}\tQ0\t{row['target']}\t1\n"
        f.write(line)

import random
def create_train_data(dataset):
    publications,datasets = get_data(dataset)
    f = open(f'./train_data.jsonl','w')
    df = pd.read_csv(f'{args.path}/pubdataedges_kcore_train.csv')
    df = df.groupby('source')['target'].agg(concatenate_targets).reset_index()

    for i,row in df.iterrows():
        json_obj = {}
        pub = publications[publications['id'] == row['source']]
        json_obj['documents'] = row['target'].split()
        json_obj['paper_id'] = pub['id'].iloc[0]
        json_obj['title'] = pub['title'].iloc[0]
        json_obj['abstract'] = pub['description'].iloc[0]
        query = str(pub['title'].iloc[0]) + ' ' + str(pub['description'].iloc[0])
        json_obj['query'] = json_obj['keyphrase_query'] = query
        json_obj['positives'] = row['target'].split()

        def find_negatives():
            datasets_filtered = datasets[~datasets['id'].isin(json_obj['positives'])]

            datasets_ids = datasets_filtered['id'].unique().tolist()
            random_elements = random.sample(datasets_ids, 10)
            return random_elements

        json_obj['negatives'] = find_negatives()

        f.write(json.dumps(json_obj) + '\n')

def create_test_data(dataset):
    publications,datasets = get_data(dataset)
    f = open(f'./test_data.jsonl','w')
    df = pd.read_csv(f'{args.path}/pubdataedges_kcore_test.csv')
    df = df.groupby('source')['target'].agg(concatenate_targets).reset_index()

    for i,row in df.iterrows():
        json_obj = {}
        pub = publications[publications['id'] == row['source']]
        json_obj['documents'] = row['target'].split()
        query = str(pub['title'].iloc[0]) + ' ' + str(pub['description'].iloc[0])
        json_obj['abstract'] = str(pub['description'].iloc[0])
        json_obj['id'] = str(pub['id'].iloc[0])
        date = 200
        try:
            date = int(pub['date'].iloc[0][0:4])
        except Exception as e:
            pass
        json_obj['year'] = date
        json_obj['query'] = json_obj['keyphrase_query'] = query
        f.write(json.dumps(json_obj) + '\n')

def main():
    args = parser.parse_args()
    dataset = args.dataset
    create_qrels(dataset)
    create_dataset_search_collection(dataset)
    create_train_data(dataset)
    create_test_data(dataset)

if __name__ == '__main__':
    main()
