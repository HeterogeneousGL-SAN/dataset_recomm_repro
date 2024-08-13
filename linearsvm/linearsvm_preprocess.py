import pandas as pd
import json
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mes',
                    choices=['mes','pubmed_kcore','pubmed'])

parser.add_argument('--path_dataset', type=str)

args = parser.parse_args()
path_dataset = args.path_dataset

def mapping():
    map_datasets = path_dataset+'/datasets_all.csv'
    map_publications = path_dataset+'/publications_all.csv'
    pubdataedges_test = pd.read_csv(path_dataset+'pubdataedges_kcore_test.csv')

    df_pub_train = pd.read_csv(map_publications)
    df_pub_test = pd.read_csv(map_publications)['id'].isin(pubdataedges_test['source'].tolist())
    df_pub = pd.concat([df_pub_test,df_pub_train],ignore_index=True)
    df_data = pd.read_csv(map_datasets)


    pub_map = {r['id'].replace('"',''):'p_'+str(i) for i,r in df_pub.iterrows()}
    dat_map = {r['id'].replace('"',''):'d_'+str(i) for i,r in df_data.iterrows()}
    return pub_map,dat_map


def create_corpus():
    # line: id \t abstract \t comma separated datasets
    pub_map, dat_map = mapping()
    all_pubs = path_dataset+'/publications_all.csv'
    edges_train = path_dataset+'/pubdataedge_kcore_train.csv'
    edges_test = path_dataset+'/pubdataedge_kcore_test.csv'

    df_pub_train = pd.read_csv(all_pubs)['id'].isin(pd.read_csv(edges_train)['source'].tolist)
    df_pub_test = pd.read_csv(all_pubs).isin(pd.read_csv(edges_test)['source'].tolist)
    df_pub = pd.concat([df_pub_test,df_pub_train],ignore_index=True)
    df_edges = pd.concat([edges_test,edges_train],ignore_index=True)

    lines = []
    for i,r in df_pub:
        id = r['id'].replace('"','')
        rows = df_edges[df_edges['source'].replace('"','') == id]
        abstract = r['description']
        datasets = ', '.join(rows['target'].replace('"','').unique().tolist())
        id = pub_map[id]
        datasets = [dat_map[id] for id in datasets]
        line = id+'\t'+abstract+'\t'+datasets+'\n'
        lines.append(line)
    f = open(path_dataset+'linearsvm_data/Abstract_New_Database.txt', 'w')
    for line in lines:
        f.write(line)
    f.close()

def create_datasets_list():
    datasets = path_dataset + '/datasets_all.csv'
    datasets = pd.read_csv(datasets)
    titles = datasets['titles'].unique().tolist()

    f = open(path_dataset+'linearsvm_data/Dataset_Titles.txt', 'w')
    for line in titles:
        f.write(line+'\n')
    f.close()


def create_training_test_set():
    json_dict = {"training":{"queries":[],"query_ids":[],"datasets":[]},"test":{"queries":[],"query_ids":[],"datasets":[]}}
    publications = path_dataset+'/publications_all.csv'
    pubdataedges_train = pd.read_csv(path_dataset+'pubdataedges_kcore_train.csv')
    publications = pd.read_csv(publications)

    training_pubs = path_dataset+'/training/raw/publications.csv'
    test_pubs = path_dataset+'/test/raw/publications.csv'
    f = open(path_dataset + 'linearsvm_data/Abstract_New_Database.txt', 'r')
    lines = f.readlines()


    training_pubs = pd.read_csv(pubdataedges_train)['id'].isin(pubdataedges_test['source'].tolist())
    test_pubs = pd.read_csv(test_pubs)
    test_pubs_num = test_pubs.shape[0]
    training_pubs_num = training_pubs.shape[0]
    lines_test = lines[:test_pubs_num]
    lines_train = lines[test_pubs_num:training_pubs_num]


    training_queries = [line.split()[1] for line in lines_train]
    training_queries_ids = [line.split()[0] for line in lines_train]
    training_datasets = [line.split()[2].split(', ') for line in lines_train]
    test_queries = [line.split()[1] for line in lines_test]
    test_queries_ids = [line.split()[0] for line in lines_test]
    test_datasets = [line.split()[2].split(', ') for line in lines_test]

    json_dict["training"]["queries"] = training_queries
    json_dict["training"]["query_ids"] = training_queries_ids
    json_dict["training"]["datasets"] = training_datasets
    json_dict["test"]["queries"] = test_queries
    json_dict["test"]["query_ids"] = test_queries_ids
    json_dict["test"]["datasets"] = test_datasets

    g = open(path_dataset+'linearsvm_data/split.json','w')
    json.dump(json_dict,g,indent=4)









def main():
    print('creating corpus...')
    create_corpus()

    print('create datasets list...')
    create_datasets_list()

    print('create training test sets...')
    create_training_test_set()

if __name__ == '__main__':
    main()