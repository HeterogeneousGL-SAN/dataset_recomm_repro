import json
import pandas as pd
import tqdm
import argparse
import re
import os
import argparse
from rdflib import Graph
from rdflib_hdt import HDTStore
from rdflib.namespace import FOAF
import json
import multiprocessing
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
if not os.path.isdir(args['data']):
    raise ValueError("Please input correct directory path. -data/--data [path_to_dir]")
data_dir = os.path.abspath(args['data'])+'/mes_data'
from pykeen.models import ComplEx

from rdflib import Graph, Literal, URIRef, Namespace
from hdt import HDTDocument
import csv
import re
import pandas as pd
import pykeen
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mes',
                    choices=['Cora', 'CiteSeer','mes','pubmed_kcore','pubmed'])

parser.add_argument('--path_dataset', type=str)

args = parser.parse_args()
path_dataset = args.path_dataset


def form_nt_title_files():
    pub_csv = pd.read_csv(path_dataset+'/publications_all.csv')
    dat_csv = pd.read_csv(data_dir+'/datasets_all.csv')

    g = Graph()
    ex = Namespace("http://example.org/")

    nt_file = open(path_dataset+'/ensemble_cn_data/Paper.nt','w')
    c = 0
    for index,row in pub_csv.iterrows():
        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['id']))
        print(s)
        s = URIRef(ex[s])
        o = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['title']))
        g.add((URIRef(s), ex.Title, Literal(o)))

    for index,row in dat_csv.iterrows():
        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['id']))
        print(s)
        s = URIRef(ex[s])
        o = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['title']))

        g.add((URIRef(s), ex.Title, Literal(o)))
    nt_data = g.serialize(format='nt').decode('utf-8')
    nt_file.write(nt_data)

def form_nt_abstract_files():
    pub_csv = pd.read_csv(path_dataset+'/publications_all.csv')
    dat_csv = pd.read_csv(data_dir+'/datasets_all.csv')

    g = Graph()
    ex = Namespace("http://example.org/")

    nt_file = open(path_dataset+'/ensemble_cn_data/PaperAbs.nt','w')
    c = 0
    for index, row in pub_csv.iterrows():
        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['id']))
        print(s)
        s = URIRef(ex[s])
        o = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['description']))
        g.add((URIRef(s), ex.Title, Literal(o)))

    for index, row in dat_csv.iterrows():
        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['subject']))
        print(s)
        s = URIRef(ex[s])
        o = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['description']))
        g.add((URIRef(s), ex.Description, Literal(o)))

    nt_data = g.serialize(format='nt').decode('utf-8')
    nt_file.write(nt_data)

def form_nt_links_files():
    edge_csv = pd.read_csv(path_dataset+'/pubdataedges_all.csv')
    edges_cn = pd.read_csv(path_dataset+'/pubpubedges.csv')




    g = Graph()
    ex = Namespace("http://example.org/")
    nt_file = open(path_dataset+'/ensemble_cn_data/StandardSchLink.nt','w')
    c = 0
    for index,row in edges.iterrows():

        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['source']))
        s = URIRef(ex[s])
        o = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['target']))
        o = URIRef(ex[o])
        g.add((s, ex.LinkedTo, o))




    for index,row in edges_cn.iterrows():
        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['source']))
        s = URIRef(ex[s])
        o = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['target']))
        o = URIRef(ex[o])
        g.add((s, ex.LinkedTo, o))

    nt_data = g.serialize(format='nt').decode('utf-8')
    nt_file.write(nt_data)


def form_nt_authors_files():
    pub_to_data_csv_0 = pd.read_csv(path_dataset+'/pubauthedges.csv')
    pub_to_data_csv = pd.read_csv(path_dataset+'/dataauthedges.csv')
    pub_to_data_csv = pd.concat([pub_to_data_csv_0,pub_to_data_csv],ignore_index=True)


    g = Graph()
    ex = Namespace("http://example.org/")

    nt_file = open(path_dataset+'/ensemble_cn_data/PaperAuthorAffiliations.nt','w')
    c = 0
    for index,row in pub_to_data_csv.iterrows():

        # source is the publication, target is the author
        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['target']))
        s = URIRef(ex[s])
        o = row['source'][1:-1]
        o = re.sub(r'[^a-zA-Z0-9 ]', '', o)
        o = URIRef(ex[o])
        g.add((s, ex.Creator, o))

    nt_data = g.serialize(format='nt').decode('utf-8')
    nt_file.write(nt_data)

def form_nt_authors_names_files():
    pub_to_data_csv = pd.read_csv(path_dataset+'/authors.csv')


    g = Graph()
    ex = Namespace("http://example.org/")
    nt_file = open(path_dataset+'/ensemble_cn_data/AuthorsNames.nt','w')
    c = 0
    for index,row in pub_to_data_csv.iterrows():


        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['id']))
        s = URIRef(ex[s])
        o = row['fullname'][1:-1]
        g.add((s, ex.Author, Literal(o)))

    nt_data = g.serialize(format='nt').decode('utf-8')
    nt_file.write(nt_data)


def create_qrels_file():
    pub_to_data_csv = pd.read_csv(path_dataset+'/pubdataedges_all.csv')

    pub_to_pub_csv = pd.read_csv(path_dataset+'/pubpubedges.csv')
    ex = Namespace("http://example.org/")

    nt_file = open(data_dir+'/raw_data/mes.qrels','w')
    nt_file.write('QueryID'+'\t'+'0'+'\t'+'DocID'+'\t'+'Relevance'+'\n')

    for index,row in pub_to_data_csv.iterrows():

        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['subject']))
        s = URIRef(ex[s])
        o = row['object'][1:-1]
        o = re.sub(r'[^a-zA-Z0-9 ]', '', o)
        o = URIRef(ex[o])
        line = str(s) + '\t'+'Q0'+'\t'+str(o) +'\t'+ '1'+'\n'
        nt_file.write(line)



    for index,row in pub_to_pub_csv.iterrows():
        s = re.sub(r'[^a-zA-Z0-9 ]', '', str(row['subject']))
        s = URIRef(ex[s])
        o = row['object'][1:-1]
        o = re.sub(r'[^a-zA-Z0-9 ]', '', o)
        o = URIRef(ex[o])
        line = str(s) + '\t'+'Q0'+'\t'+str(o) +'\t'+ '1'+'\n'
        nt_file.write(line)

def main():
    print('writing titles...')
    form_nt_title_files()

    print('writing abstracts...')
    form_nt_abstract_files()

    print('writing links...')
    form_nt_links_files()

    print('writing authors...')
    form_nt_authors_files()
    form_nt_authors_names_files()

    print('writing qrels...')
    create_qrels_file()


if __name__ == '__main__':
    main()


