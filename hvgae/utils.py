import pandas as pd

import torch_geometric.transforms as T

def write_qrels(docs):
    # ogni elemento è una tupla (topic,doc)
    head = 'queryId\t0\tdocumentId\trelevance\n'
    rows = []
    for row in docs:
        line = str(row[0])+'\t'+'0'+'\t'+str(row[1])+'\t'+'1'+'\n'
        rows.append(line)
    g = open('data/mes/qrels_file.qrels', 'w')
    for r in rows:
        g.write(r)

def write_run(run,path):
    head = 'queryId\tQ0\tdocumentId\trank\tscore\ttag\n'
    rows = []

    for i in range(len(run)): # query
        runf = sorted(run[i], key=lambda x: x[1], reverse=True)
        runf = runf[0:100]

        for r in runf:
            line = str(i) + '\t' + 'Q0' + '\t' + str(r[0]) + '\t' + str(runf.index(r)) +'\t' + str(r[1])+'\t'+'runid'+'\n'
            rows.append(line)

    f = open(path, 'w')

    for r in rows:
        f.write(r)

