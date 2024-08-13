import random
from hdt import HDTDocument
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from gensim.parsing.preprocessing import preprocess_string
from tqdm import tqdm
import bz2
from rank_bm25 import BM25Okapi
import networkx as nx
import os
import argparse
import pathlib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

all_args = argparse.ArgumentParser()
all_args.add_argument("-th", "--threshold", required=False, default=float(0.7), type=float,
                      help="Threshold for similarity between author embedding")
all_args.add_argument("-bm25t", "--bm25_threshold", required=False, default=int(2), type=int,
                      help="Threshold for BM25 ranking")
all_args.add_argument("-hop", "--hop", required=False, default=int(3), type=int,
                      help="Hop number for graph walk")
all_args.add_argument("-data", "--data", required=True, type=pathlib.Path,
                      help="Path to directory contains all needed data")
all_args.add_argument("-out", "--out", required=False, default=os.path.dirname(os.getcwd()),
                      help="Directory to store all result files. Default is directory of this python file")
all_args.add_argument("-random", "--random", required=False, type=int,
                      help="Random select seeds from seed file. Recommended argument for large-scale seed file")
all_args.add_argument("-all", "--all", default=False,
                      help="Run with all hops and thresholds for author embedding similarity")
all_args.add_argument("-top", "--top", required=False, default=float(0.8), type=float,
                      help="Threshold for Bert and citation embedding similarity approaches")

args = vars(all_args.parse_args())
if not os.path.isdir(args['data']):
    raise ValueError("Please input correct directory path. -data/--data [path_to_dir]")
if float(args['threshold']) < 0.0 or float(args['threshold']) > 1.0:
    raise ValueError("Please input valid threshold (float/double) for similarity between entity(author) embedding")
if int(args['hop']) < 0 or int(args['hop']) > 3:
    raise ValueError("Please input hop number (int) for graph walk in range [1-3]")
if int(args['bm25_threshold']) < 0:
    raise ValueError("Please input threshold (int) for BM25 ranking >0")

check_list = ["seeds.txt","cands.txt","StandardSchLink.hdt","StandardSchLink.hdt.index.v1-1","mag_authors_2020_ComplEx_entity.npy","authors_entities.dict","Citation_vectors.txt","Paper.hdt","Paper.hdt.index.v1-1","PaperAbs.hdt","PaperAbs.hdt.index.v1-1","PaperAuthorAffiliations.hdt","PaperAuthorAffiliations.hdt.index.v1-1"]
dir_list = os.listdir(args['data'])
non_exist_files = []
for check in tqdm(check_list,desc="Checking data files in data path"):
    if check not in dir_list:
        non_exist_files.append(check)
if len(non_exist_files)!=0:
    raise Exception("Data files "+str(non_exist_files)+" is missing. Please recheck README.")

# optimize_sparql()
model = SentenceTransformer('all-mpnet-base-v2')


def graphwalk(G: nx.Graph, hop: int, seed, cand_list):
    walk_authors = set()
    if G.has_node(seed.replace("https://makg.org/entity/", "")):
        temp_set = set(
            nx.single_source_shortest_path(G, seed.replace("https://makg.org/entity/", ""), cutoff=(2 * hop)).keys())
        temp_set = set([str("https://makg.org/entity/" + t) for t in temp_set])
        walk_authors = cand_list.intersection(temp_set)
    return walk_authors


def read_seed_dataset(path):
    seed_datasets = set()
    with open(path) as seed_file:
        for line in seed_file:
            seed_datasets.add(line.strip())
    return seed_datasets


def read_candidate_dataset(path):
    dataset_list = set()
    with open(path) as seed_file:
        for line in seed_file:
            dataset_list.add(line.strip())
    return dataset_list


def reduce_candidates(seeds, candidates, standard_hdt):
    new_cands = set()
    for seed in seeds:
        (triples, cardinality) = standard_hdt.search_triples_bytes(seed, "", "")
        for subj, pred, obj in triples:
            s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
            if o in candidates:
                new_cands.add(o)
    return new_cands


def get_authors_based_on_datasets(dataset_list, dataset_author_hdt):
    author_list = set()
    for dataset in dataset_list:
        (triples, cardinality) = dataset_author_hdt.search_triples_bytes("", "", dataset)
        for subj, pred, obj in triples:
            s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
            author_list.add(s)
    return author_list


def get_all_seed_author(els_author_hdt):
    all_seed_author = set()
    (triples, cardinality) = els_author_hdt.search_triples_bytes("", "", "")
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        all_seed_author.add(o)
    return all_seed_author


def get_seed_author(seed_dataset, coauthor_hdt):
    seed_authors = set()
    (triples, cardinality) = coauthor_hdt.search_triples_bytes("", "", seed_dataset)
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        seed_authors.add(s)
    return seed_authors


def read_candidate_authors(sch_author_hdt):
    candidate_authors = set()
    (triples, cardinality) = sch_author_hdt.search_triples_bytes("", "", "")
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        candidate_authors.add(o)
    return candidate_authors


def cosine_np(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def read_author_embedding(author_list, ent_dict, makg_embed):
    ent_embed = {}
    for author in author_list:
        ent_embed[author] = makg_embed[ent_dict[author]]
    return ent_embed


def clean_candidate_with_ent_embed(seed_author, candidate_set, ent_embed, threshold=0.0):
    new_set = []
    seed_vec = ent_embed[seed_author]
    for candidate in candidate_set:
        candidate_vec = ent_embed[candidate]
        sim = cosine_np(seed_vec, candidate_vec)
        if sim >= threshold:
            new_set.append(candidate)
    return new_set


def get_standard(dataset, standard_hdt):
    standards = set()
    (triples, cardinality) = standard_hdt.search_triples_bytes(dataset, "", "")
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        standards.add(o)
    return standards


def get_bert_embed(dataset_dict):
    bert_embed = {}
    for dataset in dataset_dict.keys():
        bert_embed[dataset] = model.encode(dataset_dict[dataset], convert_to_numpy=True)
    return bert_embed


def bm25_build(candidate_text_list):
    tokens = [preprocess_string(text) for text in candidate_text_list]
    bm25 = BM25Okapi(tokens)
    return bm25


def bm25_rank(bm25, query_text):
    return bm25.get_scores(preprocess_string(query_text))


def read_vec_dict(path):
    vec_dict = {}
    with open(path) as rfile:
        for line in rfile:
            spline = line.split()
            if spline[1].startswith("entity/"):
                vec_dict[str(spline[1].replace("entity/", ""))] = int(spline[0])
    rfile.close()
    return vec_dict


def authors_to_embed_dict(author_list, vec_dict, fp):
    embed_dict = {}
    for author in author_list:
        embed_dict[author] = fp[vec_dict[author.replace("https://makg.org/entity/", "")]]
    return embed_dict


def authors_to_datasets(author_list, dataset_author_hdt):
    dataset_list = set()
    for author in author_list:
        (triples, cardinality) = dataset_author_hdt.search_triples_bytes(author, "", "")
        for subj, pred, obj in triples:
            s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
            dataset_list.add(o)
    return dataset_list


def pre_walk(hop, graph, author_list):
    walk_author_dict = {}
    for author in tqdm(author_list):
        walk_author_dict[author] = globals()['graphwalk_hop' + str(hop)](graph, author)
    return walk_author_dict


def read_walk(bzpath, author_list):
    dict_test = {str(a): set() for a in author_list}
    with bz2.BZ2File(bzpath) as file:
        for line in file:
            sline = "".join(chr(x) for x in line)
            sline = sline.split()
            if str(sline[0]) in author_list:
                dict_test[str(sline[0])].add(sline[2])
    return dict_test


def get_title(dataset, title_hdt):
    title = ""
    (triples, cardinality) = title_hdt.search_triples_bytes(dataset, "", "")
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        title = o.replace("\"", "")
        break
    return title


def get_abs(dataset, abs_hdt):
    title = ""
    (triples, cardinality) = abs_hdt.search_triples_bytes(dataset, "", "")
    for subj, pred, obj in triples:
        s, p, o = subj.decode('utf-8'), pred.decode('utf-8'), obj.decode('utf-8')
        title = o.replace("\"", "")
        break
    return title


def get_datasets_title_desc_with_dict(dataset_list, title_hdt, abs_hdt):
    dataset_dict = {}
    for dataset in dataset_list:
        dataset_dict[dataset] = get_title(dataset, title_hdt) + " " + get_abs(dataset, abs_hdt)
    return dict(enumerate(list(dataset_dict.keys()))), list(dataset_dict.values())


def get_datasets_title_desc(dataset_list, title_hdt, abs_hdt):
    dataset_dict = {}
    for dataset in dataset_list:
        dataset_dict[dataset] = get_title(dataset, title_hdt) + " " + get_abs(dataset, abs_hdt)
    return dataset_dict


def get_ciataion_embed_dict(dataset_list, vec_path):
    # non è detto che qua i candidates ci siano tutti
    dataset_vec = {}
    vec_file = open(vec_path)
    gived_ids = []
    for line in vec_file:
        sline = line.split()
        id = sline[0].replace("<", "").replace(">", "")
        if id in dataset_list:
            gived_ids.append(id)
            dataset_vec[id] = [float(a) for a in sline[1:50]]
    missing_ids = [i for i in dataset_list if i not in gived_ids]
    print('missing ids')
    for m in missing_ids:
        print(m)
    return dataset_vec


def get_frommakg_dict(doi_match):
    frommakg_dict = {}
    match_file = open(doi_match)
    for line in match_file:
        sline = line.split()
        frommakg_dict[sline[1]] = sline[0]
    return frommakg_dict


def get_tomakg_dict(doi_match):
    frommakg_dict = {}
    match_file = open(doi_match)
    for line in match_file:
        sline = line.split()
        frommakg_dict[sline[0]] = sline[1]
    return frommakg_dict


def get_datasets_with_citation(tomakg_dict: dict, ciatation_embed: dict):
    dataset_list = set()
    for item in list(tomakg_dict.items()):
        if item[1] in ciatation_embed.keys():
            dataset_list.add(item[0])
    return dataset_list


def loop_build_graph(G: nx.Graph, step: int, node, coauthor_hdt):
    if step % 2 == 0:
        (triples, cardinality) = coauthor_hdt.search_triples_bytes(
            node, "", "")
        for sub, _, obj in triples:
            s, o = sub.decode('utf-8'), obj.decode('utf-8')
            G.add_edge(s.replace("https://makg.org/entity/", ""), o.replace("https://makg.org/entity/", ""))
            step += 1
            if step < 3:
                loop_build_graph(G, step, o, coauthor_hdt)
    elif step % 2 == 0:
        (triples, cardinality) = coauthor_hdt.search_triples_bytes(
            "", "", node)
        for sub, _, obj in triples:
            s, o = sub.decode('utf-8'), obj.decode('utf-8')
            G.add_edge(s.replace("https://makg.org/entity/", ""), o.replace("https://makg.org/entity/", ""))
            step += 1
            if step < 3:
                loop_build_graph(G, step, s, coauthor_hdt)


def step(i, seed, standard_hdt, coauthor_hdt, Graph, candidate_author, all_author, threshold, result_w, result_w1,
         result_w2, result_w3, result_w4, seed_bm, bm25_index, cand_bm_dict, ciatation_embed, seed_bert, result_w5,
         result_w6, result_w7, result_w8, bert_percent, citation_percent, bm_t):
    walk_dataset = set()
    walk_dataset_embed = set()
    standards = get_standard(seed, standard_hdt)
    seed_authors = get_seed_author(seed, coauthor_hdt)
    assert len(seed_authors) > 0
    for seed_author in seed_authors:
        walk_author = graphwalk(Graph, i, seed_author, candidate_author)
        walk_author = set([element for element in walk_author if element != None])
        walk_author = walk_author.intersection(candidate_author)
        walk_dataset.update(authors_to_datasets(walk_author, coauthor_hdt))
        walk_author = clean_candidate_with_ent_embed(seed_author, walk_author, all_author, threshold)
        walk_dataset_embed.update(authors_to_datasets(walk_author, coauthor_hdt))

    G = float(len(standards))
    correct_dataset = standards.intersection(walk_dataset)
    T = float(len(correct_dataset))
    N = float(len(walk_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w.flush()
    correct_dataset = standards.intersection(walk_dataset_embed)
    T = float(len(correct_dataset))
    N = float(len(walk_dataset_embed))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w1.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w1.flush()
    search_tokens = preprocess_string(seed_bm[seed])
    scores = bm25_index.get_scores(search_tokens)
    bm_results = {}
    for i in range(len(cand_bm_dict)):
        bm_results[cand_bm_dict[i]] = scores[i]
    bm_results = dict(sorted(bm_results.items(), key=lambda x: x[1], reverse=True))
    bm_results = list(bm_results.keys())
    bm_results = set(bm_results[:(bm_t * int(len(standards)))])
    bm_results = bm_results.intersection(walk_dataset_embed)
    N = float(len(bm_results))
    correct_dataset = standards.intersection(bm_results)
    T = float(len(correct_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w2.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w2.flush()
    ci_embed_results = {}
    seed_ci_embed = ciatation_embed[seed]
    for ci_cand in bm_results:
        ci_embed_results[ci_cand] = cosine_np(seed_ci_embed, ciatation_embed[ci_cand])
    ci_embed_results = dict(sorted(ci_embed_results.items(), key=lambda x: x[1], reverse=True))
    ci_embed_results = list(ci_embed_results.keys())
    ci_embed_results = set(ci_embed_results[:int(citation_percent * len(ci_embed_results))])
    ci_embed_results = ci_embed_results.intersection(walk_dataset_embed)
    N = float(len(ci_embed_results))
    correct_dataset = standards.intersection(ci_embed_results)
    T = float(len(correct_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w3.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w3.flush()
    # bert_seed = model.encode(seed_bm[seed])
    bert_seed = seed_bert[seed]
    scores = [cosine_np(bert_seed, bert_cand) for bert_cand in bert_embeds]
    bert_results = {}
    for i in range(len(cand_bm_dict)):
        bert_results[cand_bm_dict[i]] = scores[i]
    bert_results = dict(sorted(bert_results.items(), key=lambda x: x[1], reverse=True))
    bert_results = list(bert_results.keys())
    bert_results = set(bert_results[:int(bert_percent * len(bert_results))])
    bert_results = bert_results.intersection(walk_dataset_embed)
    N = float(len(bert_results))
    correct_dataset = standards.intersection(bert_results)
    T = float(len(correct_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w4.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w4.flush()
    ensembel2_results = bm_results.intersection(bert_results)
    N = float(len(ensembel2_results))
    correct_dataset = standards.intersection(ensembel2_results)
    T = float(len(correct_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w7.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w7.flush()
    ensembel3_results = bert_results.intersection(ci_embed_results)
    N = float(len(ensembel3_results))
    correct_dataset = standards.intersection(ensembel3_results)
    T = float(len(correct_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w8.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w8.flush()
    ensembel1_results = bm_results.intersection(ci_embed_results)
    N = float(len(ensembel1_results))
    correct_dataset = standards.intersection(ensembel1_results)
    T = float(len(correct_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w5.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w5.flush()
    ensembel1_results = ensembel1_results.intersection(bert_results)
    N = float(len(ensembel1_results))
    correct_dataset = standards.intersection(ensembel1_results)
    T = float(len(correct_dataset))
    precision = 0.0
    if N != 0.0:
        precision = T / N
    recall = 0.0
    if G != 0.0:
        recall = T / G
    result_w6.write(
        seed + "\t" + str(T) + "\t" + str(G) + "\t" + str(N) + "\t" + str(recall) + "\t" + str(
            precision) + "\r\n")
    result_w6.flush()


if __name__ == '__main__':
    seed_percent = 1.0
    if 'random' in args:
        seed_percent = float(args['random'])
    data_dir = os.path.abspath(args['data'])
    out_dir = os.path.abspath(args['out'])
    seed_path = data_dir + "/seeds.txt"
    cand_path = data_dir + "/cands.txt"
    standard_path = data_dir + "/StandardSchLink.hdt"
    vec_path = data_dir + "/mag_authors_2020_ComplEx_entity.npy"
    vec_dict_path = data_dir + "/authors_entities.dict"
    citation_vec_path = data_dir + "/Citation_vectors.txt"
    paper_authors = data_dir + "/PaperAuthorAffiliations.hdt"
    title_path = data_dir + "/Paper.hdt"
    des_path = data_dir + "/PaperAbs.hdt"
    print("prework")
    print(paper_authors)
    print(os.path.exists(paper_authors))
    coauthor_hdt = HDTDocument(paper_authors)
    print("nb triples: %i" % coauthor_hdt.total_triples)
    standard_hdt = HDTDocument(standard_path)
    print("nb triples: %i" % standard_hdt.total_triples)
    out_dir = data_dir+'/results'
    seeds = read_seed_dataset(seed_path)
    candidates = read_candidate_dataset(cand_path)
    if seed_percent < 1.0:
        seeds = list(seeds)
        random.shuffle(seeds)
        seeds = set(seeds[:int(seed_percent * float(len(seeds)))])
        candidates = reduce_candidates(seeds, candidates, standard_hdt)
    print("Experiment with " + str(len(seeds)) + " seeds and " + str(
        len(candidates)) + " candidates")
    ciatation_embed = get_ciataion_embed_dict(seeds.union(candidates), citation_vec_path)
    for c in candidates:
        try:
            a = ciatation_embed[c]
        except Exception as e:
            print(e)
            print('ERROR', c)
    title_hdt = HDTDocument(title_path)
    des_hdt = HDTDocument(des_path)
    fp = np.memmap(vec_path, mode='r', dtype='float32', shape=(333085368, 100))
    vec_dict = read_vec_dict(vec_dict_path)
    all_seed_author = get_authors_based_on_datasets(seeds, coauthor_hdt)
    candidate_author = get_authors_based_on_datasets(candidates, coauthor_hdt)
    all_author = all_seed_author.union(candidate_author)
    Graph = nx.Graph()
    for author in tqdm(all_author, desc="Building hop1-3 coauthor graph"):
        loop_build_graph(Graph, 0, author, coauthor_hdt)
    all_author = authors_to_embed_dict(all_author, vec_dict, fp)
    seed_bm = get_datasets_title_desc(seeds, title_hdt, des_hdt)
    cand_bm_dict, cand_bm_titles = get_datasets_title_desc_with_dict(candidates, title_hdt, des_hdt)
    bm25_index = BM25Okapi([preprocess_string(title) for title in tqdm(cand_bm_titles, desc="Building bm25 index")])
    seed_bert = {str(seed): model.encode(seed_bm[seed]) for seed in seeds}
    bert_embeds = model.encode(cand_bm_titles, show_progress_bar=True)
    hop_list = []
    threshold_list = []
    if 'all' in args:
        hop_list = [1, 2, 3]
        threshold_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    if 'threshold' in args:
        if float(args['threshold']) not in threshold_list:
            threshold_list.append(float(args['threshold']))
    if 'hop' in args:
        if int(args['hop']) not in hop_list:
            hop_list.append(int(args['hop']))
    for i in hop_list:
        hop = str(i)
        for threshold in threshold_list:
            bm_t = int(args['bm25_threshold'])
            citation_percent = 0.8
            bert_percent = 0.8
            result_w = open(out_dir + "/hop" + hop + "_01_pure.tsv", 'w')
            result_w1 = open(
                out_dir + "/hop" + hop + "_02_embed_" + str(threshold) + ".tsv",
                'a+')
            result_w2 = open(
                out_dir + "/hop" + hop + "_04_embed_" + str(
                    threshold) + "_bm25.tsv",
                'a+')
            result_w3 = open(
                out_dir + "/hop" + hop + "_03_embed_" + str(
                    threshold) + "_citation.tsv",
                'a+')
            result_w4 = open(
                out_dir + "/hop" + hop + "_05_embed_" + str(
                    threshold) + "_bert.tsv",
                'a+')
            result_w5 = open(
                out_dir + "/hop" + hop + "_06_embed_" + str(
                    threshold) + "_bm25_citation.tsv", 'a+')
            result_w6 = open(
                out_dir + "/hop" + hop + "_09_embed_" + str(
                    threshold) + "_bm25_citation_bert.tsv", 'a+')
            result_w7 = open(
                out_dir + "/hop" + hop + "_07_embed_" + str(
                    threshold) + "_bm25_bert.tsv", 'a+')
            result_w8 = open(
                out_dir + "/hop" + hop + "_08_embed_" + str(
                    threshold) + "_citation_bert.tsv", 'a+')
            for seed in tqdm(seeds, desc="hop" + hop + " with threshold " + str(threshold)):
                step(i, seed, standard_hdt, coauthor_hdt, Graph, candidate_author, all_author, threshold, result_w,
                     result_w1, result_w2, result_w3, result_w4, seed_bm, bm25_index, cand_bm_dict, ciatation_embed,
                     seed_bert, result_w5, result_w6, result_w7, result_w8, bert_percent, citation_percent, bm_t)
