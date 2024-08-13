"""
MODIFIED VERSION TAKEN FROM DATAREC REPO: https://github.com/michaelfaerber/datarec

"""
import sys
import argparse

from sklearn import decomposition
import re

import pickle
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import sys

import time
import json
import preprocessing_original
import evaluation_original
import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mes',
                    choices=['Cora', 'CiteSeer','mes','pubmed_kcore','pubmed'])

parser.add_argument('--path_dataset', type=str)

args = parser.parse_args()
path_dataset = args.path_dataset

def parallel_saving(i, line, titles):
    query = str(line).split("\t")[1].replace("\n", "").strip()
    id = str(line).split("\t")[0]
    for title in titles:
        query = query.replace(title, "")
    dataset = str(str(line).split("\t")[2]).replace("\n", "").strip()
    final_dataset = dataset.split(", ")
    preprocessed_query = preprocessing_original.preprocess(query)
    preprocessed_tuple = (final_dataset, preprocessed_query, id)
    preprocessed_tuple_single = (dataset, preprocessed_query, id)
    preprocessed_data_list.append(preprocessed_tuple)
    preprocessed_data_list_single.append(preprocessed_tuple_single)
    return preprocessed_data_list, preprocessed_data_list_single


def preprocess_string(input_string):
    cleaned_string = re.sub(r'[\n\t\s]', ' ', input_string)
    cleaned_string = re.sub(r'[^A-Za-z0-9 ]', '', cleaned_string)
    cleaned_string = ' '.join(cleaned_string.split()).lower()

    return cleaned_string


# open sample data for abstracts or citations, for citations encoding 'ISO-8859-1' needs to be specified

dataframe = open(path_dataset+"/linearsvm_data/Abstract_New_Database.txt")


documentation_file_all_metrics = open(path_dataset+"/linearsvm_results/metrics_tfidf.json", "w+")
documentation_file_parameteropt = open(path_dataset+"/linearsvm_results/parameter_optimization_tfidf_mes_finale2.txt", "w+")
documentation_file_modelopt = open(path_dataset+"/linearsvm_results/classifier_optimization_tfidf.txt", "w+")

titles = []
with open(path_dataset+"/linearsvm_data/Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

# create list with preprocessed text and corresponding datasets for traditionale machine learning
# models (preprocessed_data_list) and for neural networks (preprocessed_data_list_single)
preprocessed_data_list = []
preprocessed_data_list_single = []

i = 0

if os.path.exists(path_dataset+"/linearsvm_data/tmp_data.json"):
    f = open(path_dataset+"/linearsvm_data/tmp_data.json", 'r')
    data = json.load(f)
    datasets = data['datasets']
    queries = data['queries']
    datasets_single = data['datasets_single']
    queries_single = data['queries_single']
    query_ids = data['queries_id']

    print(len(datasets))
    print(len(queries))
    print(len(datasets_single))
    print(len(queries_single))
    print(len(query_ids))

else:
    for i, line in enumerate(dataframe):
        if i % 100 == 0:
            print(i)
        id = str(line).split("\t")[0]
        query = str(line).split("\t")[1].replace("\n", "").strip()
        # for title in titles:
        #     title_fil = preprocess_string(title)
        #     query_fil = preprocess_string(query)
        #     query = query_fil.replace(title_fil, "")
        dataset = str(str(line).split("\t")[2]).replace("\n", "").strip()
        final_dataset = dataset.split(", ")
        preprocessed_query = preprocessing_original.preprocess(query)
        preprocessed_tuple = (final_dataset, preprocessed_query, id)
        preprocessed_tuple_single = (dataset, preprocessed_query, id)
        preprocessed_data_list.append(preprocessed_tuple)
        preprocessed_data_list_single.append(preprocessed_tuple_single)
        i += 1
    datasets, queries, query_ids = zip(*preprocessed_data_list)
    datasets_single, queries_single, query_ids_single = zip(*preprocessed_data_list_single)
    g = open(path_dataset+"/linearsvm_data/tmp_data.json", 'w')
    json.dump({'queries_id': query_ids, 'datasets': datasets, 'datasets_single': datasets_single, 'queries': queries,
               'queries_single': queries_single}, g)

q_tfidf = preprocessing_original.tfidf(queries)
documentation_file_parameteropt.write("tfidf Evaluation \n")
documentation_file_modelopt.write("tfidf Evaluation \n")
print("  Actual number of tfidf features: %d" % q_tfidf.get_shape()[1])
print('write split')

f = open(path_dataset+"/linearsvm_data/split.json", 'r')
json_data = json.load(f)
d_train = json_data['training']['datasets']
d_test = json_data['test']['datasets']
print('d_train',len(d_train))
print('d_test',len(d_test))
d_train_indexes = [datasets.index(d) for d in d_train]
d_test_indexes = [datasets.index(d) for d in d_test]
print('d_train',len(d_train_indexes))
print('d_test',len(d_test_indexes))

selected_row_indices = d_train_indexes

q_train_ids = json_data['training']['query_ids']
q_test_ids = json_data['test']['query_ids']
q_train_indexes = [query_ids.index(q) for q in q_train_ids]
q_test_indexes = [query_ids.index(q) for q in q_test_ids]


q_train = q_tfidf[q_train_indexes]  # Contains the selected rows
q_test = q_tfidf[q_test_indexes]  # Contains all other rows
print('q_test', q_test.shape)
print('q_train', q_train.shape)
print('q_tfidf', q_tfidf.shape)

label_encoder = MultiLabelBinarizer()
label_encoder.fit(datasets)
d_train_encoded = label_encoder.transform(d_train)
pickle.dump(label_encoder, open('label_encoder_tfidf.sav', 'wb'))

start = time.time()
# Linear SVM: optimizing parameters with grid search
print("SVM model evaluation")
svm_dict = dict(estimator__C=[5, 10, 20, 50, 100])
classifier_svm = RandomizedSearchCV(estimator=OneVsRestClassifier(LinearSVC()),
                                    param_distributions=svm_dict,
                                    n_iter=5, n_jobs=-1)
classifier_svm.fit(q_train, d_train_encoded)
documentation_file_parameteropt.write(
    "Linear SVM: Best parameters {}, reached score: {} \n".format(
        classifier_svm.best_params_, classifier_svm.best_score_))
svm_model = classifier_svm.best_estimator_
pickle.dump(svm_model, open("svm_tfidf.sav", 'wb'))
pred_svm = svm_model.predict(q_test)

y_pred = label_encoder.inverse_transform(pred_svm)

svm_evaluation_scores, svm_evaluation_scores_all, svm_cm = evaluation_original.multilabel_evaluation_multilabelbinarizer(
    d_test, y_pred, "LinearSVM")
json.dump(svm_evaluation_scores_all, documentation_file_all_metrics)
documentation_file_modelopt.write(svm_evaluation_scores)
documentation_file_parameteropt.close()
documentation_file_modelopt.close()
end = time.time()
print('finished in ', end - start)
