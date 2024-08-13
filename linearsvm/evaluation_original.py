"""
In the evaluation module the functions for evaluation parameters and the confusion matizes
are defined.
"""

import seaborn as sn
from sklearn import metrics
from sklearn.metrics import recall_score,precision_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import numpy as np

def precision_recall_ndcg_map_mrr_at_k(y_true, y_pred, k=None):
    # Sort predictions and get indices of the top-k predictions
    if k:
        top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
    else:
        top_k_indices = np.argsort(y_pred, axis=1)[:, :]
    # Initialize variables to store precision, recall, nDCG, mAP, and MRR values
    precision_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    map_sum = 0
    mrr_sum = 0

    for i in range(len(y_true)):
        true_labels = np.where(y_true[i] == 1)[0]
        predicted_labels = top_k_indices[i]

        # Calculate precision and recall for the current instance
        common_labels = np.intersect1d(true_labels, predicted_labels)
        precision = len(common_labels) / k if k > 0 else 0
        recall = len(common_labels) / len(true_labels) if len(true_labels) > 0 else 0

        # Accumulate precision and recall values
        precision_sum += precision
        recall_sum += recall

        # Calculate nDCG for the current instance
        dcg = sum(1 / np.log2(rank + 2) for rank in range(k) if predicted_labels[rank] in true_labels)
        idcg = sum(1 / np.log2(rank + 2) for rank in range(len(true_labels)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_sum += ndcg

        # Calculate Average Precision (AP) for the current instance
        num_relevant_items = len(true_labels)
        if num_relevant_items > 0:
            precision_at_i = [len(np.intersect1d(true_labels, predicted_labels[:j+1])) / (j + 1) for j in range(k)]
            average_precision = sum(precision_at_i) / num_relevant_items
            map_sum += average_precision

        # Calculate Reciprocal Rank (RR) for the current instance
        rr = 0 if len(common_labels) == 0 else 1 / (np.where(predicted_labels == common_labels[0])[0][0] + 1)
        mrr_sum += rr

    # Calculate average precision, recall, nDCG, mAP, and MRR
    avg_precision = precision_sum / len(y_true)
    avg_recall = recall_sum / len(y_true)
    avg_ndcg = ndcg_sum / len(y_true)
    avg_map = map_sum / len(y_true)
    avg_mrr = mrr_sum / len(y_true)

    return avg_precision, avg_recall, avg_ndcg, avg_map, avg_mrr





def multi_confusion_matrix(y_true, y_pred, classes, multilabelencoder, name):
    """
    Computes confusion matrix for multi-label data (one-vs-rest for each class) and displays all
    confusion matrizes as heatmaps in a figure
    input: list actual and list of predicted y values and list of classnames,
    output: confusion matrizes and figure of confusion matrizes as heatmaps
    """
    if multilabelencoder:
        confusion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
    else:
        confusion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred, labels=classes)
    pdf_name = "confusion_matrix" + name + ".pdf"
    with PdfPages(pdf_name) as pdf_page:
        for i in range(0, len(confusion_matrix)):
            dataframe = pd.DataFrame(confusion_matrix[i])
            figure = plt.figure()
            confusion_matrix_plot = sn.heatmap(dataframe, annot=True, cmap="Spectral",
                                               xticklabels=['Predicted negative', 'Predicted positive'],
                                               yticklabels=['Actually negative', 'Actually positive'],
                                               cbar=False)
            confusion_matrix_plot.set_title(classes[i])
            pdf_page.savefig(figure)
            plt.close(figure)
    return confusion_matrix

def single_confusion_matrix(y_true, y_pred, classes):
    """
    Computes and a confusion matrix for single-label data and displays it as heatmap.
    input: list of actual and list of predicted y values and list of classnames,
    output: confusion matrix
    """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    df_confusion_matrix = pd.DataFrame(confusion_matrix)
    df_confusion_matrix.index.name = 'Actual'
    df_confusion_matrix.columns.name = 'Predicted'
    sn.heatmap(df_confusion_matrix, annot=True, cmap="Spectral", xticklabels=classes,
               yticklabels=classes)
    plt.show()
    return df_confusion_matrix

def eval_metrics(trans_y_true, trans_y_pred, name):
    """
    Prints classification report and computes different kinds of evaluation metrics.
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores
    """
    report = metrics.classification_report(trans_y_true, trans_y_pred)
    evaluation_scores = "{} Classification report: \n{}\n".format(name, report)
    return evaluation_scores

def eval_metrics_all(trans_y_true, trans_y_pred, name):
    """
    Prints classification report and computes different kinds of evaluation metrics.
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores
    """



    precision1, recall1, ndcg1, map_value, mrr = precision_recall_ndcg_map_mrr_at_k(trans_y_true, trans_y_pred, 1)
    precision5, recall5, ndcg5, map_value, mrr = precision_recall_ndcg_map_mrr_at_k(trans_y_true, trans_y_pred, 5)
    precision10, recall10, ndcg10, map_value, mrr = precision_recall_ndcg_map_mrr_at_k(trans_y_true, trans_y_pred, 10)
    print(trans_y_true.shape)
    print(trans_y_pred.shape)
    json_eval = {}
    json_eval['nDCG_1'] = ndcg1
    json_eval['nDCG_5'] = ndcg5
    json_eval['nDCG_10'] = ndcg10
    # json_eval['nDCG'] = ndcg
    json_eval['P_1'] = precision1
    json_eval['P_5'] = precision5
    json_eval['P_10'] = precision10
    # json_eval['P'] = precision
    json_eval['R_1'] = recall1
    json_eval['R_5'] = recall5
    json_eval['R_10'] = recall10
    # json_eval['R'] = recall10
    # json_eval['R'] = recall
    json_eval['MRR'] = mrr
    json_eval['MAP'] = map_value
    json_eval['P_macro'] = precision_score(trans_y_true,trans_y_pred,average='macro')
    json_eval['R_macro'] = recall_score(trans_y_true,trans_y_pred,average='macro')


    return json_eval

def multilabel_evaluation(y_true, y_pred, name):
    """
    Performes evaluation on multi-label classification: compute evaluation metrics and confusion
    matrix
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores, confusion matrix and figure of confusion matrix as heatmap
    """
    print(name, " evaluation:")
    datasets = []
    for dataset_label in y_true:
        datasets.append(dataset_label)
    for dataset_label in y_pred:
        datasets.append(dataset_label)
    label_encoder = LabelEncoder()
    label_encoder.fit(datasets)
    classnames = list(label_encoder.classes_)
    trans_y_true = label_encoder.transform(y_true)
    trans_y_pred = label_encoder.transform(y_pred)
    evaluation_scores = eval_metrics(trans_y_true, trans_y_pred, name)
    confusion_matrix = multi_confusion_matrix(y_true, y_pred, classnames, False, name)
    return evaluation_scores, confusion_matrix

def multilabel_evaluation_multilabelbinarizer(y_true, y_pred, name):
    """
    Performes evaluation on multi-label classification: compute evaluation metrics and confusion
    matrix.
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores, confusion matrix and figure of confusion matrix as heatmap
    """
    print(name, " evaluation:")
    datasets = []
    for dataset_label in y_true:
        datasets.append(dataset_label)
    for dataset_label in y_pred:
        datasets.append(dataset_label)
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(datasets)
    classnames = list(mlb.classes_)
    trans_y_true = mlb.transform(y_true)
    trans_y_pred = mlb.transform(y_pred)
    print(f'trans_y_true {trans_y_true}')
    print(f'trans_y_pred {trans_y_pred}')
    print(f'trans_y_true {len(trans_y_true)}')
    print(f'trans_y_pred {len(trans_y_pred)}')
    print(f'trans_y_true {len(trans_y_true[0])}')
    print(f'trans_y_pred {len(trans_y_pred[0])}')
    print(f'trans_y_true {len(trans_y_true[1])}')
    print(f'trans_y_pred {len(trans_y_pred[1])}')
    evaluation_scores = eval_metrics(trans_y_true, trans_y_pred, name)
    # evaluation_scores_all = eval_metrics_all(trans_y_true, trans_y_pred, name)
    confusion_matrix = multi_confusion_matrix(trans_y_true, trans_y_pred, classnames, True, name)
    return  evaluation_scores, confusion_matrix


def singlelabel_evaluation(y_true, y_pred, name):
    """
    Performes evaluation on single-label classification: compute evaluation metrics and confusion
    matrix.
    input: list of actual and list of predicted y values and name of the used classifier,
    output: list of evaluation scores, confusion matrix and figure of confusion matrix as heatmap
    """
    print(name, " evaluation:")
    datasets = []
    for dataset_label in y_true:
        datasets.append(dataset_label)
    for dataset_label in y_pred:
        datasets.append(dataset_label)
    label_encoder = LabelEncoder()
    label_encoder.fit(datasets)
    classnames = list(label_encoder.classes_)
    trans_y_true = label_encoder.transform(y_true)
    trans_y_pred = label_encoder.transform(y_pred)
    evaluation_scores = eval_metrics(trans_y_true, trans_y_pred, name)
    confusion_matrix = single_confusion_matrix(y_true, y_pred, classnames)
    return evaluation_scores, confusion_matrix