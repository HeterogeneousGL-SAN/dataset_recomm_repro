import argparse
import cornac
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit, BaseMethod
from cornac.models import BPR
import pandas as pd
from cornac.metrics import Precision, Recall, NDCG
from cornac.data import Reader
from cornac.utils import cache
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import GridSearch, RandomSearch

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", default='mes',choices=['mes','pubmed','pubmed_kcore'],
                    type=str)
parser.add_argument("-seed",default=42,type=int)
parser.add_argument("-path",type=str, help='path to data folder',default='./data/mes')
args = parser.parse_args()

path_to_data = args.path
traindf = pd.read_csv(f'{path_to_data}/pubdataedges_kcore_train.csv')
train_data = []
print(traindf.shape[0])

for i,row in traindf.iterrows():
    train_data.append((row['source'],row['target'],1))
valdf = pd.read_csv(f'{path_to_data}/pubdataedges_kcore_validation.csv')
vlidation_data = []
print(valdf.shape[0])
for i,row in valdf.iterrows():
    vlidation_data.append((row['source'],row['target'],1))
print(len(vlidation_data))

testdf = pd.read_csv(f'{path_to_data}/pubdataedges_kcore_test.csv')
test_data = []
print(testdf.shape[0])
for i,row in testdf.iterrows():
    test_data.append((row['source'],row['target'],1))
print(len(test_data))

rs = BaseMethod.from_splits(
    train_data=train_data, val_data=vlidation_data, test_data=test_data, exclude_unknowns=False, verbose=True
)
# Instantiate the LightGCN model
rec10 = cornac.metrics.Recall(10)
metrics = [Precision(k=5),Recall(k=5),NDCG(k=5)]


bpr = BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=args.seed)


lightgcn = cornac.models.LightGCN(
    seed=args.seed,
    num_epochs=500,
    num_layers=3,
    early_stopping={"min_delta": 1e-4, "patience": 50},
    batch_size=4096,
    learning_rate=0.01,
    lambda_reg=1e-3,
    verbose=True
)


most_pop = cornac.models.MostPop()


for name in [most_pop,bpr,lightgcn]:
    test_result, _ = rs.evaluate(
        model=name, metrics=metrics, user_based=False
    )

    print(test_result)