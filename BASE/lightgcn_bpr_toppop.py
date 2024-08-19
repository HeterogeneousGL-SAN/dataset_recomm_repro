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
parser.add_argument("-lr",default=0.01,type=float)
parser.add_argument("-reg",default=0.001,type=float)
parser.add_argument("-iters",default=300,type=int)
parser.add_argument("-grid_search",action='store_true')
parser.add_argument("-path",type=str, help='path to data folder')
args = parser.parse_args()
lr = args.lr
reg = args.reg
iters = args.iters
path_to_data = args.path
if path_to_data is None:
    path_to_data = f'./data/all/data/{args.dataset}'
    
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


# bpr = BPR(k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=args.seed)
bpr = BPR(k=10, max_iter=iters, learning_rate=lr, lambda_reg=reg, seed=args.seed)


lightgcn = cornac.models.LightGCN(
    seed=args.seed,
    num_epochs=500,
    # num_epochs=500,
    num_layers=2,
    #num_layers=3,
    early_stopping={"min_delta": 1e-4, "patience": 50},
    batch_size=4096,
    learning_rate=0.01,
    lambda_reg=1e-3,
    verbose=True
)


most_pop = cornac.models.MostPop()

# if i want all of them with default parameters, just add bpr and lightgcn to the list of methods below (for cycle)

if args.grid_search:
    f = open('./BASE/cornac_results.txt', 'w')
    f.write(f'dataset: {args.dataset}\n')
    for iter in [100,200,300]:
        for lr in [0.1,0.001,0.0001]:
            for lambda_reg in [0.1,0.0001,0.001]:
                bpr = BPR(k=10, max_iter=iter, learning_rate=lr, lambda_reg=lambda_reg, seed=args.seed)
                print(f'LR {lr}, ITER {iter}, LAMBDA_REG {lambda_reg}')
                f.write(f'bpr\nLR {lr}, ITER {iter}, LAMBDA_REG {lambda_reg}\n')
                test_result, _ = rs.evaluate(
                    model=bpr, metrics=metrics, user_based=False
                )
                print(test_result)
                f.write(f'bpr: \n{test_result}\n')
                print('\n\n\n')

    for iter in [100,500,300]:
        for lr in [0.01,0.001,0.0001]:
            for layers in [1,2,3]:
                for batch_size in [512,1024,2048,4096]:
                    print(f'LR {lr}, ITER {iter}, LAYERS {layers}, BATCH {batch_size}')
                    f.write(f'lightgcn\nLR {lr}, ITER {iter}, LAYERS {layers}, BATCH {batch_size}\n')

                    lightgcn = cornac.models.LightGCN(
                        seed=args.seed,
                        num_epochs=iter,
                        num_layers=layers,
                        early_stopping={"min_delta": 1e-4, "patience": 50},
                        batch_size=batch_size,
                        learning_rate=lr,
                        lambda_reg=1e-3,
                        verbose=True
                    )
                    print(f'LR {lr}, ITER {iter}, LAMBDA_REG {lambda_reg}')
                    test_result, _ = rs.evaluate(
                        model=lightgcn, metrics=metrics, user_based=False
                    )
                    print(test_result)
                    f.write(f'lightgcn: \n{test_result}\n')
                    print('\n\n\n')
else:
    test_result, _ = rs.evaluate(
        model=bpr, metrics=metrics, user_based=False
    )
    print(test_result)
