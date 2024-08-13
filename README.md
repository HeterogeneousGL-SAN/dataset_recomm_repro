# Code and Data for the Reproducibility and Analysis of Scientific Dataset Recommendation Methods

This repository contains the code and the data for the paper: _Reproducibility and Analysis of Scientific Dataset Recommendation Methods_ submitted at RecSys24. 

We reproduced four dataset recommendation methods (references at the end of the repository):

- **LineraSVM**. This method treats the problem of dataset recommendation as a multiclass multilabel problem. The original paper considers two types of queries: abstracts and citation contexts: in this paper we consider only abstracts as queries for reproducibility purposes.
- **bi-encoder retriever**. This method recommend datasets basing on a textual query or a set of keywords describing the user's information need.
- **Ensemble_CN method**. This method considers an ensemble of the following methods: citation network embeddings, co-authors paths, co-authors embeddings, BM25 and BERT. 
- **HVGAE**. This method relies on a graph variational autoencoder to provide the recommendation.

## Preliminaries
To run this code, the recommended python version is 3.8. 
To install all the dependencies, create a new environment and, from the main folder of the project type:

```pip install -r requirements.txt```


To reporduce each method described in the paper, follow the instructions below each method in this readme.

#### Project structure
The repository is organized as follows:
- ```BASE``` contains the baselines for: lightgcn, bpr and toppop, cosine-based ranking and bm25
- ```biencoder``` contains the code used to preprocess the data and prepare them to make them adapt to the biencoder original implementation of the original paper and repository
- ```linearsvm``` contains the code used to preprocess the data and prepare them to make them adapt to the linearsvm original implementation of the original paper and repository
- ```hvgae``` folder contains our implementation of hvgae heterogeneous variational graph autoencoder.
- ```data``` contains the three datasets used in each method.

#### Data 
MES, PubMed_KCore2 and PubMed are three datasets originally available in CSV format. If you want to re-run the code in this repository with your data, please, refer to the following format.

- ```data/all``` contains two zip files: one for authors and other one for all the other data. These data are used for: the hvgae, the base recommendations and the ensemble_cn
- ```data/biencoder_data``` contains the data already processed for the biencoder
- ```data/linearsvm_data``` this folder will be automatically created once that the preprocessing for linearsvm method is run.
- ```data/ensemble_cn``` contains the seeds and cands data used to reproduce the ensemble_cn method


## Methods
### LinearSVM
To run LinearSVM, the reference repository is the original one available at: [https://github.com/michaelfaerber/datarec](https://github.com/michaelfaerber/datarec). 
To replicate the code with the original code and data, just follow the instructions on the original repository.

To run the LinearSVM implementation with MES, PubMed_KCore2 and PubMed, preprocess the datasets:

```python linearsvm/linearsvm_preprocess.py --path_dataset path/to/data/folder```

At the end of the procedure, inside the ```path_dataset``` folder there will be `linearsvm_data` folder there are all the files needed to run the linearSVM, namely a corpus, a datasets list and the training/test splits.

Once pre-processed the files, it is possible to run the code evaluating it on one of the three datasets. The ```path_dataset``` denotes the path to one of the three datasets, for  example: ```./data/mes/```.

```python methods/linearsvm/linearsvm.py --path_dataset=path/to/data/folder```



### Bi-Encoder Retriever
To run the bi-encooder retriever, the reference repository is the original one available at: [https://github.com/viswavi/datafinder/tree/main](https://github.com/viswavi/datafinder/tree/main). To replicate the original experiment, follow the instructions provided in the original repository of the method.

To run the bi-enocder implementation with MES, PubMed_KCore2 and PubMed, preprocess the datasets:

```python preprocessing/biencoder_preprocess.py --path_dataset=path/to/data/folder```

The ```path_dataset``` denotes the path to one of the three datasets, for  example: ```./data/mes/```.

At the end of the procedure, inside the ```path_dataset``` folder there will be a `biencoder_data` folder there are all the files needed to run the linearSVM, namely the qrels, the dataset search collection, the training and test data.

Then, once these files are collected, run the code available in the original repository with these files. Nothing should be changed, as the files' names are the same as in the original implementation. 

### Ensemble_CN
To run the Ensemble_CN method, the reference repository is the original one available at: [https://github.com/viswavi/datafinder/tree/main](https://github.com/viswavi/datafinder/tree/main). To replicate the original experiment, follow the instructions provided in the original repository of the method.
To provide an easy and fast way to run the original code with the original data (replicability) or the three shared datasets (MES, PubMed_KCore2, PubMed), we provide the Dockerfile that will build the image and install all the dependencies at one. The docker implementation is provided in: ```preprocessing/ensemble```. This implementation can be ran with the original method in ```Recommendation.py``` file, taken from the original repository and added here for clarity and ease of reproduction.

To reproduce this method with PubMed, PubMed_KCore2, MES or your custom data:
- Run preprocessing: ```python ensemble_cn/preprocessing/ensemble_cn.py --path_dataset=/path/to/your/data/folder```
- build the ComplEx embeddings: ```python ensemble_cn/preprocessing/ensemble_cn_complex_embeddings.py```
- build the citation network embeddings following the instructions on [KGlove](https://github.com/miselico/KGlove) and [Glove](https://github.com/stanfordnlp/GloVe) repositories
- All the files in .nt format must be converted in hdt. To this aim, we used the [hdt-cpp](https://github.com/rdfhdt/hdt-cpp/blob/develop/README.md) library 

Build the docker image in ```python methods/ensemble```
```docker build -t datarecommendimage .```

Run:
```docker run --rm -ti --name datarecommendimage0 -v /path/to/data/folder/data/dataset:/code/data datarecommendimage:latest python3 Recommendation.py -data=/code/data -out=/code/results```

The results of the implementation will be placed in the results folder.
In order to compute the precision, recall and F! scores, we provide the script which was not provided in the original implementation. The code is in: ```methods/ensemble/compute_results.py```.

### HVGAE
To run HVGAE original:

```python hvgae/main.py --path=path/to/data/folder```

To run HVGAE reproducing the component analysis on the citation network:

```python hvgae/main.py --path=path/to/data/folder```

To run HVGAE reproducing the component analysis on the bipartite network:

```python hvgae/main_bip.py --path=path/to/data/folder```

These files already run on the standard, reduced, enriched setups.




### Base Recommendation
To run Base recommendation (lightgcn, bpr and toppop are computed in the same moment; bm25 and cosine are in two separated files):

```python BASE/lightgcn_bpr_toppop.py --path=path/to/data/folder```
```python BASE/bm25.py --path=path/to/data/folder```
```python BASE/cosine.py --path=path/to/data/folder```



## References
__LinearSVM__

Färber, M., & Leisinger, A. K. (2021, October). Recommending datasets for scientific problem descriptions. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 3014-3018).

__bi-encoder retriever__

Viswanathan, V. & Gao, L. & Wu T. , Liu, P. , and Neubig, G.. 2023. DataFinder: Scientific Dataset Recommendation from Natural Language Descriptions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10288–10303, Toronto, Canada. Association for Computational Linguistics.

__Ensemble_CN__

Wang, X., van Harmelen, F., Cochez, M., & Huang, Z. (2022, July). Scientific item recommendation using a citation network. In International Conference on Knowledge Science, Engineering and Management (pp. 469-484). Cham: Springer International Publishing.

__HVGAE__

Altaf, B., Akujuobi, U., Yu, L., & Zhang, X. (2019, November). Dataset recommendation via variational graph autoencoder. In 2019 IEEE International Conference on Data Mining (ICDM) (pp. 11-20). IEEE.










