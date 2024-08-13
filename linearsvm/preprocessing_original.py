"""
In the preprocessing module functions for language preprocessing, transformation of textual to
numeric representation and sampling methods for imbalenced data are defined.
"""
import json
import string
from collections import namedtuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec
from transformers import BertTokenizer, BertModel
from transformers import TransfoXLTokenizerFast, TransfoXLModel
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import torch
import fasttext
import pandas as pd


# --- datafinder
def create_titles():
    g = open('../data/datafinder/Dataset_Titles.txt', 'w')
    f = open('../data/datafinder/dataset_search_collection.jsonl', 'r')
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        title = data['title'] +'\n'
        g.write(title)
    f.close()
    g.close()


def create_dataframe():

    """This method unifies train and test data"""

    g = open('../data/datafinder/Abstract_New_Database_1.txt', 'w')
    # train
    train_file = open('../data/datafinder/train_data.jsonl', 'r')
    lines = train_file.readlines()
    for line in lines:
        data = json.loads(line)
        id = data['paper_id']
        abs = data['abstract']
        pos = ', '.join(data['positives'])
        line_to_add = id +'\t'+abs+'\t'+pos+'\n'
        g.write(line_to_add)
    print(len(lines))
    test_file = open('../data/datafinder/test_data.jsonl', 'r')
    lines_test = test_file.readlines()
    for line in lines_test:
        data = json.loads(line)
        id = 'test_'+str(lines_test.index(line))
        abs = data['abstract']
        pos = ', '.join(data['documents'])
        line_to_add = id +'\t'+abs+'\t'+pos+'\n'
        g.write(line_to_add)
    print(len(lines_test))

# create_dataframe()


# --- mes
def convert_csv(csv):
    df = pd.read_csv(csv)
    g = open('../../data/mes/Abstract_New_Database_2.txt','w')
    df = df.dropna()
    for index, row in df.iterrows():
        # print(row)
        print('\n')
        line = row['pid']+'\t'+row['abstract']+'\t'+row['datasets'][1:-1]
        line = line.replace('"','')
        print(line)
        # break
        g.write(line)
        g.write('\n')


def convert_csv_titles(csv):
    df = pd.read_csv(csv)
    g = open('../../data/mes/Dataset_Titles.txt', 'w')
    print(df.shape)
    for index, row in df.iterrows():
        g.write(row['title'][1:-1].replace('\n',''))
        g.write('\n')


# ---
def preprocess(input_string):
    """
    Natural language preprocessing: only lower-case, apostrophes, punctuation and numbers are
    removed, stopwords are erased and single characters are removed, word stemming is applied
    input: input_string,
    output: preprocessed_string
    """
    input_string = input_string.lower()
    input_string = input_string.replace("'", "")
    input_string = input_string.translate(str.maketrans("", "", string.punctuation))
    input_string = input_string.translate(str.maketrans("", "", string.digits))
    tokens = word_tokenize(input_string)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    result_string = ""
    for word in tokens:
        if word not in stop_words:
            if len(word) > 1:
                transformed_word = stemmer.stem(word)
                result_string = result_string + transformed_word + " "
    preprocessed_string = result_string.strip()
    return preprocessed_string

def tfidf(document_collection):
    """
    Computes tfidf for each document in the document_collectionlection that is input
    input: list of documents document_collection,
    output: list of tfidfs scores for documents tf_document_collection
    """
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, max_df=0.8,
                                 min_df=5)
    tf_document_collection = vectorizer.fit_transform(document_collection)
    return tf_document_collection

def tfidf_pubmed(document_collection):
    """
    Computes tfidf for each document in the document_collectionlection that is input
    input: list of documents document_collection,
    output: list of tfidfs scores for documents tf_document_collection
    """
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, max_df=0.8,max_features=5000,
                                 min_df=50)
    tf_document_collection = vectorizer.fit_transform(document_collection)
    return tf_document_collection

def doc2vec(document_collection):
    """
    Computes doc2vec representation of each document handed over
    input: list of documents document_collection,
    output: list of dov2vec vectors for all documents trans_doc2vec
    """
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(document_collection):
        words = text.split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    model = Doc2Vec(docs, dm=0, epochs=10, workers=5)
    trans_doc2vec = []
    for i in range(0, len(document_collection)):
        trans_doc2vec.append(model.docvecs[i])
    return trans_doc2vec

def sci_bert_embeddings(document_collection):
    """
    Computes SciBERT Embedding of each document handed over
    input: list of documents document_collection,
    output: list of SciBERT Embeddings for all documents embeddings
    """
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
                                              do_lower_case=True)
    model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
    model.eval()
    embeddings = []
    for doc in document_collection:
        input_ids = torch.tensor(tokenizer.encode(doc, max_length=512,
                                                  add_special_tokens=True)).unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs[0]
        sentence_embedding = torch.mean(last_hidden_states[0], dim=0)
        embeddings.append(sentence_embedding.detach().numpy())
    return embeddings

def transformersxl_embeddings(document_collection):
    """
    Computes TransformersXL Embedding of each document handed over
    input: list of documents document_collection,
    output: list of TransformersXL Embeddings for all documents embeddings
    """
    tokenizer = TransfoXLTokenizerFast.from_pretrained('transfo-xl-wt103', do_lower_case=True,
                                                       add_space_before_punct_symbol=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
    embeddings = []
    for doc in document_collection:
        inputs = tokenizer.encode(doc, add_special_tokens=True,
                                  return_tensors="pt").clone().detach()
        outputs = model(inputs)
        last_hidden_states = outputs[0]
        sentence_embedding = torch.mean(last_hidden_states[0], dim=0)
        embeddings.append(sentence_embedding.detach().numpy())
    return embeddings

def fast_text_embeddings(document_collection):
    """
    Computes FastText Embedding of each document handed over
    input: list of documents document_collection,
    output: list of FastText Embeddings for all documents embeddings
    """
    file = open("data.txt", "w+")
    for document in document_collection:
        file.write(document + "\n")
    file.close()
    model = fasttext.train_unsupervised("data.txt", model='skipgram')
    embeddings = []
    for document in document_collection:
        embeddings.append(model.get_sentence_vector(document))
    return embeddings

def random_oversampling(x, y):
    """
    Random oversamling.
    input: list of x-values x, list of y-values y,
    output: oversampled lists x and y
    """
    x_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(x, y)
    return x_resampled, y_resampled

def random_undersampling(x, y):
    """
    Random undersamling.
    input: list of x-values x, list of y-values y,
    output: undersampled lists x and y
    """
    x_resampled, y_resampled = RandomUnderSampler(random_state=0).fit_resample(x, y)
    return x_resampled, y_resampled
