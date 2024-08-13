import os
import pathlib
import argparse
import json
import multiprocessing
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch
import csv
from rdflib import Graph

import re
import pandas as pd
import pykeen
import numpy as np
all_args = argparse.ArgumentParser()
all_args.add_argument("-data", "--data", required=False, type=pathlib.Path,
                      help="Path to directory contains all needed data")
args = vars(all_args.parse_args())


from pykeen.models import ComplEx


def convert_to_csv():


    # Load the NTriples data
    g = Graph()
    g.parse('./PaperAuthorAffiliations.nt', format='nt')

    # Create a TSV file to write the data
    tsv_file = './AuthorPubmed.tsv'

    # Open the TSV file for writing
    with open(tsv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Iterate through the NTriples data and write it to the TSV file
        for s, p, o in g:
            subject = str(s)
            predicate = str(p)
            obj = str(o)
            csvwriter.writerow([subject, predicate, obj])

    print(f'NTriples data has been converted to csv format and saved as {tsv_file}.')


def create_split():
    input_csv_file = './AuthorPubmed.csv'

    # Replace 'output.tsv' with the desired path for your output TSV file
    output_tsv_file = './authors_pub_train.tsv'
    output_tsv_file_val = './authors_pub_val.tsv'
    output_tsv_file_test = './authors_pub_test.tsv'

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv_file)
    total_rows = len(df)
    train_size = int(0.8 * total_rows)
    valid_size = int(0.1 * total_rows)

    # Split the DataFrame into training, validation, and test sets
    train_df = df.iloc[:train_size, :]
    valid_df = df.iloc[train_size:(train_size + valid_size), :]
    test_df = df.iloc[(train_size + valid_size):, :]
    # Write the DataFrame to a TSV file
    train_df.to_csv(output_tsv_file, sep='\t', index=False)
    valid_df.to_csv(output_tsv_file_val, sep='\t', index=False)
    test_df.to_csv(output_tsv_file_test, sep='\t', index=False)




def create_ComplEX_embeddingd_pubmed():
    # triples_kg = TriplesFactory.from_path('./pykeen/PaperAuthorAffiliations.nt')
    # Step 3: Train a ComplEx model

    # training = TriplesFactory.from_path('dataset_pykeen/training_data.tsv')
    training = TriplesFactory.from_path('AuthorPubmed.tsv')
    validation = TriplesFactory.from_path('authors_pub_val.tsv')
    test = TriplesFactory.from_path('authors_pub_test.tsv')

    f = open('PaperAuthorAffiliations.nt','r')
    print(len(f.readlines()))
    ff = open('AuthorsNames.nt','r')
    print(len(ff.readlines()))
    fff = open('authors_pub_train.tsv','r')
    print(len(fff.readlines()))

    # training_factory, testing_factory, validation_factory = factory.split(ratios)
    if os.listdir('./models_0') == []:
        result = pipeline(
            model="ComplEx",
            training=training,
            validation=validation,
            testing=test,
            device='cuda',
            model_kwargs=dict(embedding_dim=50),  # Set the embedding dimension here
            training_kwargs=dict(num_epochs=5),
        )
        result.save_to_directory('models_0')

        model = result.model
    else:
        model = torch.load('./models_0/trained_model.pkl')

    #vecs = model.entity_representations[0]

    # entity_embedding_tensor = model.entity_representations[0](indices=None).detach().numpy()
    #entity_embedding_tensor = model.entity_representations[0](indices=None).detach().numpy()
    # Specify the file path where you want to save the ndarray
    vecs = model.entity_representations[0]._embeddings.weight.cpu().detach().numpy()
    print(len(vecs))
    file_path = 'author_embeddings_pubmed.npy'

    # Save the ndarray to the specified file
    np.save(file_path, vecs)


def tsv_to_dict(file_path, delimiter='\t'):

    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        for row in reader:
            data.append(row)
    return data

def save_dict_to_json(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)


if __name__ == '__main__':
    create_ComplEX_embeddingd_pubmed()



