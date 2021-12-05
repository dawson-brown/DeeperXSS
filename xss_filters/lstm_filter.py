import tensorflow as tf

## for deep learning and cbow
import keras
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda

from sklearn.model_selection import KFold


# visualize CBOW model structure
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

## helper libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import dataclasses as dc
from sklearn.model_selection import KFold

## for tokens
import pickle as pk
from pickle import dump, load
from data.tokenizer import URLTokens, JSToken 

## LSTM based on DeepXSS by Yong Fang, Yang Li, Liang Liu, Cheng Huang

def load_tokenized_urls(filename: str) -> URLTokens:

    with open(filename, "rb") as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


## Deep Learning Model ------------------------------------------------------------------------------------------------

def get_data(token_contents):

    weights, vocab, inverse_vocab = get_CBOW(token_contents)
    labels = []

    #load benign urls
    tokenized_urls = []
    for i, tokenized_url in enumerate(load_tokenized_urls('data\dmoz_dir.txt__20211203-134415_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 0 for benign
        labels.append(0)

        if i > 500: ##for testing purposes
            break

    #load malicious urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data\dec_xss_urls.txt__20211203-134417_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 1 for malicious
        labels.append(1)

        if i > 500: ##for testing purposes
            break    

    vector_urls = []
    for tokenized_url in tokenized_urls:

        vector_url = []

        for token in tokenized_url.token_list:

            token_no = vocab[dc.asdict(token)[token_contents]]

            vector_url.append(
                weights[token_no-1] ## -1 because weights has no 0 index
            )
        vector_urls.append(vector_url)

    data_labels_zip = list(zip(vector_urls, labels)) ## zip to keep data and labels aligned during shuffle

    random.seed(5318008)
    random.shuffle(data_labels_zip)
    
    vector_urls, labels = zip(*data_labels_zip)
           
    return vector_urls, labels

def get_CBOW(token_contents):
    if token_contents == "type":
        vocab_name = "vocab_type.pickle"
        inv_vocab_name = "inv_vocab_type.pickle"
        model_name = "cbow_model_token_type"
    else:
        vocab_name = "vocab_value.pickle"
        inv_vocab_name = "inv_vocab_value.pickle"
        model_name = "cbow_model_token_value"

    with open(vocab_name, 'rb') as handle:
        vocab = pk.load(handle)
    with open(inv_vocab_name, 'rb') as handle:
        inverse_vocab = pk.load(handle)

    cbow = keras.models.load_model(model_name)
    weights = cbow.get_weights()[0]

    weights = weights[1:]
    print(weights.shape) ## print shape of weights

    print(pd.DataFrame(weights, index=list(inverse_vocab.values())[1:]).head())

    from sklearn.metrics.pairwise import euclidean_distances

    # # compute pairwise distance matrix
    # distance_matrix = euclidean_distances(weights)

    return weights, vocab, inverse_vocab


def create_model():

    model = tf.keras.Sequential([

        # ## input layer
        # tf.keras.layers.Flatten(input_shape=(28, 28)), ## based on the input

        ## LSTM layer
        tf.keras.layers.LSTM(4), ## units are the dimensionality of the output space

        ## Dropout Layer
        tf.keras.layers.Dropout(.2, input_shape=(4,)), ## fraction of input units to drop, and input shape

        ## Softmax Output Layer
        tf.keras.layers.Softmax()
    ])
    

    return model

def train_model(model, train_data, train_labels):

    model.fit(train_data, train_labels, epochs=10)

    return model

def test_model(model, test_data, test_labels):

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

    print("Test loss is: ", test_loss)
    print("Test accuracy is: ", test_acc)


def main():
    token_contents = "value"
    features, labels = get_data(token_contents)

    kf = KFold(n_splits=10)
    kf.get_n_splits(features)

    model = create_model()
    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    for train_indices, test_indices in kf.split(features):
        
        loss=0
        for index in train_indices:
            loss += model.train_on_batch(features[index], labels[index])

        print(f'TRAIN: {train_indices}, TEST: {test_indices}')

        # for index in test_indices:
            # test_loss, test_acc = model.evaluate(validation_features,  validation_labels, verbose=2)
            # print("Test loss is: ", test_loss)
            # print("Test accuracy is: ", test_acc)


    # tenfold_features = [[],[],[],[],[],[],[],[],[],[]]
    # tenfold_labels = [[],[],[],[],[],[],[],[],[],[]]
    # for i, feature in enumerate(features):
    #     j = i%10
    #     tenfold_features[j].append(feature)
    #     tenfold_labels[j].append(labels[i])

    # ## create and compile model
    # model = create_model()
    # model.compile(optimizer='adam',
    #         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         metrics=['accuracy'])


    # for i, features in enumerate(tenfold_features):
    #     ## each subset will be used as the validation set exactly once, and part of the training set all other times
    #     validation_features = tenfold_features.pop(i)
    #     validation_labels = tenfold_labels.pop(i)


    #     loss = 0
    #     for j, batch in enumerate(tenfold_features):

    #         for k, url in enumerate(batch):
    #             loss += model.train_on_batch(url, tenfold_labels[j][k])
    #         print('Trained on {} of 9 datasets'.format(j))

    #     ## test model
    #     test_loss, test_acc = model.evaluate(validation_features,  validation_labels, verbose=2)

    #     print("Test loss is: ", test_loss)
    #     print("Test accuracy is: ", test_acc)

    #     ## insert back into array
    #     tenfold_features.insert(i, validation_features)
    #     tenfold_labels.insert(i, validation_labels)

        ##  save model
        model.save("lstm_{}".format(token_contents))

        print("Finished epoch {}".format(i))

if __name__ == '__main__':
    main()
