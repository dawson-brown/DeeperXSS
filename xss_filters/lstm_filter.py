from typing import Tuple
import tensorflow as tf

## for deep learning and cbow
import keras
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Model

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
from keras.layers import Embedding, Input

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

def get_data(token_contents, vocab):

    labels = []

    #load benign urls
    tokenized_urls = []
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dmoz_dir.txt__20211203-134415_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 0 for benign
        # labels.append(np.array([1,0]))
        labels.append(0)

        if i > 2000: ##for testing purposes
            break

    #load malicious urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dec_xss_urls.txt__20211203-134417_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 1 for malicious
        # labels.append(np.array([0,1]))
        labels.append(1)

        if i > 2000: ##for testing purposes
            break    

    vector_urls = []
    for tokenized_url in tokenized_urls:

        vector_url = []

        for token in tokenized_url.token_list:
            token_no = vocab[dc.asdict(token)[token_contents]]
            vector_url.append(token_no) ## -1 because weights has no 0 index
        
        vector_urls.append(vector_url)

    data_labels_zip = list(zip(vector_urls, labels)) ## zip to keep data and labels aligned during shuffle

    random.seed(5318008)
    random.shuffle(data_labels_zip)
    
    vector_urls, labels = zip(*data_labels_zip)
           
    return vector_urls, np.array(labels)

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

    # weights = weights[1:]
    # print(weights.shape) ## print shape of weights

    print(pd.DataFrame(weights, index=list(inverse_vocab.values())).head())

    # from sklearn.metrics.pairwise import euclidean_distances

    # # compute pairwise distance matrix
    # distance_matrix = euclidean_distances(weights)

    return weights, vocab, inverse_vocab


def create_model(cbow_weights, features):

    model = tf.keras.Sequential([

        # ## input layer
        # tf.keras.layers.Flatten(input_shape=(28, 28)), ## based on the input
        Input(shape=(features.shape[1],), dtype='int32'),


        # add cbow embeddings
        # see https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        Embedding(
            cbow_weights.shape[0], # maybe +1? see tutorial
            cbow_weights.shape[1],
            weights=[cbow_weights],
            input_length=features.shape[1],
            trainable=False
        ),

        ## LSTM layer
        tf.keras.layers.LSTM(2), ## units are the dimensionality of the output space

        ## Dropout Layer
        tf.keras.layers.Dropout(.2, input_shape=(4,)), ## fraction of input units to drop, and input shape

        ## Softmax Output Layer
        Dense(2, activation='softmax', name='softmax_output')
    ])
    

    return model

def train_model(model, train_data, train_labels):

    model.fit(train_data, train_labels, epochs=10)

    return model

def test_model(model, test_data, test_labels):

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

    print("Test loss is: ", test_loss)
    print("Test accuracy is: ", test_acc)


def prediction_precision(predictions, actual):

    total_correct = 0
    for p,l in zip(predictions, actual):
        pred = np.argmax(p)
        label = np.argmax(l)
        if pred == label:
            total_correct+=1

    return total_correct / len(actual)


def main():

    token_contents = "value"

    weights, vocab, inv_vocab = get_CBOW(token_contents)
    features, labels = get_data(token_contents, vocab)
    features = sequence.pad_sequences(features, padding='post')

    kf = KFold(n_splits=10)

    for train_indices, test_indices in kf.split(features, labels):
        
        # model = create_model(weights, features)
        embedding_layer = Embedding(
            weights.shape[0], # maybe +1? see tutorial
            weights.shape[1],
            weights=[weights],
            input_length=features.shape[1],
            trainable=False
        )

        model = Sequential()
        model.add(Embedding(weights.shape[0], # maybe +1? see tutorial
            weights.shape[1],
            weights=[weights],
            input_length=features.shape[1],
            trainable=False))
        model.add(tf.keras.layers.LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        
        # see: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
        _history = model.fit(x=features[train_indices],
            y=labels[train_indices],
            epochs=5,
            batch_size=32
        )

        scores = model.evaluate(features[test_indices], labels[test_indices], verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == '__main__':
    main()
