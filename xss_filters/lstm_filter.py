from typing import Tuple
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
from keras.layers import Embedding

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
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dmoz_dir.txt__20211203-134415_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 0 for benign
        labels.append(np.array([1,0]))

        # if i > 10000: ##for testing purposes
        #     break

    #load malicious urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dec_xss_urls.txt__20211203-134417_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 1 for malicious
        labels.append(np.array([0,1]))

        # if i > 10000: ##for testing purposes
        #     break    

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

    weights = weights[1:]
    print(weights.shape) ## print shape of weights

    print(pd.DataFrame(weights, index=list(inverse_vocab.values())[1:]).head())

    from sklearn.metrics.pairwise import euclidean_distances

    # # compute pairwise distance matrix
    # distance_matrix = euclidean_distances(weights)

    return weights, vocab, inverse_vocab


def create_model(cbow_weights, features):

    model = tf.keras.Sequential([

        # ## input layer
        # tf.keras.layers.Flatten(input_shape=(28, 28)), ## based on the input

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
        tf.keras.layers.LSTM(10), ## units are the dimensionality of the output space

        ## Dropout Layer
        tf.keras.layers.Dropout(.5, input_shape=(10,)), ## fraction of input units to drop, and input shape

        ## Softmax Output Layer
        Dense(2, activation='softmax', name='softmax_output')
    ])
    

    return model



def build_data_sets(features, labels, indices) -> Tuple[np.array, np.array]:

    data = list()
    data_labels = list()

    for index in indices:
        data.append(features[index])
        if labels[index] == 0:
            data_labels.append(np.array([1,0]))
        else:
            data_labels.append(np.array([0,1]))

    return np.array(data), np.array(data_labels)


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
    features, labels = get_data(token_contents)
    features = sequence.pad_sequences(features, padding='post')


    ## reduce learning rate on plateau callback

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.05,
                              patience=5, min_lr=0.001)
    model_optimizer = tf.keras.optimizers.SGD(learning_rate=0.15, momentum=0.9)


    cbow = keras.models.load_model('cbow_model_token_value')
    weights = cbow.get_weights()[0]
    kf = KFold(n_splits=10)


    for train_indices, test_indices in kf.split(features, labels):

        model = create_model(weights, features)
        model.compile(optimizer=model_optimizer,
        loss='categorical_crossentropy', ## binary_crossentropy, categorical_crossentropy
        metrics=['accuracy'])


        # print(f'TRAIN: {train_indices}, TEST: {test_indices}')
        # training_set, training_targets = build_data_sets(features, labels, train_indices)
        # testing_set, testing_targets = build_data_sets(features, labels, test_indices)
        
        # see: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
        _history = model.fit(x=features[train_indices],
            y=labels[train_indices],
            epochs=10,
            verbose=1,
            batch_size=32,
            callbacks=[reduce_lr]
        )

        # x = model.predict(np.array(testing_set))
        # print(f'Precision: {prediction_precision(x, testing_targets)}')

if __name__ == '__main__':
    main()
