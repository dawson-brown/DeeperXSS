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

    #load malicious urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dec_xss_urls.txt__20211203-134417_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 1 for malicious
        # labels.append(np.array([0,1]))
        labels.append(1)

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

    print(pd.DataFrame(weights, index=list(inverse_vocab.values())).head())

    return weights, vocab, inverse_vocab


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
