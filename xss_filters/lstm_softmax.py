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
from keras.wrappers.scikit_learn import KerasClassifier


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

    if token_contents == "type":
        vocab_name = "vocab_type.pickle"
    else:
        vocab_name = "vocab_value.pickle"

    ## open vocab for token type or token value
    with open(vocab_name, 'rb') as handle:
        vocab = pk.load(handle)

    labels = []

    #load benign urls
    tokenized_urls = []
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dmoz_dir.txt__20211203-134415_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 0 for benign
        labels.append(np.array([1,0]))

        if i > 10000: ##for testing purposes
            break

    #load malicious urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dec_xss_urls.txt__20211203-134417_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 1 for malicious
        labels.append(np.array([0,1]))

        if i > 10000: ##for testing purposes
            break    

    vector_urls = []

    ## get token identifiers from CBOW dict for each token
    for tokenized_url in tokenized_urls:

        vector_url = []

        for token in tokenized_url.token_list:
            token_no = vocab[dc.asdict(token)[token_contents]]
            vector_url.append(token_no)
        
        vector_urls.append(vector_url)

    data_labels_zip = list(zip(vector_urls, labels)) ## zip to keep data and labels aligned during shuffle

    random.seed(5318008)

    random.shuffle(data_labels_zip)
    
    vector_urls, labels = zip(*data_labels_zip)
           
    return vector_urls, np.array(labels)

def get_CBOW_weights(token_contents):

    if token_contents == "type":
        model_name = "cbow_model_token_type"
    else:
        model_name = "cbow_model_token_value"

    cbow = keras.models.load_model(model_name)
    weights = cbow.get_weights()[0] ## get CBOW weights for embed layer of model

    return weights


def create_model(cbow_weights, features):

    model = tf.keras.Sequential([

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

        tf.keras.layers.LSTM(50), ## units are the dimensionality of the output space...this might be useful to change if training on token.type

        ## Dropout Layer
        tf.keras.layers.Dropout(.5, input_shape=(50,)), ## fraction of input units to drop, and input shape

        ## Softmax Output Layer
        Dense(2, activation='softmax', name='softmax_output')
    ])
    

    return model


def prediction_precision(predictions, actual):

    total_correct = 0
    for p,l in zip(predictions, actual):
        pred = np.argmax(p)
        label = np.argmax(l)
        if pred == label:
            total_correct+=1

    return total_correct / len(actual)


def main():

    token_contents = "value" ## "type" to train on token.type "value" to train on token.value


    features, labels = get_data(token_contents)
    features = sequence.pad_sequences(features, padding='post')
    cbow_weights = get_CBOW_weights(token_contents)


    ## for future work we can try a callback in model.fit where we set callbacks=[reduce_lr]
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.05,
    #                           patience=5, min_lr=0.001)  ## reduce learning rate on plateau callback

    ## future work could try a different optimizer such as: model_optimizer = tf.keras.optimizers.SGD(learning_rate=0.6, momentum=0.9)  ## specific model optimizer using SGD


    kf = KFold(n_splits=10, shuffle=True)
    i=0

    for train_indices, test_indices in kf.split(features, labels):
        i+=1
        print("\n Beginning fold {}:\n".format(i))

        model = create_model(cbow_weights, features)
        model.compile(optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
        
        # see: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
        _history = model.fit(x=features[train_indices],
            y=labels[train_indices],
            epochs=1,
            verbose=1,
            batch_size=32,
        )

        x = model.predict(np.array(features[test_indices]))
        print(f'Accuracy: {prediction_precision(x, labels[test_indices])}')
        print("Fold {} complete.".format(i))

if __name__ == '__main__':
    main()
