from typing import List, Tuple
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
from word2vec import get_CBOW, get_data_and_labels
from metrics import get_precision, get_recall, get_f1, get_accuracy

## LSTM based on DeepXSS by Yong Fang, Yang Li, Liang Liu, Cheng Huang


## Deep Learning Model ------------------------------------------------------------------------------------------------


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


def lstm_model(token_contents: str, cross_val=10, num_elems=-1) -> List[dict]:

    cbow_weights, vocab, _ = get_CBOW(token_contents)
    features, labels = get_data_and_labels(token_contents, vocab, (np.array([1,0]), np.array([0,1])) )
    
    if num_elems > 0:
        features = features[:num_elems]
        labels = labels[:num_elems]
    
    features = sequence.pad_sequences(features, padding='post')

    ## for future work we can try a callback in model.fit where we set callbacks=[reduce_lr]
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.05,
    #                           patience=5, min_lr=0.001)  ## reduce learning rate on plateau callback

    ## future work could try a different optimizer such as: model_optimizer = tf.keras.optimizers.SGD(learning_rate=0.6, momentum=0.9)  ## specific model optimizer using SGD


    kf = KFold(n_splits=cross_val) ## shuffle=True
    i=0

    all_results = []

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
            epochs=10,
            verbose=1,
            batch_size=16,
        )

        predictions = model.predict(np.array(features[test_indices]))
        predictions = np.argmax(predictions, axis=1)
        max_labels = np.argmax(labels[test_indices], axis=1)

        result_of_fold = {
            'precision': get_precision(predictions, max_labels),
            'recall': get_recall(predictions, max_labels),
            'f1': get_f1(predictions, max_labels),
            'accuracy': get_accuracy(predictions, max_labels)
        }
        all_results.append(result_of_fold)

    return all_results



if __name__ == '__main__':
    results = lstm_model()
