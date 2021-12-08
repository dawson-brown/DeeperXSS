from typing import List, Tuple
import tensorflow as tf

## for deep learning and cbow
import keras
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras import initializers

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
from word2vec import get_data_and_labels
from metrics import get_precision, get_recall, get_f1, get_accuracy

## LSTM based on DeepXSS by Yong Fang, Yang Li, Liang Liu, Cheng Huang


## Deep Learning Model ------------------------------------------------------------------------------------------------

def create_model(max_url_length, vocab_size):

    model = tf.keras.Sequential([

        Embedding(
            vocab_size, # input size 
            50, #output dimensions (50 matches token.value)
            input_length=max_url_length, # max_url_length
            embeddings_initializer="random_normal"
        ),

        ## LSTM layer

        tf.keras.layers.LSTM(100, input_shape=(1, 442)), ## units are the dimensionality of the output space...this might be useful to change if training on token.type

        ## Dropout Layer
        tf.keras.layers.Dropout(.5, input_shape=(100,)), ## fraction of input units to drop, and input shape

        ## Softmax Output Layer
        Dense(2, activation='softmax', name='softmax_output')
    ])
    

    return model



def lstm_model(token_contents: str, cross_val=10, num_elems=-1) -> List[dict]:

    if token_contents == "type":
        vocab_name = "vocab_type.pickle"
    else:
        vocab_name = "vocab_value.pickle"

    ## open vocab for token type or token value
    with open(vocab_name, 'rb') as handle:
        vocab = pk.load(handle)

    features, labels = get_data_and_labels(token_contents, vocab, (np.array([1,0]), np.array([0,1])) )
    
    if num_elems > 0:
        features = features[:num_elems]
        labels = labels[:num_elems]

    features = sequence.pad_sequences(features, padding='post')

    vocab_size = len(vocab) ## amount of unique tokens in vocab
    max_url_length = len(features[0]) ## maximum url size we pad for


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

        model = create_model(max_url_length, vocab_size)
        model.compile(optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
        
        # see: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
        _history = model.fit(x=features[train_indices],
            y=labels[train_indices],
            epochs=2,
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
    lstm_model()
