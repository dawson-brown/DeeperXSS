import tensorflow as tf

## for CBOW

from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda

# visualize CBOW model structure
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

## helper libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## for tokens
from pickle import dump, load
from ..data.tokenizer import URLTokens, JSToken

## LSTM based on DeepXSS by Yong Fang, Yang Li, Liang Liu, Cheng Huang

def load_tokenized_urls(filename: str):

    print('Pickle Load:\n')
    with open(filename, "rb") as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


## Word2vec using CBOW ------------------------------------------------------------------------------------------------
## used for reference: https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html

def word2vec():
    return


## Deep Learning Model ------------------------------------------------------------------------------------------------

def create_model():

    model = tf.keras.Sequential([

        ## input layer
        tf.keras.layers.Flatten(input_shape=(28, 28)), ## based on the input

        ## LSTM layer
        tf.keras.layers.LSTM(4), ## units are the dimensionality of the output space

        ## Dropout Layer
        tf.keras.layers.Dropout(.2, input_shape=(4,)), ## fraction of input units to drop, and input shape

        ## Softmax Output Layer
        tf.keras.layers.Softmax()
    ])

    return model

## model compilation (may need to alter this)

def compile_model(model):
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def train_model(model, train_data, train_labels):

    model.fit(train_data, train_labels, epochs=10)

    return model

def test_model(model, test_data, test_labels):

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

    print("Test loss is: ", test_loss)
    print("Test accuracy is: ", test_acc)


def main():
    
    for tokenized_url in load_tokenized_urls(''):
        for token in tokenized_url:
            pass
        train_data = []
        train_labels = []

        test_data = []
        test_labels = []

    

if __name__ == '__main__':
    main()