from typing import List
import tensorflow as tf

## for deep learning and cbow
import keras
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.models import Model

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda

from sklearn.model_selection import KFold
import numpy as np


# visualize CBOW model structure
from keras.utils.vis_utils import model_to_dot

## helper libraries
from sklearn.model_selection import KFold
from keras.layers import Embedding

## for tokens
import pickle as pk
from word2vec import get_CBOW, get_data_and_labels
from metrics import get_precision, get_recall, get_f1, get_accuracy
from data.tokenizer import URLTokens, JSToken

## LSTM based on DeepXSS by Yong Fang, Yang Li, Liang Liu, Cheng Huang

def lstm_model(token_contents: str, cross_val=10, num_elems=-1) -> List[dict]:

    weights, vocab, _ = get_CBOW(token_contents)
    features, labels = get_data_and_labels(token_contents, vocab, (0, 1))

    if num_elems > 0:
        features = features[:num_elems]
        labels = labels[:num_elems]

    features = sequence.pad_sequences(features, padding='post')

    kf = KFold(n_splits=cross_val)
    i=0

    all_results = []

    for train_indices, test_indices in kf.split(features, labels):
        i+=1
        print("\n Beginning fold {}:\n".format(i))
        
        model = Sequential()
        model.add(Embedding(weights.shape[0], # maybe +1? see tutorial
            weights.shape[1],
            weights=[weights],
            input_length=features.shape[1],
            trainable=False))
        model.add(tf.keras.layers.LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # see: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
        _history = model.fit(x=features[train_indices],
            y=labels[train_indices],
            epochs=2,
            batch_size=32
        )

        predictions = model.predict(np.array(features[test_indices]))
        predictions = np.rint(predictions.flatten())
        expected = labels[test_indices]

        result_of_fold = {
            'precision': get_precision(predictions, expected),
            'recall': get_recall(predictions, expected),
            'f1': get_f1(predictions, expected),
            'accuracy': get_accuracy(predictions, expected)
        }
        all_results.append(result_of_fold)

    return all_results


if __name__ == '__main__':
    lstm_model()
