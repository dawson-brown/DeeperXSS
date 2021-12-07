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


# visualize CBOW model structure
from keras.utils.vis_utils import model_to_dot

## helper libraries
from sklearn.model_selection import KFold
from keras.layers import Embedding

## for tokens
import pickle as pk
from word2vec import get_CBOW, get_data_and_labels
from data.tokenizer import URLTokens, JSToken

## LSTM based on DeepXSS by Yong Fang, Yang Li, Liang Liu, Cheng Huang

def main():

    token_contents = "value"
    weights, vocab, _ = get_CBOW(token_contents)
    features, labels = get_data_and_labels(token_contents, vocab)
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
