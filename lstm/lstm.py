import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


## init data

train_data = []
train_labels = []

test_data = []
test_labels = []

## LSTM based on DeepXSS by Yong Fang, Yang Li, Liang Liu, Cheng Huang

## decoder

## generalization

## tokenization

## Word2vec

## Deep Learning Model

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

## model compilation (may need to alter this)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
