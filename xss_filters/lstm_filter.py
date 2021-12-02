import tensorflow as tf

## for CBOW
import keras
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
import random

## for tokens
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


## Word2vec using CBOW ------------------------------------------------------------------------------------------------
## used for reference: https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html

def create_vocab(tokens):
    vocab = {}  
    index = 1 # start indexing from 1
    vocab['<pad>'] = 0  # add a padding token
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1
    vocab_size = len(vocab)

    inverse_vocab = {index: token for token, index in vocab.items()}

    return vocab, inverse_vocab

def vectorize(tokens, vocab):
    example_sequence = [vocab[word] for word in tokens]
    return example_sequence

def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size*2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word   = []            
            start = index - window_size
            end = index + window_size + 1
            
            context_words.append([words[i]
                                 for i in range(start, end) 
                                 if 0 <= i < sentence_length 
                                 and i != index])
            label_word.append(word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            
            y = np_utils.to_categorical(label_word, vocab_size)
            
            yield (x, y)

def init_CBOW(vocab_size, embed_size, window_size):
    cbow = Sequential()
    cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
    cbow.add(Dense(vocab_size, activation='softmax'))
    cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    # view model summary
    print(cbow.summary())

    return cbow

    # # visualize model structure
    # from IPython.display import SVG
    # from keras.utils.vis_utils import model_to_dot

    # SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False, 
    #                 rankdir='TB').create(prog='dot', format='svg'))


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
    
    train_data = []
    tokens = []

    # loop for all benign DMOZ urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dmoz_dir.txt_0--1.dat')):
        
        train_data.append(tokenized_url)

        for token in tokenized_url.token_list:
            tokens.append(token.value)

    # loop for first 500 malicious urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dec_xss_urls.txt_0--1.dat')):
        
        train_data.append(tokenized_url)

        for token in tokenized_url.token_list:
            tokens.append(token.value)

    ## add data

    random.shuffle(train_data)
    random.shuffle(tokens)

    ## CBOW

    vocab, inverse_vocab = create_vocab(tokens)
    
    corpus = [[vocab[token.value] for token in tokenized_url.token_list] for tokenized_url in train_data] ## init corpus of word_ids

    embed_size = 25
    vocab_size = len(vocab)
    window_size = 2 # context window size
    print("Vocabulary size is: ", vocab_size)

    ## test cbow for some samples
    i = 0

    # for tokenized_url.token_list in train_data:
    for x, y in generate_context_word_pairs(corpus=corpus, window_size=window_size, vocab_size=vocab_size):
        if 0 not in x[0]:
            print('Context (X):', [inverse_vocab[w] for w in x[0]], '-> Target (Y):', inverse_vocab[np.argwhere(y[0])[0][0]]) ## this is just for testing the first 10 context_word_pairs
        
            if i == 10:
                break
            i += 1

    cbow = init_CBOW(vocab_size, embed_size, window_size) ## init cbow architecture


    ## train model
    for epoch in range(1, 5):
        loss = 0.
        i = 0
        for x, y in generate_context_word_pairs(corpus=corpus, window_size=window_size, vocab_size=vocab_size):
            i += 1
            loss += cbow.train_on_batch(x, y)
            if i % 10000 == 0:
                print('Processed {} (context, word) pairs'.format(i))

        print('Epoch:', epoch, '\tLoss:', loss)
        print()


    ## save model if training
    cbow.save('cbow_model')


    ## load model
    ## cbow = keras.models.load_model('cbow_model')

    weights = cbow.get_weights()[0]
    weights = weights[1:]
    print(weights.shape)

    pd.DataFrame(weights, index=list(inverse_vocab.values())[1:]).head()

    from sklearn.metrics.pairwise import euclidean_distances

    # compute pairwise distance matrix
    distance_matrix = euclidean_distances(weights)
    print(distance_matrix.shape)


    # view contextually similar words
    similar_words = {search_term: [inverse_vocab[idx] for idx in distance_matrix[vocab[search_term]-1].argsort()[1:6]+1] 
                    for search_term in ['/']}

    print(similar_words)

if __name__ == '__main__':
    main()
