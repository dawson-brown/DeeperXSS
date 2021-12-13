from typing import Any, Tuple
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
import dataclasses as dc

## for tokens
import pickle as pk
from pickle import dump, load
from data.tokenizer import URLTokens, JSToken 


## Word2vec using CBOW
## used for reference:
# https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html
# https://www.jasonosajima.com/word2vec


def load_tokenized_urls(filename: str) -> URLTokens:

    with open(filename, "rb") as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break

def get_data_and_labels(token_contents, vocab, bin_labels: Tuple[Any, Any]):

    labels = []

    #load benign urls
    tokenized_urls = []
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dmoz_dir.txt__20211203-134415_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 0 for benign
        # labels.append(np.array([1,0]))
        labels.append(bin_labels[0])

    #load malicious urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data/dec_xss_urls.txt__20211203-134417_0--1.dat')):
        tokenized_urls.append(tokenized_url) ## 1 for malicious
        # labels.append(np.array([0,1]))
        labels.append(bin_labels[1])

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

    return weights, vocab, inverse_vocab


def load_tokenized_urls(filename: str) -> URLTokens:

    with open(filename, "rb") as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break

def create_vocab(tokens):
    vocab = {}  
    index = 1 # start indexing from 1
    vocab['<pad>'] = 0  # add a padding token
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1

    inverse_vocab = {index: token for token, index in vocab.items()}

    return vocab, inverse_vocab

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

            ## inputs for model training, each x and y is counted as a batch
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

def train_CBOW(token_contents, num_epochs):

    if token_contents == "type":
        vocab_name = "vocab_type.pickle"
        inv_vocab_name = "inv_vocab_type.pickle"
        model_name = "cbow_model_token_type"
    else:
        vocab_name = "vocab_value.pickle"
        inv_vocab_name = "inv_vocab_value.pickle"
        model_name = "cbow_model_token_value"


    tokenized_urls = []
    tokens = []

    #load benign urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data\dmoz_dir.txt__20211203-134415_0--1.dat')):
        
        tokenized_urls.append(tokenized_url)

        for token in tokenized_url.token_list:
            token_dict = dc.asdict(token)
            tokens.append(token_dict[token_contents])
        # if i > 500: ##for testing purposes
        #     break

    #load malicious urls
    for i, tokenized_url in enumerate(load_tokenized_urls('data\dec_xss_urls.txt__20211203-134417_0--1.dat')):
        
        tokenized_urls.append(tokenized_url)

        for token in tokenized_url.token_list:
            token_dict = dc.asdict(token)
            tokens.append(token_dict[token_contents])
        # if i > 500: ##for testing purposes
        #     break

    ## DO NOT SHUFFLE TOKENS OR YOU WILL NEVER BE ABLE TO PROPERLY RECREATE THE MODEL

    ## CBOW

    vocab, inverse_vocab = create_vocab(tokens)

    with open(vocab_name, 'wb') as handle:
        dump(vocab, handle)

    with open(inv_vocab_name, 'wb') as handle:
        dump(inverse_vocab, handle)
    
    corpus = [[vocab[dc.asdict(token)[token_contents]] for token in tokenized_url.token_list] for tokenized_url in tokenized_urls] ## init corpus of word_ids


    ## shuffle corpus for training
    random.seed(5318008)
    random.shuffle(corpus)

    if token_contents == "type":
        embed_size = 10
    else:
        embed_size = 50

    vocab_size = len(vocab) ## vocab size
    window_size = 2 ## context window size

    print("Vocabulary size is: ", vocab_size)


    cbow = init_CBOW(vocab_size, embed_size, window_size) ## init cbow architecture

    ## train model
    for epoch in range(0, num_epochs):
        loss = 0.
        i = 0
        for x, y in generate_context_word_pairs(corpus=corpus, window_size=window_size, vocab_size=vocab_size):
            i += 1
            ## train cbow model on each tokenized_url in corpus
            loss += cbow.train_on_batch(x, y)
            if i % 10000 == 0:
                print('Processed {} (context, word) pairs'.format(i))
            if epoch == 0:
                if i < 10:
                    print('Context (X):', [inverse_vocab[w] for w in x[0]], '-> Target (Y):', inverse_vocab[np.argwhere(y[0])[0][0]]) ## this is just for testing the first 10 context_word_pairs

        print('Epoch:', epoch+1, '\tLoss:', loss)

        ## save model after every epoch
        cbow.save(model_name)


def main():
    
    # train_CBOW("value", 5)

    cbow = keras.models.load_model('cbow_model_token_type')

    with open('vocab_type.pickle', 'rb') as handle:
        vocab = pk.load(handle)
    with open('inv_vocab_type.pickle', 'rb') as handle:
        inverse_vocab = pk.load(handle)


    weights = cbow.get_weights()[0]
    weights = weights[1:]
    print(weights.shape)

    # print(pd.DataFrame(weights, index=list(inverse_vocab.values())[1:]).head())

    from sklearn.metrics.pairwise import euclidean_distances

    # compute pairwise distance matrix
    distance_matrix = euclidean_distances(weights)


    search_terms = []
    for i in range(1, len(inverse_vocab)):
        if i == 6:
            break
        search_terms.append(inverse_vocab[i])

    # view contextually similar words
    similar_words = {search_term: [inverse_vocab[idx] for idx in distance_matrix[vocab[search_term]-1].argsort()[1:6]+1] 
                    for search_term in search_terms}#search_terms}

    print(similar_words)


if __name__ == '__main__':
    main()
