'''Trains a (very simple) LSTM language model on a corpus of documents using Keras'''
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers import Dense, Activation, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

import numpy as np
import pandas as pd
import tensorflow as tf

# Kludgy function for looking up a word by its vocabulary index
def index_lookup(num, df, word_column=0, index_column=1, word_dtype='unicode'):
	words = np.array(df.iloc[:, word_column], dtype=word_dtype)
	indices = np.array(df.iloc[:, index_column])
	spot = np.where(indices == num)
	return words[spot][0]

# Loading the corpus, previously stored as a 1D array of integers (i.e. vocab indices)
int_corpus = np.array(pd.read_csv(input))

#global parameters
train_in_batches = True
embed_sequences = True
epochs = 4
batch_size = 32
window_size = 10

# Doing some math to make the word windows
vocab_size = np.max(int_corpus)
num_tokens = len(int_corpus)
num_windows = (num_tokens / (window_size + 1)) + (num_tokens % (window_size + 1))
num_batches = num_windows / batch_size

# Building and compiling the model
model = Sequential()
model.add(LSTM(128, input_shape=(window_size, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

if train_in_batches:
    for epoch in range(epochs):
        print('Epoch number %s' %epoch)
        
        # Training the model on batches
        for batch_num in range(num_batches):
            start = batch_num * batch_size
            end = start + batch_size
            current_windows = range(num_windows)[start:end]
            
            # Setting up empty arrays for the training data
            X = np.zeros([batch_size, window_size, vocab_size], dtype=int)
            y = np.zeros([batch_size, vocab_size + 1], dtype=int)
            
            # Converting the 1D arrays of integers to 2D matrics of 1-hot vectors
            for window_num in current_windows:
                max = window_num + window_size
                phrase = int_corpus[window_num:max]
                for j in range(window_size):
                    X[window_num - (batch_num * batch_size), j, phrase[j]] = 1
                next_word = int_corpus[max]
                y[window_num - (batch_num * batch_size), next_word] = 1
            
            # Removing dummy column from y
            y = np.delete(y, 0, 1)
            
            # Train model on current batch of windows
            model.train_on_batch(X, y)

else:
    # Setting up the training data
    X = np.zeros([num_windows, window_size, vocab_size], dtype=int)
    y = np.zeros([num_windows, vocab_size + 1], dtype=int)
    
    for i in range(num_windows):
    	max = i + window_size
    	phrase = int_corpus[i:max]
    	for j in range(window_size):
    		X[i, j, phrase[j]] = 1
    	next = int_corpus[max]
    	y[i, next] = 1
    
    # Removing the empty first column from y
    y = np.delete(y, 0, 1)
    
    # Training the model
    model.fit(X, y, batch_size=128, epochs=4)
