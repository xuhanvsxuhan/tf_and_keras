'''Trains a (very simple) LSTM language model on a corpus of documents using Keras'''
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

# Kludgy function for looking up a word by its vocabulary index
def index_lookup(num, df, word_column=0, index_column=1, word_dtype='unicode'):
	words = np.array(df.iloc[:, word_column], dtype=word_dtype)
	indices = np.array(df.iloc[:, index_column])
	spot = np.where(indices == num)
	return words[spot][0]

# Loading the corpus, previously stored as a 1D array of integers (i.e. vocab indices)
int_corpus = np.array(pd.read_csv(input))

#global parameters
TRAIN_IN_BATCHES = True
EMBED_SEQUENCES = True
EPOCHS = 4
BATCH_SIZE = 32
WINDOW_SIZE = 10

# Doing some math to make the word windows
vocab_size = np.max(int_corpus)
num_tokens = len(int_corpus)
num_windows = (num_tokens / (WINDOW_SIZE + 1)) + (num_tokens % (WINDOW_SIZE + 1))
num_batches = num_windows / BATCH_SIZE

# Building and compiling the model
model = Sequential()
model.add(LSTM(128, input_shape=(WINDOW_SIZE, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

if TRAIN_IN_BATCHES:
    for epoch in range(EPOCHS):
        print('Epoch number %s' %epoch)
        
        # Training the model on batches
        for batch_num in range(num_batches):
            start = batch_num * BATCH_SIZE
            end = start + BATCH_SIZE
            current_windows = range(num_windows)[start:end]
            
            # Setting up empty arrays for the training data
            X = np.zeros([BATCH_SIZE, WINDOW_SIZE, vocab_size], dtype=int)
            y = np.zeros([BATCH_SIZE, vocab_size + 1], dtype=int)
            
            # Converting the 1D arrays of integers to 2D matrics of 1-hot vectors
            for window_num in current_windows:
                max = window_num + WINDOW_SIZE
                phrase = int_corpus[window_num:max]
                for j in range(WINDOW_SIZE):
                    X[window_num - (batch_num * BATCH_SIZE), j, phrase[j]] = 1
                next_word = int_corpus[max]
                y[window_num - (batch_num * BATCH_SIZE), next_word] = 1
            
            # Removing dummy column from y
            y = np.delete(y, 0, 1)
            
            # Train model on current batch of windows
            model.train_on_batch(X, y)

else:
    # Setting up the training data
    X = np.zeros([num_windows, WINDOW_SIZE, vocab_size], dtype=int)
    y = np.zeros([num_windows, vocab_size + 1], dtype=int)
    
    for i in range(num_windows):
    	max = i + WINDOW_SIZE
    	phrase = int_corpus[i:max]
    	for j in range(WINDOW_SIZE):
    		X[i, j, phrase[j]] = 1
    	next = int_corpus[max]
    	y[i, next] = 1
    
    # Removing the empty first column from y
    y = np.delete(y, 0, 1)
    
    # Training the model
    model.fit(X, y, BATCH_SIZE=128, epochs=EPOCHS)
