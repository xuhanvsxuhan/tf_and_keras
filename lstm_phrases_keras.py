'''Trains a bidirectional LSTM to classify phrases (code from Keras documentation)'''
import numpy as np
import pandas as pd

from keras import optimizers
from keras.models import Sequential
from keras.initializers import RandomUniform
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Masking
from sklearn.model_selection import StratifiedShuffleSplit

'''Some helper functions for manipulating the data'''
#thresholds probabilities for binary classification
def bin_thresh(vals, threshold=.5):
	out = np.zeros([len(vals)], dtype=int)
	for i in range(len(vals)):
		if vals[i] >= threshold:
			out[i] = 1
	return out

# Global parameters
BATCH_SIZE = 128
EMBEDDING_SIZE = 200
LSTM_SIZE = 100
EPOCHS = 4
TRIM_TO = 50

# Importing the phrase data
int_sents = np.array(pd.read_csv(corpus_location))
vocab = pd.read_csv(vocabulary_location)
vocab_dict = dict(zip(vocab.iloc[:, 0], vocab.iloc[:, 1]))
embedding_matrix = np.array(pd.read_csv(embedding_location))
targets = pd.read_csv(targets_location)

# Other parameters for the LSTM
max_features = embedding_matrix.shape[0]
max_length = int_sents.shape[1]

'''Building, training, and evaluating the model'''
# Getting the training, test, and validation indices for the batch generator
sss = StratifiedShuffleSplit(n_splits=1, test_size=.25)
for train, test in sss.split(int_sents, targets):
    X_train, X_test = int_sents[train], int_sents[test]
    y_train, y_test = targets[train], targets[test]

for train, val in sss.split(X_train, y_train):
    X_train, X_val = X_train[train], X_train[val]
    y_train, y_val = y_train[train], y_train[val]

# Optionally initializing the embedding weights with the pretrained vectors
def my_init(shape, dtype=None):
    return embedding_matrix

# Building the model graph
model = Sequential()
model.add(Embedding(max_features, EMBEDDING_SIZE, embeddings_initializer=my_init, mask_zero=True))
model.add(Bidirectional(LSTM(LSTM_SIZE)))
model.add(Dropout(.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training the model
model.fit(X_train, y_train, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS, validation_data=[X_val, y_val])

#evaluating the model
guesses = model.predict(X_test)
threshed = bin_thresh(guesses)
