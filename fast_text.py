'''
A not-so-fast implementation of fastText in Tensorflow;
Based on the paper "Bag of Tricks for Efficient Text Classification" by Joulin, Grave, Bojanowski, and Mikolov;
Uses TF-IDF feature normalization and a couple of weight matrices in ScikitLearn and TensorFlow.
'''
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from text_helpers import opposites, IndexBatchGenerator, pad_integers, to_integer

# Quick function for resetting the graph
def reset():
    tf.reset_default_graph()

# Importing the data
docs = pd.read_csv()
targs = np.array(pd.read_csv())

# Converting phrases to BoF with TF-IDF normalization
vec = TfidfVectorizer()
features = vec.fit_transform(docs).toarray()

# Fetching the pretrained word vectors
pretrained = np.array(pd.read_csv())

# Splitting into training and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)
for train, test in sss.split(features, targets):
    X_train, X_test = features[train], features[test]
    y_train, y_test = targets[train], targets[test]

for train, val in sss.split(X_train, y_train):
    X_train, X_val = X_train[train], X_train[val]
    y_train, y_val = y_train[train], y_train[val]

# Setting the model parameters
num_features  = features.shape[1]
embedding_size = 200
batch_size = 128
epochs = 30
learning_rate = 1e-1
num_classes = 2
display_step = 10

# Initializers to use for the variables
norm = tf.random_normal_initializer()
unif = tf.random_uniform_initializer(minval=-.01, maxval=.01)
zeros = tf.zeros_initializer()

# Placeholders for the feature vectors and the targets
x = tf.placeholder(tf.float32, [None, num_features])
y = tf.placeholder(tf.float32, [None, num_classes])

# Initializing the embedding matrix for the features
embeddings = tf.get_variable('embeddings', [num_features, embedding_size], dtype=tf.float32)
averaged_features = tf.matmul(x, embeddings)

# Adding the biases and weights for the linear transformation
dense_weights = tf.get_variable('dense_weights', [embedding_size, num_classes], dtype=tf.float32, initializer=unif)
dense_biases = tf.get_variable('dense_biases', [num_classes], dtype=tf.float32, initializer=zeros)
dense = tf.matmul(averaged_features, dense_weights) + dense_biases

# Getting the predictions to evaluate accuracy outside of the graph
probs = tf.nn.softmax(dense)
preds = tf.argmax(probs, 1)

# Calculating weighted cross-entropy loss; squared weights may be used for L2 regularization
squared_weights = tf.reduce_sum(tf.square(dense_weights))
squared_embeddings = tf.reduce_sum(tf.square(embeddings))
loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=dense, pos_weight=tf.constant([.1, 10])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

# Training the model in batches
with tf.Session() as sess:
    sess.run(init)
    test_dict = {x: X_test, y: y_test}
    val_dict = {x: X_val, y: y_val}
    for e in range(epochs):
        print('\nStarting epoch number %s' %e)
        epoch_loss = 0
        X = X_train
        y_ = y_train
        bg = IndexBatchGenerator(range(len(X)), batch_size, shuff=True)
        step = 1
        for batch in bg.batches:
            batch_dict = {x: X[batch], y: y_[batch]}
            sess.run(optimizer, feed_dict=batch_dict)
            cost = sess.run(loss, feed_dict=batch_dict)
            epoch_loss += cost
            if step % display_step == 0:
                print("Iter " + str(step*batch_size) + ", Mean Loss= " + \
					  "{:.6f}".format(cost))
            step += 1
        mean_epoch_loss = np.true_divide(epoch_loss, step)
        print('Mean epoch loss=%.6f' %mean_epoch_loss)

