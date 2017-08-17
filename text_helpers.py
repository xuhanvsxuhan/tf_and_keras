"""
Objects and functions to support text corpus storage and manipulation.

Generally, the functions are good-enough but not great, i.e. they prioritize
readability over compactness and readability, so feel free to adapt them to
suit your particular project.
"""
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from gensim.models.word2vec import *

# Batch iterators for working with large datasets
class BatchGenerator:
	"""
	Batch geneartor for X and y arrays.
	
	This is great for small-ish datasets where you can hold the full X and y
	arrays in memory, but for larger datasets, you'll want to use the 
	IndexBatchGenerator below, which returns an iterable of indices rather than
	of subsets of the data.
	"""
	def __init__(self, X, y, size=32):
		max = len(y)
		n_batches = (max / size) + (max % size)
		self.X = np.array_split(X, n_batches)
		self.y = np.array_split(y, n_batches)

class IndexBatchGenerator:
	"""
	Another batch generator.
	
	This takes a range of indices (usually 0 to the length of the full dataset)
	and returns a(n optionally shuffled) iterable of batch indices to use for
	training and testing.
	"""
	def __init__(self, indices, size=32, shuff=False, seed=None):
		max = len(indices)
		n_batches = (max / size)
		if max % size != 0:
			n_batches += 1
		if shuff:
			if seed is not None:
				indices = shuffle(indices, seed)
			else:
				indices = shuffle(indices)
		self.batches = np.array_split(indices, n_batches)

class BalancedBatch:
    def __init__(self, X, y, shuffle_batch=True):
        comb = np.concatenate([X, y], 1)
        pos = comb[np.where(comb[:, -1] == 1)]
        neg = comb[np.where(comb[:, -2] == 1)]
        bal = balance(pos, neg)
        if shuffle_batch:
            bal = shuffle(bal)
        self.X = bal[:, :-2]
        self.y = bal[:, -2:]

# Helper functions for working with text datasets
def balance(pos, neg):
	num_pos = len(pos)
	samps = np.random.randint(0, len(neg), num_pos)
	return np.concatenate([pos, neg[samps]])

def to_unicode(corpus, encoding='utf-8', errors='replace'):
    utf_corpus = [unicode(text, encoding=encoding, errors=errors) for text in corpus]
    return utf_corpus

# Converts a single unicode string to a list of characters
def word_to_char(word, output_type='list'):
    if output_type == 'array':
        return np.array([char.decode() for char in word])
    elif output_type == 'list':
        return [char.decode() for char in word]

# Converts a list of words to an array of characters
def doc_to_char(doc, concatenate=False):
    if concatenate:
        return np.concatenate(np.array([word_to_char(word) for word in doc]))
    else:
        return [word_to_char(word) for word in doc]

#converts a unicode corpus to string
def to_string(corpus, errors='ignore'):
    str_corpus = [doc.encode('ascii', errors) for doc in corpus]
    return str_corpus

#converts a list of tokens to an array of integers
def to_integer(tokens, vocab_dict):
	out = np.zeros([len(tokens)], dtype=int)
	for i in range(len(tokens)):
		out[i] = vocab_dict[tokens[i]]
	return out

# Looks a dict key up by its values
def get_key(value, dic, pad=0):
	if value == pad:
		out = 'pad'
	else:
		out = dic.keys()[dic.values().index(value)]
	return out

#pads a 1D sequence of words (as vocabulary indices)
def pad_integers(phrase, max_length, padder):
	pad_size = max_length - len(phrase)
	return np.concatenate((phrase, np.repeat(padder, pad_size)))

#formats a corpus for word2vec training in gensim
def gensim_sentences(corpus):
    return [[word for word in doc.lower().split()] for doc in corpus]

#gets the frequency distribution of terms in a corpus
def freq_dist(corpus, vocab=None):
    if vocab is None:
        vocab = new_vectorizer().fit(corpus).vocabulary_
    dist = dict(zip(vocab.keys(), np.zeros(len(vocab), dtype=int)))
    for doc in corpus:
        for word in doc:
            dist[word] += 1
    return dist

#does all manner of preprocessing on an a tokenized corpus
def remove_rare(doc, freqs, min_count=5, replacement='unk'):
	i = 0
	for word in doc:
		if freqs[word] < min_count:
			doc[i] = replacement
		i += 1
	return doc

#wrapper for CountVectorizer with the appropriate token pattern
def new_vectorizer():
    return CountVectorizer(token_pattern=pattern, tokenizer=tokenizer)

#counts the number of hapax legomena in a FreqDist dict
def count_hapaxes(freqs):
    return np.sum(np.array(freqs.values()) == 1)

#pads a 1D sequence of words (as vocabulary indices)
def pad_integers(phrase, max_length, padder):
	pad_size = max_length - len(phrase)
	return np.concatenate((phrase, np.repeat(padder, pad_size)))

#pads an 2D array of (column-vector) word embeddings
def pad_embeddings(phrase, max_length, dtype='float32'):
	if len(phrase.shape) == 1:
		phrase = phrase.reshape([phrase.shape[0], 1])
	pad_size = max_length - phrase.shape[1]
	embedding_size = phrase.shape[0]
	return np.concatenate((phrase, np.zeros([embedding_size, pad_size], dtype=dtype)), axis=1)

# Simple function for making a one-hot vectors
def one_hot(index, size):
    vec = np.zeros([size])
    if index != 0:
        vec.put(index, 1)
    return vec 

# Pairs a sequence of binary values with their opposites
def opposites(targets):
    opps = np.zeros(targets.shape, dtype='int64')
    for i in range(len(targets)):
        opps[i] = (targets[i] - 1) * -1
    return np.array(zip(opps, targets))

'''
# Takes a word index and returns a matrix of one-hot character vectors
def charmat(index, v_size=None, c_size=num_chars, lookup=word_lookup):
    char_seq = np.matmul(one_hot(index, vocabulary_size), lookup)
    out = np.array([one_hot(char, c_size + 1) for char in char_seq])
    return out[:, 1:]

def charseq(index, v_size=None, c_size=num_chars, lookup=word_lookup):
	return np.matmul(one_hot(index, vocabulary_size), lookup)
'''

# Converts a corpus to an array of skipgrams with shape [num_grams, 2 + k], where k
# is the number of negative samples and num_grams = 2 * win_size * num_wins
def make_skipgrams(corpus, winsize=2, k=5, flatten=True):
    num_win = len(range(winsize, len(corpus) - winsize))
    num_grams = 2 * winsize * num_win
    win_width = 2 * winsize + 1
    out = np.zeros([num_win, win_width - 1, 2 + k], dtype=int)
    for i in range(num_win):
        window = corpus[i:i+win_width]
        center = window[winsize]
        del(window[winsize])
        for j in range(len(window)):
            out[i, j, 0] = center
            out[i, j, 1] = window[j]
            out[i, j, 2:] = np.random.randint(0, vocabulary_size, size=k)
    if flatten:
        out = out.reshape([out.shape[0] * (2 * winsize), out.shape[2]])
    return out


# A list of tokens to remove from a corpus during cleaning
drop_tokens = ["\'", "*", "//", "\"", "@", "nr", "\''"]

'''The TextCorpus class: a holder for a corpus, loaded from a CSV file'''
class TextCorpus:
    def __init__(self):
        self.word_frequencies = []
    #
    #loads a corpus from a CSV file
    def load(self, corpus_path, text_column_name=None):
        raw_corpus = pd.read_csv(corpus_path)
        self.raw_data = raw_corpus
        if text_column_name is not None:
            raw_corpus = raw_corpus[text_column_name]
        [[word.lower() for word in sent] for sent in raw_corpus]
        self.str_corpus = raw_corpus
        self.corpus = to_unicode(raw_corpus)
    #
    #adds a new set of docs to the existing corpus
    def add(self, corpus_path, text_column_name=None):
        new_corpus = pd.read_csv(corpus_path)
        if text_column_name is not None:
            new_corpus = new_corpus[text_column_name]
        [[word.lower() for word in sent] for sent in new_corpus]
        self.str_corpus = list(pd.concat([self.str_corpus, new_corpus]))
        self.corpus = to_unicode(self.str_corpus)
    #
    #tokenizes the corpus; the default tokenizer is the PTB from NLTK
    def tokenize(self, pattern=u'(?u)\\b\\w+\\b', tokenizer_function=TreebankWordTokenizer().tokenize):
        vec = CountVectorizer(token_pattern=pattern, tokenizer=tokenizer_function)
        vec.fit(self.corpus)
        toker = vec.build_tokenizer()
        tokenized_corpus = [toker(doc) for doc in self.corpus]
        tokenized_corpus = [[word.lower() for word in sent] for sent in tokenized_corpus]
        self.vocabulary = vec.vocabulary_
        self.word_frequencies = freq_dist(tokenized_corpus, self.vocabulary)
        self.tokens = tokenized_corpus
        self.token_integers = [to_integer(sent, self.vocabulary) for sent in self.tokens]
        self.token_integers = [np.sum((seq, 1)) for seq in self.token_integers]
    #
    #removes stop words and other junk from the corpus
    def clean(self, to_remove=drop_tokens, floor=True, min_count=2, replacement='unk'):
        self.token_check()
        for doc in self.tokens:
            for word in doc:
                if word in to_remove:
                    doc.remove(word)
        if floor:
            [remove_rare(doc, self.word_frequencies, min_count, replacement) for doc in self.tokens]
        self.refresh()
    #
    def refresh(self):
        self.str_corpus = [' '.join(to_string(phrase)) for phrase in self.tokens]
        self.corpus = to_unicode(self.str_corpus)
        self.tokenize()
    #
    #makes sure the corpus is tokenized before running methods that operate on tokens
    def token_check(self):
        if self.word_frequencies == []:
            raise AssertionError('Please tokenize corpus with .tokenize() before cleaning.')
