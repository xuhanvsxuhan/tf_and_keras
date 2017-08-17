# tf-and-keras
This is a collection of simple scripts for doing NLP in TensorFlow and Keras. The models they implement usually rely on 2 general methods: dense vector representations for modeling characters, words, and other atomic units of language; and recurrent neural networks (RNNs), namely long short-term memory networks (LSTMs), for sequence modeling and classification. 

## what's included
  1. text_helpers.py is a bargain bin of classes and functions used in the other scripts. ```make_skipgrams()``` is kind of cool, and the batch generators can be pretty useful if you're not already using ones from another machine learning library. 
  2. slow_text.py sort of reproduces the [fastText](https://github.com/facebookresearch/fastText) algorithm released by [Facebook AI Research](https://research.fb.com/category/facebook-ai-research-fair/). It's not as fast, and it's a little hand-wavy with the assumptions behind the original model, but the main components are the same: feature vectors, embedding lookup, and a linear classifier with softmax cross-entropy loss. 
  3. lstm_lm.py implements a very simple [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) language model in Keras. Given a sequence of n words, it is trained to predict the n+1th word. Fancy!
  4. lstm_phrases_keras.py performs the slightly more complex task of phrase/sentence/document classification using LSTMs, also in Keras. Because Keras does not natively support attention layers, this model only uses the final output from the LSTM as the dense sequence representation, rather than a weighted sum of all the outputs.

## using the scripts
All of these scripts take CSV files as their input and convert them to Pandas DataFrames before model training; this is largely to facilitate data transfer between R and Python. As of now, the CSV files must be in a document-level format, i.e. with one row per document. The functions use ScikitLearn's built-in count vectorizers to vectorize the text data; Numpy functions to shape the data before feeding to a TF graph or Keras model instance; and TF or Keras to instantiate the model and perform the actual calculations. 

As of now, these are meant to be used in a Python interpreter, not from the command line. Functionality for the latter will be added at some point in the near future to promote faster prototyping and model tuning.

## system requirements
To use these modules, you'll need the [Pandas](http://pandas.pydata.org/), [ScikitLearn](http://scikit-learn.org/stable/), [Numpy](http://www.numpy.org/), [Keras](https://keras.io/), and [TensorFlow](https://www.tensorflow.org/) modules, in addition to a working installation of Python (the code was written using 2.7.x). The code is also not optimized for speed or efficiency, and calling certain functions on large datasets may cause Python to throw a MemoryError. 

## references
Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473. 2014 Sep 1. [pdf](https://arxiv.org/pdf/1409.0473.pdf)

Cao K, Rei M. A joint model for word embedding and word morphology. arXiv preprint arXiv:1606.02601. 2016 Jun 8. [pdf](https://arxiv.org/pdf/1606.02601.pdf)

Kim Y, Jernite Y, Sontag D, Rush AM. Character-Aware Neural Language Models. InAAAI 2016 Feb 12 (pp. 2741-2749). [pdf](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12489/12017)

Mikolov T, Sutskever I, Chen K, Corrado GS, Dean J. Distributed representations of words and phrases and their compositionality. InAdvances in neural information processing systems 2013 (pp. 3111-3119). [pdf](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

Sutskever I, Vinyals O, Le QV. Sequence to sequence learning with neural networks. InAdvances in neural information processing systems 2014 (pp. 3104-3112). [pdf](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

