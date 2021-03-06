# Name: Cory Nezin  
# Date: 02/08/2018
# Task: Perform a greedy gradient attack on a recurrent neural network

import tensorflow as tf
import numpy as np
import review_proc as rp, preprocess, rnn, word2vec
import matplotlib.pyplot as plt
import plotutil as putil
import argparse, os, sys, random, re

parser = argparse.ArgumentParser( \
    description = 'Perform a greedy semantic attack on a recurrent neural network')

parser.add_argument('-wi',
    help = 'Word to Index dictionary mapping string to integer',
    default = 'word_to_index.npy')

parser.add_argument('-iv',
    help = 'Index to Vector numpy array mapping integer to vector',
    default = 'index_to_vector.npy')

args = parser.parse_args()
word_embedding_filename = args.iv
word_to_embedding_index_filename = args.wi

try:
    word_embedding = np.load(word_embedding_filename)
    word_to_embedding_index = np.load(word_to_embedding_index_filename).item()
except FileNotFoundError:
    print('Word embedding not found, running word2vec')
    word2vec.w2v(corpus_filename = './corpus/imdb_train_corpus.txt')

embedding_norm = np.linalg.norm(word_embedding,axis=1)
embedding_norm.shape = (10000,1)
normalized_word_embedding = word_embedding / embedding_norm
m = word_to_embedding_index
# Reverse dictionary to look up words from indices
embedding_index_to_word = dict(zip(m.values(), m.keys()))

g = tf.Graph()
rv = rp.review('./aclImdb/test/posneg/5164_10.txt')
#ii = [242,242,837,348]
#jj = [11,215,467,680]
#ii = [369,369,837]
#jj = [72,292,495]
ii = [882,348]
jj = [165,495]
w = [embedding_index_to_word[i] for i in ii]

print(w)
for n,j in enumerate(jj):
    rv.tokens.insert(j,w[n])
rv.length = rv.length + len(ii)
with g.as_default():
    global_step_tensor = tf.Variable(0, trainable = False, name = 'global_step')
    # Create RNN graph
    r = rnn.classifier(
        batch_size = 1,
        learning_rate = 0.0,
        hidden_size = 16,
        max_time = 740,
        embeddings = word_embedding,
        global_step = global_step_tensor
    )
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
        print(rv.tokens)
        rv.translate(r.max_time,word_to_embedding_index,embedding_index_to_word)
        rv.vec(word_embedding)
        decision, probability, batch_grad = r.infer_dpg(sess,rv)
        print(probability)
        rnn_sentiment = 'pos' if not decision[0] else 'neg'
        print('Neural Net Decision: ',rnn_sentiment,' Actual: ',rv.sentiment)


