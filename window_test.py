# Name: Cory Nezin  
# Date: 02/08/2018
# Task: Perform a greedy gradient attack on a recurrent neural network

import tensorflow as tf
import numpy as np
import review_proc as rp, preprocess, rnn, word2vec
import matplotlib.pyplot as plt
import plotutil as putil
import argparse, os, sys, random, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
tot = 8
c = [0]*1024
res_dir = './window/pp/'
for file_name in os.listdir(res_dir):
    ii = np.load(res_dir[:-3]+'ii/'+file_name)
    jj = np.load(res_dir[:-3]+'jj/'+file_name)
    pp = np.load(res_dir[:-3]+'pp/'+file_name)
    rv = rp.review('./aclImdb/test/posneg/'+file_name[:-4]+'.txt')
    if rv.sentiment == 'pos':
        args = np.argsort(pp)
    else:
        args = np.flip(np.argsort(pp),axis=0)
    ii = ii[args]
    jj = jj[args]
    print(file_name)
    minm = float('inf')
    R = tot
    L = 0
    val_found = False
    while True:
        print(minm)
        rv = rp.review('./aclImdb/test/posneg/'+file_name[:-4]+'.txt')
        if not val_found:
            R = R*2
        m = (L + R) // 2
        print(L,m,R)
        g = tf.Graph()
        ix = ii[:m+1]; jx = jj[:m+1]
        w = [embedding_index_to_word[i] for i in ix]
        print(w)
        for n,j in enumerate(args[:m+1]):
            rv.tokens[j] = w[n]
        with g.as_default():
            global_step_tensor = \
                tf.Variable(0, trainable = False, name = 'global_step')
            # Create RNN graph
            r = rnn.classifier(
                batch_size = 1,
                learning_rate = 0.0,
                hidden_size = 16,
                max_time = rv.length,
                embeddings = word_embedding,
                global_step = global_step_tensor
            )
            with tf.Session() as sess:
                tf.train.Saver().restore(sess, 
                    './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
                rv.translate(r.max_time,\
                    word_to_embedding_index,embedding_index_to_word)
                rv.vec(word_embedding)
                decision, probability, batch_grad = r.infer_dpg(sess,rv)
                rnn_sentiment = 'pos' if not decision[0] else 'neg'
                print('Neural Net Decision: ',rnn_sentiment,' Actual: ',rv.sentiment)
                if rnn_sentiment == rv.sentiment and val_found:
                    print('nothing found')
                    L = m + 1
                elif rnn_sentiment != rv.sentiment:
                    minm = min(m+1,minm)
                    print('match found')
                    print('minimum middle: ',minm)
                    val_found = True
                    R = m - 1
                if L > R:
                    print('search complete, minimum value is: ',minm)
                    c[minm] += 1
                    break
print(c)

