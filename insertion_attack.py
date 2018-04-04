# Name: Cory Nezin  
# Date: 02/08/2018
# Task: Perform a greedy search attack on multiple test files

import tensorflow as tf
import numpy as np
import review_proc as rp, preprocess, rnn, word2vec
import matplotlib.pyplot as plt
import plotutil as putil
import argparse, os, sys, random, re, copy

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

name_list = np.load('name_list.npy')
print(name_list[0:5])
top_k = 2**10
for name in ['5164_10.npy']:
    for divs in [2,4,8]: 
        test_file = name[:-4]+'.txt'
        rv = rp.review('./aclImdb/test/posneg/' + test_file)
        per = (rv.length+divs) // divs
        rv.translate(rv.length,word_to_embedding_index,embedding_index_to_word)
        rv.vec(word_embedding)
        min_arr = np.array([float('inf')]*divs)
        max_arr = np.array([float('-inf')]*divs)
        g = tf.Graph()
        with g.as_default():
            global_step_tensor = tf.Variable(0,trainable=False,name='global_step')
            # Create RNN graph
            r = rnn.classifier(
                batch_size = top_k*divs,
                learning_rate = 0.0,
                hidden_size = 16,
                max_time = per,
                embeddings = word_embedding,
                global_step = global_step_tensor
            )
            with tf.Session() as sess:
                tf.train.Saver().restore(\
                    sess, './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
                print('Processing ' + test_file)
                ii = [0]*divs; jj = [0]*divs;
                for ins_location in range(per):
                    print(ins_location)
                    _,p,_,im = r.infer_insert(sess,rv,ins_location,divs,top_k)
                    p = p[:,0]
                    p = np.reshape(p,(r.batch_size//divs,divs),'F')
                    min_arr = np.minimum(np.amin(p,axis=0),min_arr)
                    for i in range(divs):
                        if min_arr[i] == np.amin(p,axis=0)[i]:
                            ii[i] = np.argmin(p[:,0])
                            jj[i] = ins_location + per*i
                            print(ii,jj)
                            print(min_arr)
                    #max_arr = np.maximum(np.amax(p,axis=0),max_arr)
                quit()



