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
top_k = 10000
restore_name = '1182_2.npy'
continue_flag = False
for name in name_list:
    if name != restore_name and not continue_flag:
        continue
    else:
        continue_flag = True
    test_file = name[:-4]+'.txt'
    rv = rp.review('./aclImdb/test/posneg/' + test_file)
    w = min(200,rv.length)
    rv.translate(rv.length,word_to_embedding_index,embedding_index_to_word)
    rv.vec(word_embedding)
    min_arr = np.array([float('inf')]*rv.length)
    max_arr = np.array([float('-inf')]*rv.length)
    g = tf.Graph()
    with g.as_default():
        global_step_tensor = tf.Variable(0,trainable=False,name='global_step')
        # Create RNN graph
        r = rnn.classifier(
            batch_size = 10000,
            learning_rate = 0.0,
            hidden_size = 16,
            max_time = w,
            embeddings = word_embedding,
            global_step = global_step_tensor
        )
        with tf.Session() as sess:
            tf.train.Saver().restore(\
                sess, './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
            print('Processing ' + test_file)
            ii = [0]*rv.length; jj = [0]*rv.length; 
            if rv.sentiment == 'pos':
                pp = [float('inf')]*rv.length
            else:
                pp = [-float('inf')]*rv.length
            for i in range(rv.length):
                _,p,_ = r.infer_window(sess,rv,i,w)
                p = p[:,0]
                pmin = np.amin(p,axis=0)
                pmax = np.amax(p,axis=0)
                if rv.sentiment == 'pos':
                    ii[i] = np.argmin(p)
                    jj[i] = i
                    pp[i] = pmin
                else:
                    ii[i] = np.argmax(p)
                    jj[i] = i
                    pp[i] = pmax
            np.save('./window/ii/'+name,np.array(ii))
            np.save('./window/jj/'+name,np.array(jj))
            np.save('./window/pp/'+name,np.array(pp))
