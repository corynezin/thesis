# Name: Cory Nezin
# Date: 03/30/2018
# Task: Perform an exponential search window attack

import tensorflow as tf
import numpy as np
import review_proc as rp
import preprocess, rnn, word2vec, wa
import plotutil as putil
import argparse, os, sys, random, re
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

word_embedding = np.load('index_to_vector.npy')
word_to_embedding_index = np.load('word_to_index.npy').item()

m = word_to_embedding_index
embedding_index_to_word = dict(zip(m.values(),m.keys()))

k = 32
c = [0]*1024
t = []
f = 0
root_dir = './aclImdb/test/posneg/'
flag = True
for file_name in os.listdir(root_dir):
    g = tf.Graph()
    print('Running attack on: ' + file_name)
    rvo = rp.review(root_dir + file_name)
    rvo.translate(rvo.length, word_to_embedding_index, embedding_index_to_word)
    rvo.vec(word_embedding)
    # Actual Neural Network
    with g.as_default():
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        r = rnn.classifier(
            batch_size = 1,
            learning_rate = 0.0,
            hidden_size = 8,
            max_time = rvo.length,
            embeddings = word_embedding,
            global_step = global_step_tensor)
        with tf.Session() as sess:
            restore_name = './ckpts/gridckpt_8_10/imdb-rnn-e15.ckpt'
            tf.train.Saver().restore(sess,restore_name)
            decision, probability, grad = r.infer_dpg(sess,rvo)
    rnn_sentiment = 'pos' if not decision[0] else 'neg'
    if rnn_sentiment != rvo.sentiment:
        print('Neural net was wrong, continuing...')
        continue
    g = tf.Graph()
    t0 = time.clock()

    window_size = 200
    hidden_size = 16
    restore_name = './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt'
    #ii,jj,pp = wa.win_atk(rvo, window_size, word_embedding, hidden_size, restore_name)
    ii,jj,pp = wa.win_atk_multi(rvo, window_size, word_embedding, \
                                hidden_size, restore_name)
    if ii is None:
        continue
    # Set source args
    if rvo.sentiment == 'pos':
        args = np.argsort(pp)
    else:
        args = np.flip(np.argsort(pp),axis=0)
    minm = float('inf')
    R = 1
    L = 0
    val_found = False
    print('Original review:')
    print(' '.join(rvo.tokens))
    while True:
        rv = rp.review(root_dir + file_name)
        if not val_found:
            R = R * 2
        m = (L + R) // 2 # m initialized as 1
        if m >= rvo.length:
            print('No derivation found, breaking...')
            f = f + 1
            break
        g = tf.Graph()
        rv.translate(rv.length, word_to_embedding_index, embedding_index_to_word)
        rv.vec(word_embedding)
        with g.as_default():
            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            r = rnn.classifier(
                batch_size = 10000,
                learning_rate = 0.0,
                hidden_size = 8,
                max_time = rv.length,
                embeddings = word_embedding,
                global_step = global_step_tensor)
            with tf.Session() as sess:
                restore_name = './ckpts/gridckpt_8_10/imdb-rnn-e15.ckpt'
                tf.train.Saver().restore(sess,restore_name)
                decision, probability, batch_grad = \
                    r.infer_multi(sess,rvo,ii,args[0:m+1])
                if rvo.sentiment == 'pos' and np.any(decision):
                    misclass = True
                elif rvo.sentiment == 'neg' and np.any(decision-1): 
                    misclass = True
                else:
                    misclass = False
                #rnn_sentiment = 'pos' if not np.any(decision) else 'neg'
                if not misclass and val_found:
                    L = m + 1
                elif misclass:
                    print('New detected sentiment:')
                    print(rnn_sentiment)
                    print('Number of changed words:')
                    print(m+1)
                    minm = min(m+1,minm)
                    val_found = True
                    R = m - 1
                if L > R:
                    print('Search complete, minimum value is: ',minm)
                    c[minm] += 1
                    break
    t1 = time.clock()
    print('Time taken',t1-t0)
    t.append(t1-t0)
    np.save('./eswa_multisample/t/'+file_name[:-4] + '.npy',np.array(t))
    np.save('./eswa_multisample/c/'+file_name[:-4] + '.npy',np.array(c))
    if sum(c) % 50 == 0:
        print('histogram:',c)
        print('Fails:',f)
