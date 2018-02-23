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
rv = rp.review('./aclImdb/test/posneg/9999_10.txt')
with g.as_default():
    global_step_tensor = tf.Variable(0, trainable = False, name = 'global_step')
    # Create RNN graph
    r = rnn.classifier(
        batch_size = rv.length,
        learning_rate = 0.0,
        hidden_size = 16,
        max_time = 1024,
        embeddings = word_embedding,
        global_step = global_step_tensor
    )
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
        print(rv.tokens)
        rv.translate(r.max_time,word_to_embedding_index,embedding_index_to_word)
        rv.vec(word_embedding)
        decision, probability, batch_grad = r.infer_rep_dpg(sess,rv,rv.index_vector[0])
        rnn_sentiment = 'pos' if not decision[0] else 'neg'
        print('Neural Net Decision: ',rnn_sentiment,' Actual: ',rv.sentiment)
        if rnn_sentiment != rv.sentiment:
            pass
        grad = batch_grad[0][0,0:rv.length,:]
        W = word_embedding; G = grad
        D = W @ (G.T)
        c = np.sum(np.multiply(rv.vector_list,G),axis=1)
        d = D - c
        actual = np.zeros(d.shape)
        n_examples = 5
        for i in range(10000):
            _,p,g = r.infer_rep_dpg(sess,rv,i)
            actual[i,:] = p[:,0] - probability[0][0]
            '''
            I = np.argsort(np.abs(actual[i,:]))
            top = list(I[rv.length-n_examples:]);
            ax = plt.figure().add_subplot(111)
            plt.plot(d[i,:],actual[i,:],'.')
            plt.axis('tight')
            for j in range(n_examples):
                ax.annotate(rv.word_list[top[j]],
                    xytext=(d[i,top[j]],actual[i,top[j]]),
                    xy = (d[i,top[j]],actual[i,top[j]]))
            #plt.xlim((-0.225,0.225)); plt.ylim((-0.225,0.225))
            plt.xlabel('Directional Derivative')
            plt.ylabel('Actual change in objective')
            plt.show()
            '''
            print(i)
        np.save('actual_diff.npy',actual)
