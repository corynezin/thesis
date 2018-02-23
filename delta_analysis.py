# Name: Cory Nezin  
# Date: 02/08/2018
# Task: Perform a greedy search attack on multiple test files

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
num_top = 10; per_batch = 50;
g = tf.Graph()
test_file_list = os.listdir('./diffs/')
delta_list = []
with g.as_default():
    global_step_tensor = tf.Variable(0, trainable = False, name = 'global_step')
    # Create RNN graph
    r = rnn.classifier(
        batch_size = 1,
        learning_rate = 0.0,
        hidden_size = 16,
        max_time = 1024,
        embeddings = word_embedding,
        global_step = global_step_tensor
    )
    succ = 0
    n = 0
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
        for test_file in test_file_list:
            review_name = test_file[12:-4] + '.txt'
            rv = rp.review('./aclImdb/test/posneg/' + review_name)
            rv.translate(r.max_time,word_to_embedding_index,embedding_index_to_word)
            rv.vec(word_embedding)
            decision, probability, batch_grad = r.infer_dpg(sess,rv)
            rnn_sentiment = 'pos' if not decision[0] else 'neg'
            if rnn_sentiment != rv.sentiment:
                continue
            grad = batch_grad[0][0,0:rv.length,:]
            W = word_embedding; G = grad
            D = W @ (G.T)
            c = np.sum(np.multiply(rv.vector_list,G),axis=1)
            d = D - c
            x = np.load('./diffs/'+test_file)
            x.shape = (100000,1)
            if rv.sentiment == 'pos':
                delta = np.amin(x)
                final = delta + probability[0][0]
                succ = succ + 1 if final < 0.5 else succ
            else:            
                delta = np.amax(x)
                final = delta + probability[0][0]
                succ = succ + 1 if final > 0.5 else succ
            n += 1
            delta_list.append(delta)
            print(succ/n)
            np.save('./grads/'+review_name[0:-4]+'.npy',d)
            np.save('./probs/'+review_name[0:-4]+'.npy',final)
            #print('Original: ',probability[0][0],'Final: ',final)
plt.hist(delta_list)
plt.show()
