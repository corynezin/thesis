import numpy as np
import tensorflow as tf
import numpy as np
import review_proc as rp, preprocess, rnn, word2vec
import matplotlib.pyplot as plt
import plotutil as putil
import argparse, os, sys, random, re
from matplotlib.colors import LogNorm

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
rv = rp.review('./aclImdb/test/posneg/9537_1.txt')
rv.translate(1024,word_to_embedding_index,embedding_index_to_word)

global_step_tensor = tf.Variable(0, trainable = False, name = 'global_step')
r = rnn.classifier(
    batch_size = 1,
    learning_rate = 0.0,
    hidden_size = 16, 
    max_time = 1024,
    embeddings = word_embedding,
    global_step = global_step_tensor
)

with tf.Session() as sess:
    tf.train.Saver().restore(sess, './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
    grad,prob = sess.run([r.pos_grad,r.probability],
        feed_dict={
            r.sequence_length:[rv.length],
            r.inputs:rv.index_vector,
            r.targets:np.array([[1,0]]),
            r.keep_prob:1.0
        })

    g = grad[0][0,1:rv.length,:]
    n = np.norm(g,axis=1)
    plot(np.sort(n))
    plt.imshow(g)
    plt.show()
    
