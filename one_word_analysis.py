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
rv = rp.review('./aclImdb/test/posneg/9999_10.txt')
rv.translate(1024,word_to_embedding_index,embedding_index_to_word)
d = np.load('predicted_diff.npy')
D = np.load('actual_diff.npy')

p = 0.9888916

d.shape = (d.size,1)
D.shape = (D.size,1)

plot = False
if plot:
    fig,ax = plt.subplots()
    ax.plot(d,p + D-0.5,'.',ms=3)
    plt.xlim((-0.2,0.2)); plt.ylim((-0.5,0.5))
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.title('Review sentiment after replacing one word',fontsize=30)
    plt.xlabel('Directional Derivative',fontsize=20)
    plt.ylabel('Positive Sentiment Confidence',fontsize=20)
    plt.show()

D.shape = (10000,114)
f = np.where(p+D < 0.5)
for i in range(f[0].size):
    word_old = rv.word_list[f[1][i]]
    word_new = embedding_index_to_word[f[0][i]]
    print(word_old.ljust(20) + ' --> ' + word_new)

indices = np.arange(f[0].size)
unique_dst, counts = np.unique(f[0], return_counts=True)
i_dst = unique_dst.size
unique_src, counts = np.unique(f[1], return_counts=True)
i_src = unique_src.size

d_dst = dict(zip(list(unique_dst),range(len(unique_dst))))
d_src = dict(zip(list(unique_src),range(len(unique_src))))

g = [d_dst[f[0][i]] for i in range(f[0].size)]
h = [d_src[f[1][i]] for i in range(f[1].size)]

if plot:
    plt.plot(g,h,'.')
    xticks = [embedding_index_to_word[u] for u in unique_dst]
    yticks = [rv.word_list[i] for i in unique_src]
    plt.xticks(range(len(xticks)), xticks,rotation=90)
    plt.yticks(range(len(yticks)), yticks)
    plt.title('Word transposition resulting in classification change', fontsize=30)
    plt.xlabel('New Word',fontsize=20)
    plt.ylabel('Old Word',fontsize=20)
    plt.show()

global_step_tensor = tf.Variable(0, trainable = False, name = 'global_step')
r = rnn.classifier(
    batch_size = 1,
    learning_rate = 0.0,
    hidden_size = 16, 
    max_time = 1024,
    embeddings = word_embedding,
    global_step = global_step_tensor
)
print(f[1])
print(f[0])
print('wow: ',embedding_index_to_word[84])
with tf.Session() as sess:
    tf.train.Saver().restore(sess, './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
    grad,prob = sess.run([r.pos_grad,r.probability],
        feed_dict={
            r.sequence_length:[rv.length],
            r.inputs:rv.index_vector,
            r.targets:np.array([[1,0]]),
            r.keep_prob:1.0
        })

    g = grad[0][0,0:rv.length,:]
    g1 = grad[0][0,16,:]
    g2 = grad[0][0,40,:]
    #n = np.linalg.norm(g,axis=1)
    n = np.sum(np.abs(g),axis=1)
    print(np.sort(n))
    print('L1 norm: ',np.sum(np.abs(g1)))
    print('L1 norm: ',np.sum(np.abs(g2)))
    n = np.linalg.norm(g,axis=1)
    print(np.sort(n))
    print('L2 norm: ',np.linalg.norm(g1))
    print('L2 norm: ',np.linalg.norm(g2))
    plt.plot(n)
    plt.show()




