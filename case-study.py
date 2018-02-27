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

root = './aclImdb/test/posneg/'
for filename in os.listdir('./ggs_results/diffs/'):
    rv = rp.review(root + filename[0:-4]+'.txt')
    diff = np.load('./ggs_results/diffs/'+filename)
    prob = np.load('./ggs_results/probs/'+filename)
    print('Filename: ',filename,'Initial Probability: ',prob[0][0])
    if rv.sentiment == 'pos':
        m = np.amin(diff)
    else:
        m = np.amax(diff)
    prob_positive = prob[0,0] + m
    # Following are conditions where one-word replacement worked.
    if prob_positive < 0.5 and rv.sentiment == 'pos':
        continue
    elif prob_positive > 0.5 and rv.sentiment == 'neg':
        continue
    g = tf.Graph()
    tf.reset_default_graph()
    rv.translate(1024,word_to_embedding_index,embedding_index_to_word)
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
            rv.translate(r.max_time,word_to_embedding_index,embedding_index_to_word)
            rv.vec(word_embedding)
            decision, probability = \
                r.infer_rep_dpg(sess,rv,rv.index_vector[0])

            #grad = batch_grad[0][0,0:rv.length,:]
            #W = word_embedding; G = grad
            #D = W @ (G.T)
            #c = np.sum(np.multiply(rv.vector_list,G),axis=1)
            #d = D - c
            actual = np.zeros((10000,rv.length))
            for i in range(10000):
                _,p = r.infer_rep_dpg(sess,rv,i)
                actual[i,:] = p[:,0] - probability[0][0]
                if not (i % 100):
                    print(i,np.amin(actual),np.amax(actual))
            np.save('./cs_results/'+filename[0:-4]+'.npy',actual)
