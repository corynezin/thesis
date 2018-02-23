# Name: Cory Nezin  
# Date: 01/17/2018
# Task: Perform a greedy semantic attack on a recurrent neural network

import tensorflow as tf
import numpy as np
import preprocess, rnn, word2vec
import matplotlib.pyplot as plt
import plotutil as putil
import argparse, os, sys, random

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

with g.as_default():
    global_step_tensor = tf.Variable(0, trainable = False, name = 'global_step')
    batch_size = 1; learning_rate = 0.0; hidden_size = 16; max_time = 1024; 
    # Create RNN graph
    r = rnn.classifier(
        batch_size = batch_size,
        learning_rate = learning_rate,
        hidden_size = hidden_size,
        max_time = max_time,
        embeddings = word_embedding,
        global_step = global_step_tensor
    )

    with tf.Session() as sess:
        # Load RNN weights
        tf.train.Saver().restore(sess, './gridckpt_16_10/imdb-rnn-e15.ckpt')
        # Open a review from the hold out set
        review = open('./aclImdb/test/posneg/9999_10.txt').read()
        print(review)
        tokens = preprocess.tokenize(review)

        inputs = np.zeros((batch_size,max_time))
        targets = np.array([0,1]); targets.shape = (1,2)
        sequence_length = np.zeros((batch_size))
        sequence_length[0] = len(tokens)

        # convert word to index
        n = len(tokens)

        for index in range(min(max_time,n)):
            inputs[0][index] = word_to_embedding_index.get(tokens[index],0)
        lookup = [embedding_index_to_word.get(inputs[0][i],'UNK') for i in range(n)]

        adv_inputs = inputs
        decision,probability,pos_grad = \
        sess.run([r.decision,r.probability,r.pos_grad],
                 feed_dict = \
                 {r.inputs:adv_inputs,
                 r.targets:targets,
                 r.sequence_length:sequence_length,
                 r.keep_prob:1.0})
        pg = pos_grad[0][0,0:n,:]
        pas = np.abs(np.sum(pg,axis=1))
        index_list = list(np.argsort(pas))
        i = 0
        while not decision[0]:
            index = index_list[i]
            decision,probability,pos_grad = \
            sess.run([r.decision,r.probability,r.pos_grad],
                     feed_dict = \
                     {r.inputs:adv_inputs,
                     r.targets:targets,
                     r.sequence_length:sequence_length,
                     r.keep_prob:1.0})

            emb_idx = int(adv_inputs[0,index])
            emb_val = word_embedding[emb_idx]
            norm_emb_val = emb_val / np.linalg.norm(emb_val)
            
            dm = np.matmul(normalized_word_embedding,norm_emb_val)
            large_ind = np.argsort(dm)[9990:]
            print( large_ind )
            print( [embedding_index_to_word[large_ind[i]] for i in range(10)] )
            print( dm[large_ind] )
            am = large_ind[8]
            w = embedding_index_to_word.get(am,'UNK')
            original = ' '.join(lookup[index-5:index+5])
            new = ' '.join(lookup[index-5:index]) + ' ' + w + ' ' + \
                  ' '.join(lookup[index+1:index+5])
            adv_inputs[0,index] = am
            print(original + ' -> \n' + new)
            print('New Probability: ', probability)
            print('New Decision: ', decision)
            i = i + 1

print(' '.join([embedding_index_to_word[adv_inputs[0,i]] for i in range(n)]))
