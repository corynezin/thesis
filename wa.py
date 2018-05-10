# Name: Cory Nezin
# Date: 03/30/2018
# Task: Perform a window attack on a recurrent neural network

import tensorflow as tf
import numpy as np
import rnn
import review_proc as rp

def win_atk(rv, window_size, word_embedding, hidden_size, restore_name):
    window_size = min(window_size,rv.length)
    g = tf.Graph()
    with g.as_default():
        global_step_tensor = tf.Variable(0,trainable=False,name='global_step')
        r = rnn.classifier(
            batch_size = 10000,
            learning_rate = 0.0,
            hidden_size = hidden_size, 
            max_time = window_size,
            embeddings = word_embedding,
            global_step = global_step_tensor)

        with tf.Session() as sess:
            tf.train.Saver().restore(sess,restore_name)
            print('Running window attack...')
            ii = [0]*rv.length
            jj = [0]*rv.length
            if rv.sentiment == 'pos':
                pp = [float('inf')]*rv.length
            else:
                pp = [-float('inf')]*rv.length

            for i in range(rv.length):
                d,p,g = r.infer_window(sess,rv,i,window_size)
                rnn_sent = 'pos' if not d[rv.index_vector[0,0]] else 'neg'
                if rnn_sent != rv.sentiment:
                    print('RNN sentiment: ',rnn_sent,'Review sentiment: ',rv.sentiment)
                    print('Neural Net was wrong')
                    return None,None,None
                p = p[:,0]
                pmin = np.amin(p,axis=0)
                pmax = np.amax(p,axis=0)
                if rv.sentiment == 'pos':
                    ii[i] = np.argmin(p)
                    pp[i] = pmin
                else:
                    ii[i] = np.argmax(p)
                    pp[i] = pmax
                jj[i] = i
    ii = np.array(ii)
    jj = np.array(jj)
    pp = np.array(pp)
    return(ii,jj,pp)

def win_atk_multi(rv, window_size, word_embedding, hidden_size, restore_name):
    window_size = min(window_size,rv.length)
    g = tf.Graph()
    pp = []
    with g.as_default():
        global_step_tensor = tf.Variable(0,trainable=False,name='global_step')
        r = rnn.classifier(
            batch_size = 10000,
            learning_rate = 0.0,
            hidden_size = hidden_size, 
            max_time = window_size,
            embeddings = word_embedding,
            global_step = global_step_tensor)

        with tf.Session() as sess:
            tf.train.Saver().restore(sess,restore_name)
            print('Running window attack...')
            #ii = [0]*rv.length
            ii = np.zeros((10000,rv.length))
            #jj = [0]*rv.length
            jj = np.zeros((10000,rv.length))
            for i in range(rv.length):
                d,p,g = r.infer_window(sess,rv,i,window_size)
                rnn_sent = 'pos' if not d[rv.index_vector[0,0]] else 'neg'
                if rnn_sent != rv.sentiment:
                    print('RNN sentiment: ',rnn_sent,'Review sentiment: ',rv.sentiment)
                    print('Neural Net was wrong')
                    return None,None,None
                p = p[:,0]
                if rv.sentiment == 'pos':
                    ii[:,i] = np.argsort(p)
                    pp.append( np.amin(p,axis=0) )
                else:
                    ii[:,i] = np.flip(np.argsort(p),axis=0)
                    pp.append( np.amax(p,axis=0) )
                jj[i] = i
    ii = np.array(ii)
    jj = np.array(jj)
    pp = np.array(pp)
    return(ii,jj,pp)

def gaws(rv, window_size, word_embedding, hidden_size, restore_name,grad,k,d):
    rnn_sentiment = 'pos' if not d[0] else 'neg'
    args = np.argsort(grad) #sorted from smallest to largest
    args = np.flip(args,axis=0)
    k = min(k,args.size) # new line
    window_size = min(window_size,rv.length)
    g = tf.Graph()
    with g.as_default():
        global_step_tensor = tf.Variable(0,trainable=False,name='global_step')
        r = rnn.classifier(
            batch_size = 10000,
            learning_rate = 0.0,
            hidden_size = hidden_size, 
            max_time = window_size,
            embeddings = word_embedding,
            global_step = global_step_tensor)

        with tf.Session() as sess:
            tf.train.Saver().restore(sess,restore_name)
            print('Running window attack...')
            ii = [0]*k
            jj = [0]*k
            if rv.sentiment == 'pos':
                pp = [float('inf')]*k
            else:
                pp = [-float('inf')]*k

            for i in range(min(rv.length,k)):
                d,p,_ = r.infer_window(sess,rv,args[i],window_size)
                #rnn_sent = 'pos' if not d[rv.index_vector[0,0]] else 'neg'
                #if rnn_sent != rv.sentiment:
                #   print('RNN sentiment: ',rnn_sent,'Review sentiment: ',rv.sentiment)
                #   print('Neural Net was wrong')
                #   return None,None,None
                p = p[:,0]
                pmin = np.amin(p,axis=0)
                pmax = np.amax(p,axis=0)
                if rv.sentiment == 'pos':
                    ii[i] = np.argmin(p)
                    pp[i] = pmin
                else:
                    ii[i] = np.argmax(p)
                    pp[i] = pmax
                jj[i] = i
    ii = np.array(ii)
    jj = np.array(jj)
    pp = np.array(pp)
    return(ii,jj,pp)

