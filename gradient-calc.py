import tensorflow as tf
import numpy as np
import preprocess, rnn
import matplotlib.pyplot as plt
import plotutil as putil
import os, sys, random, pdb

# Load pre-trained embeddings
word_embedding = np.load('final_embeddings.npy')
word_to_embedding_index = np.load('emb_dict.npy').item()
m = word_to_embedding_index
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
        tf.train.Saver().restore(sess, './ckpts/gridckpt_16_10/imdb-rnn-e15.ckpt')
        review = open('./aclImdb/test/posneg/9999_10.txt').read()
        print(review)
        tokens = preprocess.tokenize(review)

        inputs = np.zeros((batch_size,max_time))
        targets = np.array([0,1]); targets.shape = (1,2)
        sequence_length = np.zeros((batch_size))
        sequence_length[0] = len(tokens)
        
        for index in range(min(max_time,len(tokens))):
            inputs[0][index] = word_to_embedding_index.get(tokens[index],0)

        #print('Review: \n', review)
        #print('Tokens: \n', tokens)
        #print('Inputs: \n', inputs)
        probability, pos_grad, neg_grad, logits, embedded = \
        sess.run([r.probability, r.pos_grad, r.neg_grad, r.logits, r.embed],
                 feed_dict = \
                 {r.inputs:inputs,
                  r.targets:targets,
                  r.sequence_length:sequence_length,
                  r.keep_prob:1.0})
        
        print('Probability: ', probability)
        n = len(tokens)
        pg = pos_grad[0][0,0:n,:]; ps = np.sum(np.abs(pg),axis=1)
        ng = neg_grad[0][0,0:n,:]; ns = np.sum(np.abs(ng),axis=1)
        pas = np.abs(np.sum(pg,axis=1)); nas = np.abs(np.sum(ng,axis=1))
        #print('Salience: ',[ps,ns])
        x = np.linspace(1,n,num=n)
        lookup = [embedding_index_to_word.get(inputs[0][i],'UNK') for i in range(n)]
     
        word_index = [i for i in range(n)]
        random.shuffle(word_index)
        adv_inputs = inputs
        for _ in range(0): 
            for index in word_index:
                decision,probability,pos_grad = \
                sess.run([r.decision,r.probability,r.pos_grad],
                         feed_dict = \
                         {r.inputs:adv_inputs,
                         r.targets:targets,
                         r.sequence_length:sequence_length,
                         r.keep_prob:1.0})
                pg = pos_grad[0][0,0:n,:]; 
                ps = np.sqrt(np.sum(np.abs(pg)**2,axis=1))
                gs = np.sign(pg[index,:])
                emb_idx = int(adv_inputs[0,index])
                emb_val = word_embedding[emb_idx]
                # Careful here: sign deterines direction
                dm = gs + np.sign(word_embedding)
                dv = np.sum(np.abs(dm),axis=1)
                am = int(np.argmin(dv))
                w = embedding_index_to_word.get(am,'UNK')
                original = ' '.join(lookup[index-5:index+5])
                new = ' '.join(lookup[index-5:index]) + ' ' + w + ' ' + \
                      ' '.join(lookup[index+1:index+5])
                adv_inputs[0,index] = am
                print(original + ' -> \n' + new)
                print('New Probability: ', probability)

        start = 0; end = 50
        if int(sys.argv[1]):
            ax = plt.subplot(121)
            plt.suptitle("Saliency Visualization of Excerpt",fontsize=24)
            y = np.linspace(start,end-1,num=end-start)
            plt.yticks(y, lookup[start:end])
            plt.imshow(pg[start:end,:])
            plt.xlabel("Embedding Dimesnion", fontsize=22)
            ax.set_aspect(3.05)
            ax = plt.subplot(122)
            plt.yticks(y.max()-y, lookup)
            putil.vstem(y.max()-y,ps[start:end],'b',label="Cumulative Absolute Gradient")
            putil.vstem(y.max()-y,pas[start:end],'r',label="Cumulative Gradient")
            plt.legend()
            plt.xlabel("Cumulative Saliency", fontsize=22)
            plt.xlim([ps.max(),0]) 
            plt.ylim([-1,y.max()+1])
            plt.grid()
            plt.show()
        if int(sys.argv[2]):
            plt.subplot(211)
            plt.stem(pg[28,:],basefmt='k-',linefmt='k-',markerfmt='ko')
            plt.title("Gradient with respect to '%s'" % lookup[28],fontsize=24)
            plt.grid()
            plt.subplot(212)
            plt.title("Gradient with respect to '%s'" % lookup[31],fontsize=24)
            plt.stem(pg[31,:],basefmt='k-',linefmt='k-',markerfmt='ko')
            plt.grid()
            plt.show()
       
         
