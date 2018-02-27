import numpy as np
import matplotlib.pyplot as plt
import os, sys, re
import review_proc as rp

V = 10000
num_top = 10

root = './ggs_results/'
for filename in os.listdir(root+'diffs/'):
    rv = rp.review('aclImdb/test/posneg/'+filename[0:-4]+'.txt')
    grad = np.load(root+'grads/'+filename)
    diff = np.load(root+'diffs/'+filename)
    n = np.linalg.norm(grad,axis=1)
    i = np.argsort(n)[-num_top:]
    for index in i:
        top_word = rv.tokens[index]
        print(top_word)
    print('_'*20)
    #n = n[i]; n.shape = (10,1)
    #n = np.tile(n.T,(V,1))
