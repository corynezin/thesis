import numpy as np
import os, sys
import review_proc as rp
import matplotlib.pyplot as plt

init_prob = []
final_prob = []

for filename in os.listdir('./probs/'):
    review_number = filename[0:-4]
    rv = rp.review('./aclImdb/test/posneg/'+review_number+'.txt')
    final_prob_scalar = np.load('./probs/'+review_number+'.npy')
    
    delta_matrix = np.load('./diffs/actual_diff_'+review_number+'.npy')
    if rv.sentiment == 'pos':
        m = np.amin(delta_matrix,axis=0)
        delta = np.amin(m)
        i = np.argmin(m)
    elif rv.sentiment == 'neg':
        m = np.amax(delta_matrix,axis=0)
        delta = np.amax(delta_matrix)
        i = np.argmax(m)
    #if final_prob_scalar < 0.5 and rv.sentiment == 'pos':
    #    print(i)
    if final_prob_scalar > 0.5 and rv.sentiment == 'neg':
        print(m)
    final_prob.append(final_prob_scalar)
    init_prob.append(final_prob_scalar-delta)

#plt.hist(final_prob,bins=50)
#plt.show()
#plt.hist(init_prob,bins=50)
#plt.show()
