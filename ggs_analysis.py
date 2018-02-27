import numpy as np
import matplotlib.pyplot as plt
import os, sys, re

root_dir = './ggs2_results/'
result_dir = 'diffs/'
prob_dir = 'probs/'
file_list = os.listdir(root_dir+result_dir)

n = 0; N = 0
p = 0; m = 0
P = 0; M = 0
init_probs = []
final_probs = []
delta = []

for file_name in file_list:
    rating = int(re.sub('_|\.npy',' ',file_name).split()[1])
    diff = np.load(root_dir+result_dir+file_name)
    prob = np.load(root_dir+prob_dir+file_name)
    if rating > 5:
        d = np.amin(diff)
    elif rating < 5:
        d = np.amax(diff)
    prob_positive = d + prob[0][0] # this gives p[:,0], probability of negative
    if (rating > 5 and prob_positive < 0.5) or (rating < 5 and prob_positive > 0.5):
        n += 1
        m = m + 1 if rating > 5 else m
        p = p + 1 if rating < 5 else p
        print(rating,prob[0][0],d,prob[0][0]+d)
        #diff.shape = (10000*10,)
        #plt.plot(np.sort(diff),'.')
        #plt.show()
        init_probs.append(prob[0][0])
        final_probs.append(prob_positive)
        delta.append(d)
    P = P + 1 if rating < 5 else P
    M = M + 1 if rating > 5 else M
    N += 1

print('%d/%d = %f'%(n,N,n/N))
print('%d/%d = %f'%(m,M,m/M))
print('%d/%d = %f'%(p,P,p/P))
plt.hist(final_probs)
plt.show()
plt.hist(delta)
plt.show()



