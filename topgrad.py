import numpy as np
import matplotlib.pyplot as plt
import os, sys, re
import review_proc as rp
import matplotlib.colors as colors

word_to_embedding_index = np.load('word_to_index.npy').item()
w2i = word_to_embedding_index
index_to_word = dict(zip(w2i.values(),w2i.keys()))
root_dir = './ggs2_results/'
result_dir = 'diffs/'
prob_dir = 'probs/'
grad_dir = 'grads/'
file_list = os.listdir(root_dir+result_dir)

n = 0; N = 0
p = 0; m = 0
P = 0; M = 0
init_probs = []
final_probs = []
delta = []
num_arr = np.zeros((11,))
I = []; J = []
c_dist = []
n_dist = []
for file_name in file_list:
    rv = rp.review('./aclImdb/test/posneg/'+file_name[:-4]+'.txt')
    rating = int(re.sub('_|\.npy',' ',file_name).split()[1])
    diff = np.load(root_dir+result_dir+file_name)
    prob = np.load(root_dir+prob_dir+file_name)
    grad = np.load(root_dir+grad_dir+file_name)

    if rating > 5:
        d = np.amin(diff)
        da = diff + prob[0][0]
        #da = np.amin(da,axis=0)
        #c = np.sum(da < 0.5)
        #num_arr[int(c)] += 1
        i,j = np.nonzero(da < 0.5)
    elif rating < 5:
        d = np.amax(diff)
        da = diff + prob[0][0]
        #da = np.amax(da,axis=0)
        #c = np.sum(da > 0.5)
        #num_arr[int(c)] += 1
        i,j = np.nonzero(da > 0.5)
    prob_positive = d + prob[0][0] # this gives p[:,0], probability of negative
    if (rating > 5 and prob_positive < 0.5) or (rating < 5 and prob_positive > 0.5):
        n += 1
        m = m + 1 if rating > 5 else m
        p = p + 1 if rating < 5 else p
        norms = np.linalg.norm(grad,axis=1)
        centroid = norms.dot(np.arange(0,len(norms))) / np.sum(norms) / len(norms)
        centroid = np.argmax(norms) / len(norms)
        c_dist.append(centroid)
        print(centroid)
        sorted_indices = np.argsort(norms)
        top_indices = sorted_indices[-10:]
        for iind,index in enumerate(j):
            j[iind] = word_to_embedding_index.get(rv.tokens[index],0)
        I.extend(list(i))
        J.extend(list(j))
        for ni,index in enumerate(set(i)):
            print(index_to_word[index],end=', ')
            if ni > 5:
                break
        print('')
        init_probs.append(prob[0][0])
        final_probs.append(prob_positive)
        delta.append(d)
    else:
        norms = np.linalg.norm(grad,axis=1)
        centroid = norms.dot(np.arange(0,len(norms))) / np.sum(norms) / len(norms)
        centroid = np.argmax(norms) / len(norms)
        n_dist.append(centroid)
    P = P + 1 if rating < 5 else P
    M = M + 1 if rating > 5 else M
    N += 1

print('%d/%d = %f'%(n,N,n/N))
print('%d/%d = %f'%(m,M,m/M))
print('%d/%d = %f'%(p,P,p/P))
plt.hist(np.array(c_dist),bins=np.linspace(0,1,21),label='Successful 1-word attack')
plt.hist(np.array(n_dist),bins=np.linspace(0,1,21),label='Unsuccessful attack')
plt.xlabel('Fraction of words before sample')
plt.ylabel('Frequency')
plt.show()
plt.hist(np.array(I),bins=500); plt.show()
plt.hist(np.array(J),bins=500); plt.show()
plt.hist(final_probs)
plt.show()
plt.hist(delta)
plt.show()
