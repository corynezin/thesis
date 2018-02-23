import numpy as np
import matplotlib.pyplot as plt
import os, sys, re

diff_dir = './diffs/'
dir_list = os.listdir(diff_dir)
delta = []
n = 0
for file_name in dir_list:
    name = file_name[12:]
    rating = int(re.sub('_|\.npy',' ',name).split()[1])
    sentiment = 'pos' if rating > 5 else 'neg'
    x = np.load(diff_dir + file_name)
    x.shape = (100000,1)
    print(file_name)
    if sentiment == 'pos':
        #print(sentiment,np.amin(x),np.amax(x))
        delta.append(abs(np.amin(x)))
    else:
        #print(sentiment,np.amax(x),np.amin(x))
        delta.append(abs(np.amax(x)))
    if delta[-1] >= 0.5:
        n += 1

delta.sort()
print(delta)
print(n)
print(len(delta))
