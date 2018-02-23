import numpy as np
import matplotlib.pyplot as plt
import os, sys, re

root_dir = './ggs_results/'
result_dir = 'diffs/'
prob_dir = 'probs/'
file_list = os.listdir(root_dir+result_dir)

n = 0; N = 0
final_prob = []
init_prob = []
diffs = []

for file_name in file_list:
    rating = int(re.sub('_|\.npy',' ',file_name).split()[1])
    diff = np.load(root_dir+result_dir+file_name)
    prob = np.load(root_dir+prob_dir+file_name)
    if rating > 5:
        m = np.amin(diff)
    elif rating < 5:
        m = np.amax(diff)
    if (rating<5 and prob[0,0]+m>0.5) or (rating>5 and prob[0,0]+m<0.5):
        n += 1
    else:
        print(file_name)
    N += 1
    final_prob.append(prob[0,0]+m)
    init_prob.append(prob[0,0])
    diffs.append(m)

print(n/N)
plt.hist(final_prob)
plt.show()
plt.hist(init_prob)
plt.show()
plt.hist(diffs)
plt.show()




