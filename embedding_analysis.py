import numpy as np
import matplotlib.pyplot as plt

w2i = np.load('word_to_index.npy').item()
i2w = dict(zip(w2i.values(),w2i.keys()))
i2v = np.load('index_to_vector.npy')

if False:
    v = i2v[w2i['you']]
    d = i2v - v
    n = np.linalg.norm(d,axis=1)
    i = np.argmin(n)
    n[i] = float('inf')
    i = np.argmin(n)
    print(i)
    print(i2w[i])

if True:
    m = np.std(i2v,axis=0)
    print(m)
    print(m.shape)
    print(np.mean(m))
     
    plt.hist(m)
    plt.show()
