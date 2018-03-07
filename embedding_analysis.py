import numpy as np
import matplotlib.pyplot as plt
import sys

w2i = np.load('word_to_index.npy').item()
i2w = dict(zip(w2i.values(),w2i.keys()))
i2v = np.load('index_to_vector.npy')
#n = np.linalg.norm(i2v,axis=1)
#n.shape = (10000,1)
#ni2v = i2v / n
if False:
    #v = i2v[w2i['king']] - i2v[w2i['man']]
    #v = v + i2v[w2i['woman']]
    v = i2v[w2i['not']]
    d = ni2v @ v.T
    i = np.argsort(d)[-10:]
    for elem in i:
        print(i2w[elem])

if True:
    m = np.mean(i2v,axis=0)
    print(m.shape)
    plt.hist(m,bins=int(sys.argv[1]))
    plt.show()
