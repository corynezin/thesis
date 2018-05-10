import numpy as np

t = []
with open('out.txt','r') as f:
    for line in f:
        if len(line) >= 10 and line[:10] == 'Time taken':
            t.append(float(line[11:]))
            if len(t) == 500:
                break

print(t)
            
