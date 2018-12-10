import numpy as np

def tolist(L):
    L = [i for i in L.split('\n')[:-1]]
    for i in range(len(L)):
        L[i] = [float(j) for j in L[i].split(',')[:-1]]
    return np.asarray(L)

with open('u.csv', 'r') as f:
    u = tolist(f.read())
with open('v.csv', 'r') as f:
    v = tolist(f.read())
with open('x.csv', 'r') as f:
    x = tolist(f.read())
with open('y.csv', 'r') as f:
    y = tolist(f.read())


import matplotlib.pyplot as plt

plt.streamplot(x, y, u, v, density=2.0)
plt.show()
