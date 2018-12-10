import numpy as np
import matplotlib.pyplot as plt


with open('u.csv', 'r') as f:
    u = f.read()
with open('v.csv', 'r') as f:
    v = f.read()
with open('x.csv', 'r') as f:
    x = f.read()
with open('y.csv', 'r') as f:
    y = f.read()

u = np.asarray([[float(j) for j in i.split(',')[:-1]] for i in u.split('\n')[:-1]])
v = np.asarray([[float(j) for j in i.split(',')[:-1]] for i in v.split('\n')[:-1]])
x = np.asarray([[float(j) for j in i.split(',')[:-1]] for i in x.split('\n')[:-1]])
y = np.asarray([[float(j) for j in i.split(',')[:-1]] for i in y.split('\n')[:-1]])

fig = plt.figure(figsize=(30, 1.5), dpi=300)
plt.streamplot(x, y, u, v, density=0.6)
plt.show()

