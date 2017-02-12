#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.load('input/X_train.npy')

if len(sys.argv) > 1:
    pos = int(sys.argv[1])
else:
    pos = np.random.randint(0, X.shape[0])

width = int(np.sqrt(X.shape[1]))
digit = X[pos, :].reshape((width, width))

plt.imshow(digit, cmap='gray')
plt.savefig('figs/mnist-' + str(pos) + '.png')
plt.close()

plt.plot(range(X.shape[1]), X[pos, :])
plt.savefig('figs/flat-' + str(pos) + '.png')
plt.close()
