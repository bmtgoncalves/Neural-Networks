#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('overfit.dat')

plt.plot(data.T[0], data.T[1], 'r-')
plt.plot(data.T[0], data.T[2], 'b-')
plt.ylabel('J')
plt.xlabel('epoch')
plt.legend([r'$X_{train}$', r'$X_{test}$'])
plt.savefig('figs/overfit_J.png')
plt.close()

plt.plot(data.T[0], data.T[4], 'r-')
plt.plot(data.T[0], data.T[3], 'b-')
plt.ylabel('Error')
plt.xlabel('epoch')
plt.legend([r'$X_{train}$', r'$X_{test}$'])
plt.savefig('figs/overfit_Error.png')
plt.close()
