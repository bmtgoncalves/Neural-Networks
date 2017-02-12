#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


def linear(z):
    return z


def binary(z):
    return np.where(z > 0, 1, 0)


def relu(z):
    return np.where(z > 0, z, 0)


def sigmoid(z):
    return 1./(1+np.exp(-z))


def tanh(z):
    return np.tanh(z)

z = np.linspace(-6, 6, 100)

plt.style.use('ggplot')

plt.plot(z, linear(z), 'r-')
plt.xlabel('z')
plt.title('Linear activation function')
plt.savefig('figs/linear.png')
plt.close()

plt.plot(z, binary(z), 'r-')
plt.xlabel('z')
plt.title('Binary activation function')
plt.ylim([-0.01, 1.01])
plt.savefig('figs/binary.png')
plt.close()

plt.plot(z, relu(z), 'r-')
plt.xlabel('z')
plt.title('Rectified Linear activation function')
plt.savefig('figs/relu.png')
plt.close()

plt.plot(z, sigmoid(z), 'r-')
plt.xlabel('z')
plt.title('Sigmoid activation function')
plt.savefig('figs/sigmoid.png')
plt.close()

plt.plot(z, tanh(z), 'r-')
plt.xlabel('z')
plt.title('Hyperbolic Tangent activation function')
plt.savefig('figs/tanh.png')
plt.close()