#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


def linear(z):
    return z


def linearGradient(z):
    return np.ones(z.shape)


def relu(z):
    return np.where(z > 0, z, 0)


def reluGradient(z):
    return np.where(z > 0, 1, 0)


def sigmoid(z):
    return 1./(1+np.exp(-z))


def sigmoidGradient(z):
    h = sigmoid(z)
    return h*(1-h)


def tanh(z):
    return np.tanh(z)


def tanhGradient(z):
    return 1-np.power(tanh(z), 2.0)


z = np.linspace(-6, 6, 100)

plt.style.use('ggplot')

plt.plot(z, linear(z), 'r-')
plt.plot(z, linearGradient(z), 'b-')
plt.xlabel('z')
plt.title('Linear activation function')
plt.legend(['function', 'gradient'])
plt.savefig('figs/linear.png')
plt.close()

plt.plot(z, relu(z), 'r-')
plt.plot(z, reluGradient(z), 'b-')
plt.legend(['function', 'gradient'])
plt.xlabel('z')
plt.title('Rectified Linear activation function')
plt.savefig('figs/relu.png')
plt.close()

plt.plot(z, sigmoid(z), 'r-')
plt.plot(z, sigmoidGradient(z), 'b-')
plt.legend(['function', 'gradient'])
plt.xlabel('z')
plt.title('Sigmoid activation function')
plt.savefig('figs/sigmoid.png')
plt.close()

plt.plot(z, tanh(z), 'r-')
plt.plot(z, tanhGradient(z), 'b-')
plt.legend(['function', 'gradient'])
plt.xlabel('z')
plt.title('Hyperbolic Tangent activation function')
plt.savefig('figs/tanh.png')
plt.close()
