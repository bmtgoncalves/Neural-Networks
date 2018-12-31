#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


class Activation(object):
    def f(z):
        pass

    def df(z):
        pass

class Linear(Activation):
    def f(z):
        return z

    def df(z):
        return np.ones(z.shape)

class ReLu(Activation):
    def f(z):
        return np.where(z > 0, z, 0)

    def df(z):
        return np.where(z > 0, 1, 0)

class Sigmoid(Activation):
    def f(z):
        return 1./(1+np.exp(-z))
    
    def df(z):
        h = Sigmoid.f(z)
        return h*(1-h)

class TanH(Activation):
    def f(z):
        return np.tanh(z)

    def df(z):
        return 1-np.power(np.tanh(z), 2.0)

z = np.linspace(-6, 6, 100)

plt.style.use('ggplot')

plt.plot(z, Linear.f(z), 'r-')
plt.plot(z, Linear.df(z), 'b-')
plt.xlabel('z')
plt.title('Linear activation function')
plt.legend(['function', 'gradient'])
plt.savefig('figs/linear.png')
plt.close()

plt.plot(z, ReLu.f(z), 'r-')
plt.plot(z, ReLu.df(z), 'b-')
plt.legend(['function', 'gradient'])
plt.xlabel('z')
plt.title('Rectified Linear activation function')
plt.savefig('figs/relu.png')
plt.close()

plt.plot(z, Sigmoid.f(z), 'r-')
plt.plot(z, Sigmoid.df(z), 'b-')
plt.legend(['function', 'gradient'])
plt.xlabel('z')
plt.title('Sigmoid activation function')
plt.savefig('figs/sigmoid.png')
plt.close()

plt.plot(z, TanH.f(z), 'r-')
plt.plot(z, TanH.df(z), 'b-')
plt.legend(['function', 'gradient'])
plt.xlabel('z')
plt.title('Hyperbolic Tangent activation function')
plt.savefig('figs/tanh.png')
plt.close()
