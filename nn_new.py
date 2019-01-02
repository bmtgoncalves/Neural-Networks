#!/usr/bin/env python

import numpy as np
from activation import *
from forward import *

np.random.seed(42)

hidden_layer_size = 50
num_labels = 10

X_train = np.load('input/X_train.npy')
X_test = np.load('input/X_test.npy')
y_train = np.load('input/y_train.npy')
y_test = np.load('input/y_test.npy')

input_layer_size = X_train.shape[1]

X_train /= 255.
X_test /= 255.


def init_weights(L_in, L_out):
    epsilon = 0.12

    return 2*np.random.rand(L_out, L_in+1)*epsilon - epsilon


def one_hot(K, pos):
    y0 = np.zeros(K)
    y0[pos] = 1

    return y0


def backprop(model, X, y):
    M = X.shape[0]

    Thetas=[0]
    Thetas.extend(model[0::2])
    activations = [0]
    activations.extend(model[1::2])

    layers = len(Thetas)

    K = Thetas[-1].shape[0]
    J = 0

    Deltas = [0]

    for i in range(1, layers):
        Deltas.append(np.zeros(Thetas[i].shape))

    deltas = [0]*(layers+1)

    for i in range(M):
        As = [0]
        Zs = [0, 0]
        Hs = [0, X[i]]

        # Forward propagation, saving intermediate results
        As.append(np.concatenate(([1], Hs[1])))  # Input layer

        for l in range(2, layers+1):
            Zs.append(np.dot(Thetas[l-1], As[l-1]))
            Hs.append(activations[l-1].f(Zs[l]))
            As.append(np.concatenate(([1], Hs[l])))

        y0 = one_hot(K, y[i])

        # Cross entropy
        J -= np.dot(y0.T, np.log(Hs[-1]))+np.dot((1-y0).T, np.log(1-Hs[-1]))

        deltas[layers] = Hs[layers]-y0

        # Calculate the weight deltas
        for l in range(layers-1, 1, -1):
            deltas[l] = np.dot(Thetas[l].T, deltas[l+1])[1:]*activations[l].df(Zs[l])

        Deltas[2] += np.outer(deltas[3], As[2])
        Deltas[1] += np.outer(deltas[2], As[1])

    J /= M

    grads = []

    grads.append(Deltas[1]/M)
    grads.append(Deltas[2]/M)

    return [J, grads]


if __name__ == "__main__":
    step = 0
    tol = 1e-3
    J_old = 1/tol
    diff = 1

    Thetas = []
    Thetas.append(init_weights(input_layer_size, hidden_layer_size))
    Thetas.append(init_weights(hidden_layer_size, num_labels))

    model = []

    model.append(Thetas[0])
    model.append(Sigmoid)
    model.append(Thetas[1])
    model.append(Sigmoid)

    while diff > tol:
        J_train, grads = backprop(model, X_train, y_train)

        diff = abs(J_old-J_train)
        J_old = J_train

        step += 1

        if step % 10 == 0:
            pred_train = predict(model, X_train)
            pred_test = predict(model, X_test)

            J_test, grads = backprop(model, X_test, y_test)

            print(step, J_train, J_test, accuracy(pred_train, y_train), accuracy(pred_test, y_test))

        Thetas[0] -= .5*grads[0]
        Thetas[1] -= .5*grads[1]

        if step == 30:
            break

    pred = predict(model, X_test)

    #np.save('Theta1.npy', Theta1)
    #np.save('Theta2.npy', Theta2)

    print(accuracy(pred, y_test))
