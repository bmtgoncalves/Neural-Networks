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

    Thetas = model[0::2]
    activations = model[1::2]

    layers = len(Thetas)

    K = Thetas[-1].shape[0]
    J = 0

    Deltas = []

    for i in range(layers):
        Deltas.append(np.zeros(Thetas[i].shape))

    deltas = [0, 0, 0, 0]

    for i in range(M):
        As = []
        Zs = [0]
        Hs = [X[i]]

        # Forward propagation, saving intermediate results
        As.append(np.concatenate(([1], Hs[0])))  # Input layer

        for l in range(1, layers+1):
            Zs.append(np.dot(Thetas[l-1], As[l-1]))
            Hs.append(activations[l-1].f(Zs[l]))
            As.append(np.concatenate(([1], Hs[l])))

        y0 = one_hot(K, y[i])

        # Cross entropy
        J -= np.dot(y0.T, np.log(Hs[2]))+np.dot((1-y0).T, np.log(1-Hs[2]))

        # Calculate the weight deltas
        deltas[3] = Hs[layers]-y0
        deltas[2] = np.dot(Thetas[1].T, deltas[3])[1:]*activations[1].df(Zs[1])

        Deltas[1] += np.outer(deltas[3], As[1])
        Deltas[0] += np.outer(deltas[2], As[0])

    J /= M

    Thetas[0][:, 0] = np.zeros(Thetas[0].shape[0])
    Thetas[1][:, 0] = np.zeros(Thetas[1].shape[0])

    grads = []

    grads.append(Deltas[0]/M)
    grads.append(Deltas[1]/M)

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
        J_train, Theta1_grad, Theta2_grad = backprop(model, X_train, y_train)

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
