#!/usr/bin/env python

import numpy as np
from activation_simple import sigmoid, sigmoidGradient
from forward_simple import *

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


def backprop(Theta1, Theta2, X, y):
    N = X.shape[0]
    K = Theta2.shape[0]

    J = 0

    Delta2 = np.zeros(Theta2.shape)
    Delta1 = np.zeros(Theta1.shape)

    for i in range(N):
        # Forward propagation, saving intermediate results
        a1 = np.concatenate(([1], X[i]))  # Input layer

        z2 = np.dot(Theta1, a1)
        a2 = np.concatenate(([1], sigmoid(z2)))  # Hidden Layer

        z3 = np.dot(Theta2, a2)
        a3 = sigmoid(z3)  # Output layer

        y0 = one_hot(K, y[i])

        # Cross entropy
        J -= np.dot(y0.T, np.log(a3))+np.dot((1-y0).T, np.log(1-a3))

        # Calculate the weight deltas
        delta_3 = a3-y0
        delta_2 = np.dot(Theta2.T, delta_3)[1:]*sigmoidGradient(z2)

        Delta2 += np.outer(delta_3, a2)
        Delta1 += np.outer(delta_2, a1)

    J /= N

    Theta1[:, 0] = np.zeros(Theta1.shape[0])
    Theta2[:, 0] = np.zeros(Theta2.shape[0])

    Theta1_grad = Delta1/N
    Theta2_grad = Delta2/N

    return [J, Theta1_grad, Theta2_grad]


if __name__ == "__main__":
    Theta1 = init_weights(input_layer_size, hidden_layer_size)
    Theta2 = init_weights(hidden_layer_size, num_labels)

    step = 0
    tol = 1e-3
    J_old = 1/tol
    diff = 1

    while diff > tol:
        J_train, Theta1_grad, Theta2_grad = backprop(Theta1, Theta2, X_train, y_train)

        diff = abs(J_old-J_train)
        J_old = J_train

        step += 1

        if step % 10 == 0:
            pred_train = predict(Theta1, Theta2, X_train)
            pred_test = predict(Theta1, Theta2, X_test)

            J_test, T1_grad, T2_grad = backprop(Theta1, Theta2, X_test, y_test)

            print(step, J_train, J_test, accuracy(pred_train, y_train), accuracy(pred_test, y_test))

        Theta1 -= .5*Theta1_grad
        Theta2 -= .5*Theta2_grad

    pred = predict(Theta1, Theta2, X_test)

    np.save('Theta1.npy', Theta1)
    np.save('Theta2.npy', Theta2)

    print(accuracy(pred, y_test))
