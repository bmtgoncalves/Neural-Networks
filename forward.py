#!/usr/bin/env python

import numpy as np
from activation import sigmoid


def forward(Theta, X, active):
    N = X.shape[0]

    # Add the bias column
    X_ = np.concatenate((np.ones((N, 1)), X), 1)

    # Multiply by the weights
    z = np.dot(X_, Theta.T)

    # Apply the activation function
    a = active(z)

    return a


def predict(Theta1, Theta2, X):
    h1 = forward(Theta1, X, sigmoid)
    h2 = forward(Theta2, h1, sigmoid)

    return np.argmax(h2, 1)


def accuracy(y_, y):
    return np.mean((y_ == y.flatten()))*100.


if __name__ == "__main__":
    Theta1 = np.load('input/Theta1.npy')
    Theta2 = np.load('input/Theta2.npy')

    X = np.load('input/X_train.npy')
    y = np.load('input/y_train.npy')

    y_ = predict(Theta1, Theta2, X)

    print(accuracy(y_, y))
