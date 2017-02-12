#!/usr/bin/env python

import numpy as np


def forward(Theta, X, active):
    m = X.shape[0]

    # Add the bias column
    X_ = np.concatenate((np.ones((m, 1)), X), 1)

    # Multiply by the weights
    z = np.dot(X_, Theta.T)

    # Apply the activation function
    a = active(z)

    return a

if __name__ == "__main__":
    Theta1 = np.load('input/Theta1.npy')
    X = np.load('input/X_train.npy')[:10]

    from activation import sigmoid

    active_value = forward(Theta1, X, sigmoid)
