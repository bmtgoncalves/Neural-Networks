import sys
import numpy as np

data = np.matrix(np.loadtxt("data/Anscombe1.dat"))

X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

M, N = X.shape
X = np.concatenate((np.ones((M, 1)), X), axis=1) #Add x0

alpha = 0.01
epsilon = 0.12

weights = 2*np.random.rand(N+1, 1)*epsilon - epsilon
count = 0

oldJ = 0
err = 1

Js = []

while err > 1e-6:
    Hs = np.dot(X, weights)
    deltas = alpha/M*np.dot(X.T, (Hs-y))

    count += 1
    weights -= deltas

    J = np.sum(np.power(Hs-y, 2.))/(2*M)
    Js.append(J)
    err = np.abs(oldJ-J)
    oldJ = J

    print(count, J, err, weights.flatten())