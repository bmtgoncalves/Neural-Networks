import sys
import numpy as np

data = np.matrix(np.loadtxt(sys.argv[1]))

alpha = 0.05
M, N = data.shape
N = N-1


X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

X = np.concatenate((np.ones(X.shape), X), axis=1)

epsilon = 0.12

Thetas = 2*np.random.rand(X.shape[1], 1)*epsilon - epsilon
count = 0

oldJ = 0
err = 1

Js = []

while err > 1e-6:
    Hs = np.dot(X, Thetas)
    deltas = alpha/M*np.dot(X.T, (Hs-y))
    #print(count, deltas)
    count += 1
    Thetas -= deltas

    J = np.sum(np.power(Hs-y, 2.))/(2*M)
    Js.append(J)
    err = np.abs(oldJ-J)
    oldJ = J

    print(count, J, err, Thetas.flatten())