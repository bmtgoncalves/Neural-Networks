import sys
import numpy as np
import pandas as pd

data = pd.read_csv('iris.csv', delimiter=',', header=0)

data['y'] = 0
data.loc[data['species'] == 'setosa', 'y'] = 1

X = np.matrix(data[['sepal_length','sepal_width','petal_length','petal_width']])
y = np.matrix(data['y']).T

def scale(data):
    means = np.mean(data, 0)
    stds = np.std(data, 0)

    return (data-means)/stds

def logistic(z):
    return 1./(1+np.exp(-z))

def plot_boundary(Zs, Hs, y):
    data_fit = np.concatenate((Zs, Hs), axis=1)
    data_fit.sort(axis = 0)

    z = np.linspace(Zs.min(), Zs.max(), 100)

    plt.plot(z, logistic(z), 'r-', label='Theory')
    plt.plot(Zs, Hs, 'X', label='empirical')
    plt.plot(Zs, y, '*', label = 'data')
    plt.xlabel('z')
    plt.ylabel('h(z)')
    plt.title('Logistic Regression')
    plt.legend()
    plt.show()

def plot_points(data, features, Thetas, label='y'):
    plt.plot(data[features[0]][data[label]==0], data[features[1]][data[label]==0], '*', label='y=0')
    plt.plot(data[features[0]][data[label]==1], data[features[1]][data[label]==1], '+', label='y=1')

    cols = ['intercept']
    cols.extend(data.columns)

    Th = dict(zip(cols, Thetas))

    plt.plot([0, -Th[features[1]]/Th['intercept']], [-Th[features[0]]/Th['intercept'], 0], 'r-')

    plt.xlim(data[features[0]].min()-0.1, data[features[0]].max()+0.1)
    plt.ylim(data[features[1]].min()-0.1, data[features[1]].max()+0.1)

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.show()

alpha = 0.05
lamb = .1
M, N = X.shape

X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

epsilon = 0.12

Thetas = 2*np.random.rand(X.shape[1], 1)*epsilon - epsilon
count = 0

oldJ = 0
err = 1

Js = []

normalization = np.ones((Thetas.shape[0], 1))
normalization[0, 0] = 0

while err > 1e-6:
    Zs = np.dot(X, Thetas)
    Hs = logistic(Zs)
    deltas = alpha/M*np.dot(X.T, (Hs-y))
    #print(count, deltas)
    count += 1
    Thetas = np.multiply(Thetas, 1-normalization*alpha*lamb/M)-deltas

    J = -1/M*np.dot(y.T, np.log(Hs)) - np.dot(1-y.T, np.log(1-Hs))

    norm = lamb*np.sum(np.power(Thetas, 2))
    J += norm

    Js.append(float(J))
    err = np.abs(oldJ-J)
    oldJ = J

    print(count, J, err, Thetas.flatten())