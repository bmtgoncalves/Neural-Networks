#!/usr/bin/env python

import gzip
import idx2numpy
import numpy as np

image_size = 28*28

# Original files from http://yann.lecun.com/exdb/mnist/
X_train = idx2numpy.convert_from_file(gzip.open('MNIST/train-images-idx3-ubyte.gz')).astype('float32')
y_train = idx2numpy.convert_from_file(gzip.open('MNIST/train-labels-idx1-ubyte.gz'))

X_test = idx2numpy.convert_from_file(gzip.open('MNIST/t10k-images-idx3-ubyte.gz')).astype('float32')
y_test = idx2numpy.convert_from_file(gzip.open('MNIST/t10k-labels-idx1-ubyte.gz'))

X_train = X_train.reshape((X_train.shape[0], image_size))
X_test = X_test.reshape((X_test.shape[0], image_size))

X_train = X_train[:5000, :]
X_test = X_test[:1000, :]
y_train = y_train[:5000]
y_test = y_test[:1000]

# Save numpy arrays
np.save('input/X_train.npy', X_train)
np.save('input/X_test.npy', X_test)
np.save('input/y_train.npy', y_train)
np.save('input/y_test.npy', y_test)
