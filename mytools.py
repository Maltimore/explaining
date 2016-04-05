import numpy


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

