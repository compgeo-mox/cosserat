import numpy as np


def err(x, x_ex, M):
    delta = x - x_ex
    norm = np.sqrt(x_ex @ M @ x_ex)
    return np.sqrt(delta @ M @ delta) / (norm if norm else 1)


def order(error, diam):
    return np.log(error[:-1] / error[1:]) / np.log(diam[:-1] / diam[1:])
