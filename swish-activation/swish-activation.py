import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype=float) #need to change to np.asarray
    sigmoid = 1 / (1 + np.exp(-x))
    return x*sigmoid
