import numpy as np

def exponential_decay(x, a, k, c):
    return a * np.exp(-x/k) + c

def linear_flat(x, m,c):
    return x*m + c