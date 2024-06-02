import numpy as np


# ================================================================================================================
# Matlab's corr2 function. Code taken from
# https://stackoverflow.com/questions/29481518/python-equivalent-of-matlab-corr2
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y


def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r


# ================================================================================================================
# Matlab's corr function. From
# https://stackoverflow.com/questions/61624985/python-use-of-corrcoeff-to-achieve-matlabs-corr-function
def corr(A, B):
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    B = (B - B.mean(axis=0)) / B.std(axis=0)
    correlation = (np.dot(B.T, A) / B.shape[0]).T
    return correlation

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF