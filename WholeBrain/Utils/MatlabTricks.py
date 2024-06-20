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
# Matlab's corr function. Based on the code from
# https://stackoverflow.com/questions/71563937/pandas-autocorr-returning-different-calcs-then-my-own-autocorrelation-function
def corr(A, B):
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    B = (B - B.mean(axis=0)) / B.std(axis=0)
    correlation = (np.dot(B.T, A) / B.shape[0]).T
    return correlation


# ================================================================================================================
# Matlab's autocorr function. From
# https://stackoverflow.com/questions/61624985/python-use-of-corrcoeff-to-achieve-matlabs-corr-function
def autocorr(x, lags):
    autocorrs = np.ones(lags+1)  # just to initialize autocorr[0] = 1 ;-)
    for lag in range(1, lags+1):
        series = x[lag:]
        series_auto = x[:-lag]
        corr = 0
        var_x1 = 0
        var_x2 = 0
        for j in range(len(series)):
            x1 = series[j] - np.average(series)
            x2 = series_auto[j] - np.average(series_auto)
            corr += x1*x2
            var_x1  += x1**2
            var_x2 += x2**2
        autocorrs[lag] = corr/((var_x1*var_x2) ** 0.5)
    return autocorrs

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF