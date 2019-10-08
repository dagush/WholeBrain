#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  Translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import numpy as np
from functions import BOLDFilters


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def corr2(a,b):  # 2-D correlation coefficient
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
    return r

def FCD(signal):  # Compute the FCD of an input BOLD signal
    (N, Tmax) = signal.shape

    subdiag = np.tril(np.ones((N,N)), -1)
    Isubdiag = np.nonzero(subdiag) # Indices of triangular lower part of matrix

    signal_filt = BOLDFilters.BandPassFilter(signal)

    # For each pair of sliding windows calculate the FC at t and t2 and
    # compute the correlation between the two.
    N_windows=len(range(0,190,3))  # This shouldn't be done in Python!!!
    cotsampling=np.zeros([int(N_windows*(N_windows-1)/2)])
    kk = 0
    ii2 = 0
    for t in range(0,190,3):
        jj2 = 0
        sfilt = (signal_filt[:, t:t+31]).T  # Extracts a (sliding) window between t and t+30 (included)
        cc = np.corrcoef(sfilt, rowvar=False)  # Pearson correlation coefficients
        for t2 in range(0,190,3):
            sfilt2 = (signal_filt[:, t2:t2+31]).T  # Extracts a (sliding) window between t2 and t2+30 (included)
            cc2 = np.corrcoef(sfilt2, rowvar=False)  # Pearson correlation coefficients
            ca = corr2(cc[Isubdiag],cc2[Isubdiag])  # Correlation between both FC
            if jj2 > ii2:  # Only keep the upper triangular part
                cotsampling[kk] = ca
                kk = kk+1
            jj2 = jj2+1
        ii2 = ii2+1

    return cotsampling
