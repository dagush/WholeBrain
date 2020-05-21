#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the sliding-window Functional Connectivity Dynamics (swFCD)
#
#  Translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import numpy as np
# from numba import jit
from scipy import stats

from functions import BOLDFilters


# def mean2(x):
#     y = np.sum(x) / np.size(x)
#     return y
#
# def corr2(a,b):  # 2-D correlation coefficient
#     a = a - mean2(a)
#     b = b - mean2(b)
#
#     r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
#     return r


def calc_length(start, end, step):
    # This fails for a negative step e.g., range(10, 0, -1).
    # From https://stackoverflow.com/questions/31839032/python-how-to-calculate-the-length-of-a-range-without-creating-the-range
    return (end - start - 1) // step + 1


# @jit(nopython=True)
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix
    corr_mat = np.corrcoef(x.flatten(), y.flatten())
    # Return entry [0,1]
    return corr_mat[0,1]


# @jit(nopython=True)
def KolmogorovSmirnovStatistic(FCD1, FCD2):  # FCD similarity
    d, pvalue = stats.ks_2samp(FCD1.flatten(), FCD2.flatten())
    return d


# @jit(nopython=True)
def distance(FCD1, FCD2):  # FCD similarity, convenience function
    return KolmogorovSmirnovStatistic(FCD1, FCD2)


windowSize = 30
windowStep = 3
def from_fMRI(signal, applyFilters = True):  # Compute the FCD of an input BOLD signal
    (N, Tmax) = signal.shape
    signal_filt = BOLDFilters.BandPassFilter(signal)  # Filters seem to be always applied...
    Isubdiag = np.tril_indices(N, k=-1)  # Indices of triangular lower part of matrix

    # For each pair of sliding windows calculate the FC at t and t2 and
    # compute the correlation between the two.
    lastWindow = Tmax - windowSize  # 190 = 220 - 30
    N_windows = calc_length(0, lastWindow, windowStep)  # N_windows = len(np.arange(0, lastWindow, windowStep))
    cotsampling = np.zeros((int(N_windows*(N_windows-1)/2)))
    kk = 0
    ii2 = 0
    for t in range(0, lastWindow, windowStep):
        jj2 = 0
        sfilt = (signal_filt[:, t:t+windowSize+1]).T  # Extracts a (sliding) window between t and t+windowSize (included)
        cc = np.corrcoef(sfilt, rowvar=False)  # Pearson correlation coefficients
        for t2 in range(0, lastWindow, windowStep):
            sfilt2 = (signal_filt[:, t2:t2+windowSize+1]).T  # Extracts a (sliding) window between t2 and t2+windowSize (included)
            cc2 = np.corrcoef(sfilt2, rowvar=False)  # Pearson correlation coefficients
            ca = pearson_r(cc[Isubdiag],cc2[Isubdiag])  # Correlation between both FC
            if jj2 > ii2:  # Only keep the upper triangular part
                cotsampling[kk] = ca
                kk = kk+1
            jj2 = jj2+1
        ii2 = ii2+1

    return cotsampling


# ==================================================================
# Simple generalization functions to abstract distance measures
# ==================================================================
def init(S, N):
    return np.array([], dtype=np.float64)


def accumulate(FCs, nsub, signal):
    FCs = np.concatenate((FCs, signal))  # Compute the FCD correlations
    return FCs


def postprocess(FCs):
    return FCs  # nothing to do here


def findMinMax(arrayValues):
    return np.min(arrayValues), np.argmin(arrayValues)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
