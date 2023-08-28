#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the sliding-window Functional Connectivity Dynamics (swFCD)
#
#  Translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import warnings
import numpy as np
# from numba import jit
from scipy import stats
from Observables import BOLDFilters

print("Going to use Sliding Windows Functional Connectivity Dynamics (swFCD)...")

name = 'swFCD'


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


windowSize = 30
windowStep = 3
def from_fMRI(signal, applyFilters=True, removeStrongArtefacts=True):  # Compute the FCD of an input BOLD signal
    (N, Tmax) = signal.shape
    lastWindow = Tmax - windowSize  # 190 = 220 - 30
    N_windows = calc_length(0, lastWindow, windowStep)  # N_windows = len(np.arange(0, lastWindow, windowStep))

    if not np.isnan(signal).any():  # No problems, go ahead!!!
        if applyFilters:  # Filters seem to be always applied...
            signal_filt = BOLDFilters.BandPassFilter(signal, removeStrongArtefacts=removeStrongArtefacts)  # zero phase filter the data
        else:
            signal_filt = signal
        Isubdiag = np.tril_indices(N, k=-1)  # Indices of triangular lower part of matrix

        # For each pair of sliding windows calculate the FC at t and t2 and
        # compute the correlation between the two.
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
    else:
        warnings.warn('############ Warning!!! swFCD.from_fMRI: NAN found ############')
        return np.nan


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# This code is DEPRECATED (kept for backwards compatibility)
# ==================================================================
ERROR_VALUE = 10

# @jit(nopython=True)
def KolmogorovSmirnovStatistic(FCD1, FCD2):  # FCD similarity
    d, pvalue = stats.ks_2samp(FCD1.flatten(), FCD2.flatten())
    return d


# @jit(nopython=True)
def distance(FCD1, FCD2):  # FCD similarity, convenience function
    if not (np.isnan(FCD1).any() or np.isnan(FCD2).any()):  # No problems, go ahead!!!
        return KolmogorovSmirnovStatistic(FCD1, FCD2)
    else:
        return ERROR_VALUE


def init(S, N):
    return np.array([], dtype=np.float64)


def accumulate(FCDs, nsub, signal):
    FCDs = np.concatenate((FCDs, signal))  # Compute the FCD correlations
    return FCDs


def postprocess(FCDs):
    return FCDs  # nothing to do here


def findMinMax(arrayValues):
    return np.min(arrayValues), np.argmin(arrayValues)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
