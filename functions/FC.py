#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the Functional Connectivity (FC)
#
#  Translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import numpy as np
from scipy import stats

from functions import BOLDFilters


def characterizeConnectivityMatrix(C):
    return np.max(C), np.min(C), np.average(C), np.std(C), np.max(np.sum(C, axis=0)), np.average(np.sum(C, axis=0))


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix
    corr_mat = np.corrcoef(x.flatten(), y.flatten())
    # Return entry [0,1]
    return corr_mat[0,1]


def FC_from_fMRI(signal, applyFilters = True):
    if applyFilters:
        signal_filt = BOLDFilters.BandPassFilter(signal)
        sfiltT = signal_filt.T
    else:
        sfiltT = signal.T
    cc = np.corrcoef(sfiltT, rowvar=False)  # Pearson correlation coefficients
    return cc


def FC_Similarity(FC1, FC2):  # FC Similarity
    (N, N2) = FC1.shape  # should be N == N2
    Isubdiag = np.tril_indices(N, k=-1)
    ca = pearson_r(FC1[Isubdiag], FC2[Isubdiag])  # Correlation between both FC
    return ca
