#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the Functional Connectivity (FC)
#
#  Translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import warnings
import numpy as np
# from numba import jit
from WholeBrain.Observables import BOLDFilters
import WholeBrain.Observables.measures as measures

print("Going to use Functional Connectivity (FC)...")

name = 'FC'
defaultMeasure = measures.pearsonSimilarity()
accumulator = measures.averagingAccumulator()
# -------------------- Convenience definitions. Should be overriden if the classes above are changed.
distance = defaultMeasure.distance  # FC similarity, convenience function
findMinMax = defaultMeasure.findMinMax
init = accumulator.init
accumulate = accumulator.accumulate
postprocess = accumulator.postprocess


# @jit(nopython=True)
def from_fMRI(signal, applyFilters=True, removeStrongArtefacts=True):
    if not np.isnan(signal).any():  # No problems, go ahead!!!
        if applyFilters:
            signal_filt = BOLDFilters.BandPassFilter(signal, removeStrongArtefacts=removeStrongArtefacts)
            sfiltT = signal_filt.T
        else:
            sfiltT = signal.T
        cc = np.corrcoef(sfiltT, rowvar=False)  # Pearson correlation coefficients
        return cc
    else:
        warnings.warn('############ Warning!!! FC.from_fMRI: NAN found ############')
        # n = signal.shape[0]
        return np.nan


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# ==================================================================
# ERROR_VALUE = 10

# # @jit(nopython=True)
# def pearson_r(x, y):
#     """Compute Pearson correlation coefficient between two arrays."""
#     # Compute correlation matrix
#     corr_mat = np.corrcoef(x.flatten(), y.flatten())
#     # Return entry [0,1]
#     return corr_mat[0,1]


# # @jit(nopython=True)
# def FC_Similarity(FC1, FC2):  # FC Similarity
#     (N, N2) = FC1.shape  # should be N == N2
#     Isubdiag = np.tril_indices(N, k=-1)
#     ca = pearson_r(FC1[Isubdiag], FC2[Isubdiag])  # Correlation between both FC
#     return ca


# def init(S, N):
#     return np.zeros((S, N, N))
#
#
# def accumulate(FCs, nsub, signal):
#     FCs[nsub] = signal
#     return FCs
#
#
# def postprocess(FCs):
#     return np.squeeze(np.mean(FCs, axis=0))


# def findMinMax(arrayValues):
#     return np.max(arrayValues), np.argmax(arrayValues)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
