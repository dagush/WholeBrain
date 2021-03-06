# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Phase Functional Connectivity Dynamics (phFCD)
#
#  Explained at
#  [Deco2019] Awakening: Predicting external stimulation to force transitions between different brain states
#       Gustavo Deco, Josephine Cruzat, Joana Cabral, Enzo Tagliazucchi, Helmut Laufs,
#       Nikos K. Logothetis, and Morten L. Kringelbach
#       PNAS September 3, 2019 116 (36) 18088-18097; https://doi.org/10.1073/pnas.1905534116
#
#  Translated to Python by Xenia Kobeleva
#  Revised by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import warnings
import numpy as np
from scipy import signal, stats
import functions.Measures.PhaseInteractionMatrix as PhaseInteractionMatrix

print("Going to use Phase Functional Connectivity Dynamics (phFCD)...")

ERROR_VALUE = 10

# From [Deco2019]: Comparing empirical and simulated FCD.
# We measure KS distance between the upper triangular elements of the empirical and simulated FCD matrices
# (accumulated over all participants). The KS distance quantifies the maximal difference between the cumulative
# distribution functions of the 2 samples.
def KolmogorovSmirnovStatistic(FCD1, FCD2):  # FCD similarity
    d, pvalue = stats.ks_2samp(FCD1.flatten(), FCD2.flatten())
    return d


def distance(FCD1, FCD2):  # FCD similarity, convenience function
    if not (np.isnan(FCD1).any() or np.isnan(FCD2).any()):  # No problems, go ahead!!!
        return KolmogorovSmirnovStatistic(FCD1, FCD2)
    else:
        return ERROR_VALUE


# From [Deco2019]: Comparing empirical and simulated FCD.
# For a single subject session where M time points were collected, the corresponding phase-coherence based
# FCD matrix is defined as a MxM symmetric matrix whose (t1, t2) entry is defined by the cosine similarity
# between the upper triangular parts of the 2 matrices dFC(t1) and dFC(t2) (previously defined; see above).
# For 2 vectors p1 and p2, the cosine similarity is given by (p1.p2)/(||p1||||p2||).
# Epochs of stable FC(t) configurations are reflected around the FCD diagonal in blocks of elevated
# inter-FC(t) correlations.
def from_fMRI(ts, applyFilters = True):  # Compute the FCD of an input BOLD signal
    (N, Tmax) = ts.shape
    npattmax = Tmax - 19  # calculates the size of phfcd vector
    size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...

    pattern = PhaseInteractionMatrix.from_fMRI(ts, applyFilters=applyFilters)  # Compute the Phase-Interaction Matrix

    if not np.isnan(pattern).any():  # No problems, go ahead!!!
        phfcd = np.zeros((size_kk3))
        kk3 = 0
        for t in range(npattmax - 2):
            p1 = np.mean(pattern[t:t + 3, :], axis=0)
            for t2 in range(t + 1, npattmax - 2):
                p2 = np.mean(pattern[t2:t2 + 3, :], axis=0)
                phfcd[kk3] = np.dot(p1, p2) / np.linalg.norm(p1) / np.linalg.norm(p2)  # this (phFCD) what I want to get
                kk3 = kk3 + 1
    else:
        warnings.warn('############ Warning!!! phFCD.from_fMRI: NAN found ############')
        phfcd = np.array([np.nan])
    return phfcd


# ==================================================================
# Simple generalization functions to abstract distance measures
# ==================================================================
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
