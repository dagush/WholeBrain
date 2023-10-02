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
#  Optimized by Facundo Roffet
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import warnings
import numpy as np
from scipy import signal, stats
from numba import jit

import WholeBrain.Observables.PhaseInteractionMatrix as PhaseInteractionMatrix

print("Going to use Phase Functional Connectivity Dynamics (phFCD)...")

name = 'phFCD'


# ================================= convert the triangular and save if needed
# saveMatrix = False
# save_file = "./Data_Produced/" + name + '.mat'
#
#
# def buildMatrixToSave(linear_phfcd, size):
#     import scipy.io as sio
#     tri = np.zeros((size, size))
#     i_lower = tril_indices_column(size, k=-1)
#     # i_lower = np.tril_indices(size, -1)
#     tri[i_lower] = linear_phfcd
#     tri.T[i_lower] = tri[i_lower]  # make the matrix symmetric
#     sio.savemat(save_file , {name: tri})
#     return tri


# from WholeBrain import BOLDFilters
# BOLDFilters.flp = 0.008
# BOLDFilters.fhi = 0.08


discardOffset = 10  # This was necessary in the old days when, after pre-processing, data had many errors/outliers at
# the beginning and at the end. Thus, the first (and last) 10 samples used to be discarded. Nowadays, this filtering is
# done at the pre-processing stage itself, so this value is set to 0. Thus, depends on your data...


# ==================================================================
# buildFullMatrix: given the output of from_fMRI, this function
# returns the full matrix. Not needed, except for plotting and such...
# ==================================================================
def buildFullMatrix(FCD_data):
    LL = FCD_data.shape[0]
    # T is size of the matrix given the length of the lower/upper triangular part (displaced by 1)
    T = int((1. + np.sqrt(1. + 8. * LL)) / 2.)
    fcd_mat = np.zeros((T, T))
    fcd_mat[np.triu_indices(T, k=1)] = FCD_data
    fcd_mat += fcd_mat.T
    return fcd_mat


# ==================================================================
# tril_indices_column and triu_indices_column:
# retrieve the lower/upper triangular part, but in column-major
# order, needed for compatibility with Matlab code
# ==================================================================
def tril_indices_column(N, k=0):
    row_i, col_i = np.nonzero(
        np.tril(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (col_i,
                row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag


def triu_indices_column(N, k=0):
    row_i, col_i = np.nonzero(
        np.triu(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (col_i,
                row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag


# ==================================================================
# Computes the mean of the matrix
# ==================================================================
@jit(nopython=True)
def mean(x, axis=None):
    if axis == None:
        return np.sum(x, axis) / np.prod(x.shape)
    else:
        return np.sum(x, axis) / x.shape[axis]


# ==================================================================
# numba_phFCD: convenience function to accelerate computations
# ==================================================================
@jit(nopython=True)
def numba_phFCD(phIntMatr_upTri, npattmax, size_kk3):
    phfcd = np.zeros((size_kk3))
    kk3 = 0

    for t in range(npattmax - 2):
        p1_sum = np.sum(phIntMatr_upTri[t:t + 3, :], axis=0)
        p1_norm = np.linalg.norm(p1_sum)
        for t2 in range(t + 1, npattmax - 2):
            p2_sum = np.sum(phIntMatr_upTri[t2:t2 + 3, :], axis=0)
            p2_norm = np.linalg.norm(p2_sum)

            dot_product = np.dot(p1_sum, p2_sum)
            phfcd[kk3] = dot_product / (p1_norm * p2_norm)
            kk3 += 1
    return phfcd


# ==================================================================
# From [Deco2019]: Comparing empirical and simulated FCD.
# For a single subject session where M time points were collected, the corresponding phase-coherence based
# FCD matrix is defined as a MxM symmetric matrix whose (t1, t2) entry is defined by the cosine similarity
# between the upper triangular parts of the 2 matrices dFC(t1) and dFC(t2) (previously defined; see above).
# For 2 vectors p1 and p2, the cosine similarity is given by (p1.p2)/(||p1||||p2||).
# Epochs of stable FC(t) configurations are reflected around the FCD diagonal in blocks of elevated
# inter-FC(t) correlations.
# ==================================================================
def from_fMRI(ts, applyFilters=True, removeStrongArtefacts=True):  # Compute the FCD of an input BOLD signal
    PhaseInteractionMatrix.discardOffset = discardOffset
    phIntMatr = PhaseInteractionMatrix.from_fMRI(ts, applyFilters=applyFilters,
                                                 removeStrongArtefacts=removeStrongArtefacts)  # Compute the Phase-Interaction Matrix
    if not np.isnan(phIntMatr).any():  # No problems, go ahead!!!
        (N, Tmax) = ts.shape
        npattmax = Tmax - (2 * discardOffset - 1)  # calculates the size of phfcd vector
        size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...
        Isubdiag = tril_indices_column(N, k=-1)  # Indices of triangular lower part of matrix
        phIntMatr_upTri = np.zeros((npattmax, int(N * (N - 1) / 2)))  # The int() is not needed, but... (see above)
        for t in range(npattmax):
            phIntMatr_upTri[t,:] = phIntMatr[t][Isubdiag]
        phfcd = numba_phFCD(phIntMatr_upTri, npattmax, size_kk3,)
    else:
        warnings.warn('############ Warning!!! phFCD.from_fMRI: NAN found ############')
        phfcd = np.array([np.nan])
    # if saveMatrix:
    #     buildMatrixToSave(phfcd, npattmax - 2)
    return phfcd


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# ==================================================================
# This code is DEPRECATED (kept for backwards compatibility) - DO NOT USE !!!
# ==================================================================
ERROR_VALUE = 10

# From [Deco2019]: Comparing empirical and simulated FCD.
# We measure KS distance between the upper triangular elements of the empirical and simulated FCD matrices
# (accumulated over all participants).
# ...
# The KS distance quantifies the maximal difference between the cumulative distribution functions of the 2 samples.
def KolmogorovSmirnovStatistic(FCD1, FCD2):  # FCD similarity
    d, pvalue = stats.ks_2samp(FCD1.flatten(), FCD2.flatten())
    return d


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


# --------------------------------------------------------------------------------------
# Test code
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    import scipy.io as sio
    import matplotlib.pyplot as plt

    from Observables import BOLDFilters
    BOLDFilters.flp = 0.008
    BOLDFilters.fhi = 0.08
    BOLDFilters.TR = 3

    inFilePath = "../../Data_Raw/"
    allData = sio.loadmat(inFilePath + 'all_SC_FC_TC_76_90_116.mat')
    sc90 = allData['sc90']
    C = sc90 / np.max(sc90[:]) * 0.2  # Normalization...
    ts90 = allData['tc90symm_s0004']

    # plt.plot(ts90)
    # plt.show()

    discardOffset = 0
    fcd = from_fMRI(ts90)
    full_fcd = buildFullMatrix(fcd)
    plt.imshow(full_fcd)
    plt.show()

    print('test done!')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
