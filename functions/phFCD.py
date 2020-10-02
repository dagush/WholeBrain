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
import numpy as np
from scipy import signal, stats
# from scipy import stats
from functions import BOLDFilters
from functions.Utils import demean

BOLDFilters.flp = 0.008
BOLDFilters.fhi = 0.08


def adif(a, b):
    if np.abs(a - b) > np.pi:
        c = 2 * np.pi - np.abs(a - b)
    else:
        c = np.abs(a - b)
    return c


def tril_indices_column(N, k=0):
    row_i, col_i = np.nonzero(
        np.tril(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (col_i,
                row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag


# From [Deco2019]: Comparing empirical and simulated FCD.
# We measure KS distance between the upper triangular elements of the empirical and simulated FCD matrices
# (accumulated over all participants). The KS distance quantifies the maximal difference between the cumulative
# distribution functions of the 2 samples.
def KolmogorovSmirnovStatistic(FCD1, FCD2):  # FCD similarity
    d, pvalue = stats.ks_2samp(FCD1.flatten(), FCD2.flatten())
    return d


def distance(FCD1, FCD2):  # FCD similarity, convenience function
    return KolmogorovSmirnovStatistic(FCD1, FCD2)


# From [Deco2019]: Comparing empirical and simulated FCD.
# For a single subject session where M time points were collected, the corresponding phase-coherence based
# FCD matrix is defined as a MxM symmetric matrix whose (t1, t2) entry is defined by the cosine similarity
# between the upper triangular parts of the 2 matrices dFC(t1) and dFC(t2) (previously defined; see above).
# For 2 vectors p1 and p2, the cosine similarity is given by (p1.p2)/(||p1||||p2||).
# Epochs of stable FC(t) configurations are reflected around the FCD diagonal in blocks of elevated
# inter-FC(t) correlations.
def from_fMRI(ts_emp, applyFilters = True):  # Compute the FCD of an input BOLD signal
    (N, Tmax) = ts_emp.shape
    # Data structures we are going to need...
    phases_emp = np.zeros([N, Tmax])
    dFC = np.zeros([N, N - 1])
    pattern = np.zeros([Tmax - 19, int(N * (N - 1) / 2)])  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...
    syncdata = np.zeros(Tmax - 19)

    # Filters seem to be always applied...
    ts_emp_filt = BOLDFilters.BandPassFilter(ts_emp)  # zero phase filter the data
    for n in range(N):
        Xanalytic = signal.hilbert(demean.demean(ts_emp_filt[n, :]))
        phases_emp[n, :] = np.angle(Xanalytic)

    Isubdiag = tril_indices_column(N, k=-1)  # Indices of triangular lower part of matrix
    T = np.arange(10, Tmax - 10 + 1)
    for t in T:
        kudata = np.sum(np.cos(phases_emp[:, t - 1]) + 1j * np.sin(phases_emp[:, t - 1])) / N
        syncdata[t - 10] = abs(kudata)
        for i in range(N):
            for j in range(i):
                dFC[i, j] = np.cos(adif(phases_emp[i, t - 1], phases_emp[j, t - 1]))
        pattern[t - 10, :] = dFC[Isubdiag]

    npattmax = Tmax - 19  # calculates the size of phfcd vector
    size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)  # The int() is not needed, but... (see above)
    phfcd = np.zeros((size_kk3))

    kk3 = 0
    for t in range(npattmax - 2):
        p1 = np.mean(pattern[t:t + 3, :], axis=0)
        for t2 in range(t + 1, npattmax - 2):
            p2 = np.mean(pattern[t2:t2 + 3, :], axis=0)
            phfcd[kk3] = np.dot(p1, p2) / np.linalg.norm(p1) / np.linalg.norm(p2)  # this (phFCD) what I want to get
            kk3 = kk3 + 1

    return phfcd


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
