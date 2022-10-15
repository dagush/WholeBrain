# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the independent Phase Dynamics (indPhDyn)
#
#  by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from scipy import signal, stats
from WholeBrain import BOLDFilters
from WholeBrain.Utils import demean

print("Going to use independent Phase Dynamics (indPhDyn)...")

name = 'indPhDyn'


ERROR_VALUE = 10
BOLDFilters.flp = 0.008
BOLDFilters.fhi = 0.08


# From [Deco2019]: Comparing empirical and simulated FCD.
# We measure KS distance between the upper triangular elements of the empirical and simulated FCD matrices
# (accumulated over all participants). The KS distance quantifies the maximal difference between the cumulative
# distribution WholeBrain of the 2 samples.
def KolmogorovSmirnovStatistic(iPD1, iPD2):  # FCD similarity
    d, pvalue = stats.ks_2samp(iPD1.flatten(), iPD2.flatten())
    return d


def angles(BOLDSignal, applyFilters=True):
    (N, Tmax) = BOLDSignal.shape
    phases_emp = np.zeros([N, Tmax])
    if applyFilters:
        ts_emp_filt = BOLDFilters.BandPassFilter(BOLDSignal)  # zero phase filter the data
    else:
        ts_emp_filt = BOLDSignal
    for n in range(N):
        Xanalytic = signal.hilbert(demean.demean(ts_emp_filt[n, :]))
        phases_emp[n, :] = np.angle(Xanalytic)
    return phases_emp


def distance(signal_sim, angles_emp):
    if not np.isnan(signal).any():  # No problems, go ahead!!!
        N = len(signal_sim)
        r = np.zeros(N)
        # Filters seem to be always applied...
        angles_sim = angles(signal_sim)
        for n in range(N):
            # Now, let's use the Kolmogorov-Smirnov statistic
            r[n] = KolmogorovSmirnovStatistic(angles_sim[n], angles_emp[n])
        return r
    else:
        return ERROR_VALUE

# For a single subject session with N regions where T time points were collected, the corresponding
# phase-coherence based dynamics are defined as a NxT matrix.
def from_fMRI(BOLDSignal, applyFilters = True):  # Compute the indPhDyn of an input BOLD signal
    if not np.isnan(signal).any():  # No problems, go ahead!!!
        a = angles(BOLDSignal, applyFilters=applyFilters)
        return a
    else:
        return np.nan


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# ==================================================================
def init(S, N):
    values = [np.array([], dtype=np.float64) for _ in range(N)]
    return values


def accumulate(indPhDyns, nsub, BOLDSignal):
    (N, Tmax) = BOLDSignal.shape
    for area in range(N):
        indPhDyns[area] = np.concatenate((indPhDyns[area], BOLDSignal[area]))  # Compute the FCD correlations
    return indPhDyns


def postprocess(indPhDyns):
    return np.array(indPhDyns)  # a simple conversion to be sure it has the right format


def findMinMax(arrayValues):
    return np.min(arrayValues), np.argmin(arrayValues)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
