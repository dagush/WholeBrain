# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Integration of a signal
#
#  Code by Ane López González
#  Translated by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# From Ane López González's thesis:
# Integration refers to the capacity of the brain to maintain communication
# between different parts and subnetworks. Here, we employed a metric of
# integration that assesses the connectivity out of the functional connectivity
# matrix, scanning across different scales (Deco et al., 2015a; Deco and
# Kringelbach, 2017; Deco et al., 2018a; Adhikari et al., 2017). More precisely,
# the time averaged phase-interaction matrix, hPi, is scanned through
# all possible thresholds ranging from 0 to 1. At each threshold, the matrix
# is binarised and the size of its largest connected component is identified.
# Integration is then estimated as the integral of the size of the largest connected
# component as a function of the threshold.
import warnings
import numpy as np
from scipy import signal, stats
# from scipy import stats
from WholeBrain import BOLDFilters
from WholeBrain.Utils import demean

import WholeBrain.Observables.PhaseInteractionMatrix as PhaseInteractionMatrix

print("Going to use Integration...")

name = 'Integration'

ERROR_VALUE = 10
BOLDFilters.flp = 0.008
BOLDFilters.fhi = 0.08


def distance(K1, K2):  # FCD similarity, convenience function
    if not (np.isnan(K1).any() or np.isnan(K2)):  # No problems, go ahead!!!
        return np.abs(K1-K2)
    else:
        return ERROR_VALUE


def from_fMRI(ts, applyFilters = True):  # Compute the Integration of an input BOLD signal
    # --------------------------------------------------------------------------
    #   %Integration:
    #   cc = mean(dM,3);
    #   cc = cc-eye(N);
    #   pp = 1;
    #   PR = 0:0.01:0.99;
    #   cs=zeros(1,length(PR));
    #   for p = PR
    #       A = (cc)>p;
    #       [~, csize] = get_components(A);
    #       cs(pp) = max(csize);
    #       pp = pp+1;
    #   end
    #   integ(nsub) = sum(cs)*0.01/N;
    # --------------------------------------------------------------------------
    (N, Tmax) = ts.shape
    # npattmax = Tmax - 19  # calculates the size of phfcd vector
    # size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...

    hPi = PhaseInteractionMatrix.from_fMRI(ts, applyFilters=applyFilters)  # Compute the Phase-Interaction Matrix

    if not np.isnan(ts).any():  # No problems, go ahead!!!
        # Data structures we are going to need...
        cc = np.mean(hPi,2)  # grand-average of the Phase Interaction Matrix
        cc = cc - np.eye(N)
        PR = np.arange(0, 1, 0.01)  # 0:0.01:0.99
        cs = np.zeros(PR.size)
        for pp, p in enumerate(PR):
            A = cc > p  # A is binarized version of cc
            [~, csize] = get_components(A)
            cs[pp] = max(csize)
            pp = pp+1
        integr = sum(cs)*0.01/N
    else:
        warnings.warn(f'############ Warning!!! Integration.from_fMRI: NAN found ############')
        integr = np.nan
    return integr


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# ==================================================================
def init(S, N):
    return np.zeros(S)


def accumulate(Mets, nsub, signal):
    Mets[nsub] = signal
    return Mets


def postprocess(Mets):
    return Mets  # nothing to do here


def findMinMax(arrayValues):
    return np.min(arrayValues), np.argmin(arrayValues)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
