# --------------------------------------------------------------------------------------
# Full pipeline for computing the FCD based on the Leading Eigenvector (part of the LEiDA framework)
#
# From:
# [Cabral et al 2017] Cabral, J., Vidaurre, D., Marques, P. et al. Cognitive performance in healthy older adults
# relates to spontaneous switching between states of functional connectivity during rest. Sci Rep 7, 5135 (2017).
# https://doi.org/10.1038/s41598-017-05425-7
#
# Code by Joana Cabral (modified by Gustavo Deco)
# Translated by Gustavo Patow
# --------------------------------------------------------------------------------------
import warnings
import numpy as np

import Observables.LEigen as LEigen

print("Going to use LEigenFCD...")

name = 'LEigenFCD'


def computeLEigenFCD(LEigs):
    (N, Tmax) = LEigs.shape
    size_kk3 = int(Tmax * (Tmax - 1) / 2)  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...
    leigfcd = np.zeros((size_kk3))
    kk3 = 0
    for t in range(Tmax):
        p1 = LEigs[t]
        for t2 in range(t + 1, Tmax - 1):
            p2 = LEigs[t2]
            leigfcd[kk3] = np.dot(p1, p2) / np.linalg.norm(p1) / np.linalg.norm(p2)  # this (LEigFCD) what I want to get
            kk3 += 1
    return leigfcd


def from_fMRI(nodeSignal, applyFilters=True, removeStrongArtefacts=True):
    LEigs = LEigen.from_fMRI(nodeSignal, applyFilters=applyFilters,
                             removeStrongArtefacts=removeStrongArtefacts)  # Compute the Phase-Interaction Matrix
    if not np.isnan(LEigs).any():  # No problems, go ahead!!!
        leadingEVectFCD = computeLEigenFCD(LEigs)
    else:
        warnings.warn('############ Warning!!! LEigenFCD.from_fMRI: NAN found ############')
        leadingEVectFCD = np.array([np.nan])
    return leadingEVectFCD


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF