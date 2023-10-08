# --------------------------------------------------------------------------------------
# Full pipeline for Leading Eigenvector (part of the LEiDA framework)
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
from scipy.sparse.linalg import eigs

import Observables.PhaseInteractionMatrix as PhIntMatr

print("Going to use LEigen...")

name = 'LEigen'


def computeLEigen(phIntMatr):
    timePoints = phIntMatr.shape[0]
    V1s = np.zeros((phIntMatr.shape[1], timePoints))
    for t in range(timePoints):
        # Get the leading eigenvector
        val1, V1 = eigs(phIntMatr[t], k=1, which='LM')  # get the Leading Magnitude eval and evect.
        # Make sure the largest component is negative
        if np.mean(V1 > 0) > .5:
            V1 = -V1
        elif np.mean(V1 > 0) == .5 and np.sum(V1[V1 > 0]) > -np.sum(V1[V1 < 0]):
            V1 = -V1
        V1s[:,t] = V1.flatten()
    return V1s


def from_fMRI(nodeSignal, applyFilters=True, removeStrongArtefacts=True):
    phIntMatr = PhIntMatr.from_fMRI(nodeSignal, applyFilters=applyFilters,
                                    removeStrongArtefacts=removeStrongArtefacts)  # Compute the Phase-Interaction Matrix
    if not np.isnan(phIntMatr).any():  # No problems, go ahead!!!
        leadingEVect = computeLEigen(phIntMatr)
    else:
        warnings.warn('############ Warning!!! LEigen.from_fMRI: NAN found ############')
        leadingEVect = np.array([np.nan])
    return leadingEVect


# ============ Debug/Check code
if __name__ == '__main__':
    id = np.array([np.eye(13), np.random.random((13,13))])
    v = computeLEigen(id)
    print(v)


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF