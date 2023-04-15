# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Segregation of a timeseries signal
#
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import warnings
import numpy as np
import copy as cp
from WholeBrain.Utils.iGraphTools import Array2iGraph

import leidenalg as leiden

import WholeBrain.Observables.PhaseInteractionMatrix as PhaseInteractionMatrix

print("Going to use Segregation...")

name = 'Segregation'

# Useful definitions, from Gorka's code
# wcase = 'Binary'    # Binary, Weighted
Qmethod = 'RB'         # RB, RBER
corrdiagonals = 'noDiags'     # 'Diags', 'noDiags'
nruns = 20
resolparam = 1.0
# savedata = False


# From Gorka's code: "In this script I want to compute the modularity of the time averaged
# FC matrices (phase difference matrices). I want to try the community detection on the weighted
# FC matrices and also in the binary but using the flat modularity function."
def computeSegregation(fcnet):
    N, N = np.shape(fcnet)
    # Remove the diagonal entries -- Optional
    if corrdiagonals == 'noDiags':
        fcnet[np.diag_indices(N)] = 0
    # Normalise the weights such that total weight is always the same (N)
    if fcnet.sum() > 0:
        fcnet = fcnet/fcnet.sum() * N
    # Finally, convert to igraph object
    fcignet = Array2iGraph(fcnet, weighted=True)

    # 2.2) Find the partition
    Qmax = -np.inf
    for re in range(nruns):
        # Usual Newman modularity but accepting a resolution parameter
        if Qmethod == 'RB':
            temppartition = leiden.find_partition(fcignet, leiden.RBConfigurationVertexPartition,
                                                  weights='weight', resolution_parameter=resolparam)
        # The usual Newman modularity
        elif Qmethod == 'Modularity':  # Use in the case of binarized matrices...
            temppartition = leiden.find_partition(fcignet, leiden.ModularityVertexPartition,
                                                  weights='weight')
        # Cost function based on random graphs or matrices.
        elif Qmethod == 'RBER': # Use for weighted matrices...
            temppartition = leiden.find_partition(fcignet, leiden.RBERVertexPartition,
                                                  weights='weight', resolution_parameter=resolparam)
        Qtemp = temppartition.quality()
        if Qtemp >= Qmax:
            Qmax = Qtemp
            partition = cp.copy(temppartition)

    if np.isnan(Qtemp):
        Qmax = 0.0
        partition = cp.copy(temppartition)
    return Qmax, partition


def from_fMRI(ts, applyFilters=True, removeStrongArtefacts=True):  # Compute the Metastability of an input BOLD signal
    (N, Tmax) = ts.shape
    # npattmax = Tmax - 19  # calculates the size of phfcd vector
    # size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...

    pIM = PhaseInteractionMatrix.from_fMRI(ts, applyFilters=applyFilters, removeStrongArtefacts=removeStrongArtefacts)  # Compute the Phase-Interaction Matrix

    if not np.isnan(ts).any():  # No problems, go ahead!!!
        # Data structures we are going to need...
        avgFC = np.mean(pIM, axis=0)  # take a TEMPORAL average of all phase matrices...
        avgFC = np.abs(avgFC)  # leiden, and thus segregation, needs a non-negative matrix...
        Qmax, partition = computeSegregation(avgFC)
        integr = Qmax
    else:
        warnings.warn(f'############ Warning!!! Segregation.from_fMRI: NAN found ############')
        integr = np.nan
    return integr


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# This code is DEPRECATED (kept for backwards compatibility)
# ==================================================================
ERROR_VALUE = 10
def distance(K1, K2):  # similarity, convenience function
    if not (np.isnan(K1).any() or np.isnan(K2)):  # No problems, go ahead!!!
        return np.abs(K1-K2)
    else:
        return ERROR_VALUE


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
