#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#  Computes the node-averaged estimates of functional connectivity
#  (also called global brain connectivity, or GBC)
#
#  By Gustavo Deco, translated to Python & refactoring by Gustavo Patow
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
import numpy as np
# from numba import jit
from WholeBrain.Observables import BOLDFilters
import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.measures as measures


print("Going to use Global Brain Connectivity (GBC)...")

name = 'GBC'
defaultMeasure = measures.pearsonSimilarity()
accumulator = measures.averagingAccumulator()
# -------------------- Convenience definitions. Should be overriden if the classes above are changed.
distance = defaultMeasure.distance  # FC similarity, convenience function
findMinMax = defaultMeasure.findMinMax
init = accumulator.init
accumulate = accumulator.accumulate


def characterizeConnectivityMatrix(C):
    return FC.characterizeConnectivityMatrix(C)


# @jit(nopython=True)
def from_fMRI(signal, applyFilters=True, removeStrongArtefacts=True):
    return FC.from_fMRI(signal, applyFilters=applyFilters, removeStrongArtefacts=removeStrongArtefacts)
    # This is wrong, it should be:
    # return np.mean(FC.from_fMRI(signal, applyFilters=applyFilters, removeStrongArtefacts=removeStrongArtefacts), 1)
    # but this would imply changing how the accumulators work, and for that I need to find a neat way to create the
    # accumulation array inside the averagingAccumulator... I'll leave it for another time! ;-)

# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# This code is DEPRECATED (kept for backwards compatibility)
# ==================================================================
# @jit(nopython=True)
# def pearson_r(x, y):
#     return FC.pearson_r(x, y)


# # @jit(nopython=True)
# def FC_Similarity(FC1, FC2):  # FC Similarity
#     return FC.FC_Similarity(FC1, FC2)


# @jit(nopython=True)
# def distance(FC1, FC2):  # FC similarity, convenience function
#     return pearson_r(FC1, FC2)


# def init(S, N):
#     return FC.init(S, N)


# def accumulate(FCs, nsub, signal):
#     return FC.accumulate(FCs, nsub, signal)


def postpro(FCs):
    FCemp = accumulator.postprocess(FCs)
    N = FCemp.shape[0]
    FCemp2 = FCemp - np.multiply(FCemp, np.eye(N))
    GBCemp = np.mean(FCemp2,1)
    return GBCemp

postprocess = postpro

# def findMinMax(arrayValues):
#     return FC.findMinMax(arrayValues)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
