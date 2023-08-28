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
from Observables import BOLDFilters
import WholeBrain.Observables.FC as FC

print("Going to use Global Brain Connectivity (GBC)...")

name = 'GBC'


def characterizeConnectivityMatrix(C):
    return FC.characterizeConnectivityMatrix(C)


# @jit(nopython=True)
def from_fMRI(signal, applyFilters=True, removeStrongArtefacts=True):
    return FC.from_fMRI(signal, applyFilters=applyFilters, removeStrongArtefacts=removeStrongArtefacts)


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# This code is DEPRECATED (kept for backwards compatibility)
# ==================================================================
# @jit(nopython=True)
def pearson_r(x, y):
    return FC.pearson_r(x, y)


# @jit(nopython=True)
def FC_Similarity(FC1, FC2):  # FC Similarity
    return FC.FC_Similarity(FC1, FC2)


# @jit(nopython=True)
def distance(FC1, FC2):  # FC similarity, convenience function
    return pearson_r(FC1, FC2)


def init(S, N):
    return FC.init(S, N)


def accumulate(FCs, nsub, signal):
    return FC.accumulate(FCs, nsub, signal)


def postprocess(FCs):
    FCemp = FC.postprocess(FCs)
    N = FCemp.shape[0]
    FCemp2 = FCemp - np.multiply(FCemp, np.eye(N))
    GBCemp = np.mean(FCemp2,1)
    return GBCemp


def findMinMax(arrayValues):
    return FC.findMinMax(arrayValues)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
