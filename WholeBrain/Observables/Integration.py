# --------------------------------------------------------------------------------------
# Full pipeline for integration analysis
#
# based on:
# [DecoEtAl2015] Deco, G., Tononi, G., Boly, M. et al. Rethinking segregation and integration: contributions
# of whole-brain modelling. Nat Rev Neurosci 16, 430–439 (2015). https://doi.org/10.1038/nrn3963
#
# By Gorka Zamora
# Refactored by Gustavo Patow
#
# --------------------------------------------------------------------------------------
from Observables import BOLDFilters

print("Going to use Integration (from FC)...")

name = 'integration'


"""In this script I want to write a function to compute the metric of
'integration' that Gustavo (Deco) defined some time ago, based on the sizes of
the largest components of the graphs consecutively thresholding a FC matrix.
"""

# Standard library imports
from timeit import default_timer as timer
# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import warnings
from galib.metrics_numba import FloydWarshall_Numba
from galib import ConnectedComponents
# My own libraries and packages


def IntegrationFromFC(fcmatrix, nbins=50, datarange=[0,1]):
    """Calculates integration from a correlation(-like) matrix.

    Given a functional connectivity (FC) matrix, e.g. a matrix of pair-wise
    Pearson correlations, this function does the following:
    1) It creates a series of binary graphs by thresholding the FC matrix along
    several thresholds, from 0 to 1.
    2) It finds the giant component of each binarised graphs, at each threshold.
    3) Computes 'integration' as the area under-the-curve for the curve defined
    by the sizes of the largest components along the thresholds. See,
    "Deco, G., Tononi, G., Boly, M. et al. Rethinking segregation and
    integration: contributions of whole-brain modelling. Nat Rev Neurosci 16,
    430–439 (2015). https://doi.org/10.1038/nrn3963"

    Parameters
    ----------
    fcmatrix : ndarray of rank-2.
        A pair-wise functional connectivity matrix, e.g., Pearson correlation or
        a level of pair-wise synchrony between brain regions.
    nbins : integer (optional)
        Number of bins for which the distribution of values in the matrix shall
        be estimated.
    datarange : list, tuple or array_like (optional)
        A sequence of length = 2 containing the smallest and the largest values
        expected in the statistical association matrix.

    Returns
    -------
    fcintegration : scalar
        The integration value, ranging from 0 to 1.

    """
    # 0) Security checks
    wmin, wmax = datarange[0], datarange[1]
    if len(np.shape(fcmatrix)) != 2:
        raise ValueError( "Input data not a matrix. Data not alligned." )
    if fcmatrix.min() < wmin:
        raise ValueError( "Input data not in range. Values smaller than range found." )
    if fcmatrix.max() > wmax:
        raise ValueError( "Input data not in range. Values larger than range found." )

    N = len(fcmatrix)
    diagidx = np.diag_indices(N)

    stepsize = (wmax - wmin) / nbins
    threslist = np.arange(wmin,wmax,stepsize)

    # 1) Iterate over all thresholds and find the giant component for each case
    giantsizelist = np.zeros(nbins, np.float64)
    for i, thres in enumerate(threslist):
        # Threshold the FC matrix
        bimatrix = np.where(fcmatrix > thres, 1, 0).astype(np.uint8)
        bimatrix[diagidx] = 0
        # Calculate the pair-wise graph distance matrix
        dij = FloydWarshall_Numba(bimatrix)
        # Find the connected components and get the size of the largest one
        components = ConnectedComponents(dij, directed=False, showall=False)
        # The giant component is the whole network
        if len(components) == 1:
            giantsizelist[i] = N
        # In between cases
        elif len(components) > 1:
            maxsize = 0
            for comp in components:
                if len(comp) > maxsize:
                    maxsize = len(comp)
            giantsizelist[i] = maxsize
        # All nodes are isolated
        elif not components:
            giantsizelist[i] = 0

        #print(i, thres, giantsizelist[i])
    # print( giantsizelist )

    # 2) Calculate the integration (area-under-the-curve)
    fcintegration = giantsizelist.sum() * stepsize
    # Normalise to a value from 0 to 1
    fcintegration /= N

    return fcintegration


def IntegrationFromFC_Fast(fcmatrix, nbins=50, datarange=[0,1]):
    """Calculates integration from a correlation(-like) matrix.

    FASTER VERSION. In this case I start the thresholding from the higher values.
    The moment the algorithm finds a threshold with a unique giant component,
    then, for any lower thresholds there is no need to repeat the calculation
    because all subsequent cases we will find a unique component.

    Given a functional connectivity (FC) matrix, e.g. a matrix of pair-wise
    Pearson correlations, this function does the following:
    1) It creates a series of binary graphs by thresholding the FC matrix along
    several thresholds, from 0 to 1.
    2) It finds the giant component of each binarised graphs, at each threshold.
    3) Computes 'integration' as the area under-the-curve for the curve defined
    by the sizes of the largest components along the thresholds. See,
    "Deco, G., Tononi, G., Boly, M. et al. Rethinking segregation and
    integration: contributions of whole-brain modelling. Nat Rev Neurosci 16,
    430–439 (2015). https://doi.org/10.1038/nrn3963"

    Parameters
    ----------
    fcmatrix : ndarray of rank-2.
        A pair-wise functional connectivity matrix, e.g., Pearson correlation or
        a level of pair-wise synchrony between brain regions.
    nbins : integer (optional)
        Number of bins for which the distribution of values in the matrix shall
        be estimated.
    datarange : list, tuple or array_like (optional)
        A sequence of length = 2 containing the smallest and the largest values
        expected in the statistical association matrix.

    Returns
    -------
    fcintegration : scalar
        The integration value, ranging from 0 to 1.

    """
    # 0) Security checks
    wmin, wmax = datarange[0], datarange[1]
    if len(np.shape(fcmatrix)) != 2:
        raise ValueError( "Input data not a matrix. Data not alligned." )
    if fcmatrix.min() < wmin:
        raise ValueError( "Input data not in range. Values smaller than range found." )
    if fcmatrix.max() > wmax:
        raise ValueError( "Input data not in range. Values larger than range found." )

    N = len(fcmatrix)
    diagidx = np.diag_indices(N)

    stepsize = (wmax - wmin) / nbins
    threslist = np.arange(wmin,wmax,stepsize)

    # 1) Iterate over all thresholds and find the giant component for each case
    giantsizelist = np.zeros(nbins, np.float64)
    # Start thresholding from the largest values
    for i, thres in enumerate(threslist[::-1]):
        # Threshold the FC matrix
        bimatrix = np.where(fcmatrix > thres, 1,0).astype(np.uint8)
        bimatrix[diagidx] = 0
        # Calculate the pair-wise graph distance matrix
        dij = FloydWarshall_Numba(bimatrix)
        # Find the connected components and get the size of the largest one
        components = ConnectedComponents(dij, directed=False, showall=False)
        # If all nodes are isolated
        if not components:
            giantsizelist[i] = 0
        # Look for the size of the largest component
        else:
            maxsize = 0
            for comp in components:
                if len(comp) > maxsize:
                    maxsize = len(comp)
            giantsizelist[i] = maxsize
            # If whole net is one component, no need to check further thresholds.
            if maxsize == N:
                giantsizelist[i:] = N
                break

        #print(i, thres, giantsizelist[i])
    # print( giantsizelist )

    # 2) Calculate the integration (area-under-the-curve)
    fcintegration = giantsizelist.sum() * stepsize
    # Normalise to a value from 0 to 1
    fcintegration /= N

    return fcintegration


# ==================================================================
# Protocol
# ==================================================================
def from_fMRI(signal, applyFilters=True, removeStrongArtefacts=True):  # Compute the Integration of an input BOLD signal
    if not np.isnan(signal).any():  # No problems, go ahead!!!
        if applyFilters:
            signal_filt = BOLDFilters.BandPassFilter(signal, removeStrongArtefacts=removeStrongArtefacts)
            sfiltT = signal_filt.T
        else:
            sfiltT = signal.T
        fcnet = np.corrcoef(sfiltT, rowvar=False)  # Pearson correlation coefficients -> FC
        fcnet = abs(fcnet)
        int = IntegrationFromFC_Fast(fcnet, nbins=100)
        return int
    else:
        warnings.warn('############ Warning!!! integration.from_fMRI: NAN found ############')
        # n = signal.shape[0]
        return np.nan


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# This code is DEPRECATED (kept for backwards compatibility)
# ==================================================================
ERROR_VALUE = 10

def distance(int1, int2):  # FCD similarity, convenience function
    if not (np.isnan(int1).any() or np.isnan(int2)):  # No problems, go ahead!!!
        return np.abs(int1-int2)
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


# ==================================================================
# Original Gorka's script, left here for debug/verification purposes
# ==================================================================
if __name__ == '__main__':
    ####################################################################
    # 0) LOAD THE DATA AND PREPARE FOR THE CALCULATIONS
    parcellation = 'Constellation'    # AAL, Desikan, Constellation

    base_folder = "../../Data_Raw/"
    dataroot = base_folder  # + '/Work/Data/Human/ARCHI/%s/' %parcellation

    avfcnet = np.load(dataroot + 'FC_RawAverage.npy')
    N = len(avfcnet)


    # 1) CHECK THE LIMITING CASES OF THE METRIC. It has to be between 0 and 1
    # 1.1) Perfectly uncorrelated nodes
    print(f'\nTest with a diagonal eye({N}x{N}) matrix: Perfectly uncorrelated nodes')
    fcnet_min = np.eye(N, dtype=np.float64)
    integ_min = IntegrationFromFC_Fast(fcnet_min, nbins=20)

    # 1.2) Perfectly correlated nodes
    print(f'\nTest with a ones({N}x{N}) matrix: Perfectly correlated nodes')
    fcnet_max = np.ones((N,N), dtype=np.float64)
    integ_max = IntegrationFromFC_Fast(fcnet_max, nbins=20)

    print('\nResults of these tests:')
    print('   Test min (diagonal) Integration:', integ_min)
    print('   Test max (ones) Integration:', integ_max)
    print('\n')


    # 2) CALCULATE THE INTEGRATION FROM RESTING STATE FCs. POPULATION AVERAGE
    print(f'\nTest with a real avg matrix: From resting state FCs population average ({N}x{N})')
    avfcnet = abs(s)
    integ_rs = IntegrationFromFC(avfcnet, nbins=100)

    print('Integration (resting-state):', integ_rs)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
