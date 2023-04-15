# ----------------------------------------------------------------------------
# eFC: Edge Functional Connectivity,
#
# Faskowitz, J., Esfahlani, F.Z., Jo, Y. et al.
# Edge-centric functional network representations of human cerebral cortex reveal overlapping system-level architecture.
# Nat Neurosci 23, 1644–1654 (2020). https://doi.org/10.1038/s41593-020-00719-y
#
# Code at https://github.com/brain-networks/edge-centric_demo
#
# Code by Facundo Roffet,
# refactored by Gustavo Patow
# ----------------------------------------------------------------------------
import numpy as np
from scipy.stats import zscore

print("Going to use Edge-centric Functional Connectivity (eFC)...")

name = 'eFC'


def edge_ts(ts):
    # Number of nodes
    n = ts.shape[1]
    # Normalization
    z = zscore(ts)
    # Indexes of the upper triangular matrix
    index = np.nonzero(np.triu(np.ones((n,n)),1))
    u = index[0]
    v = index[1]
    # edge time series
    e_ts = np.multiply(z[:,u], z[:,v])
    return e_ts


def edgets2edgecorr(a):
    b = np.matmul(np.transpose(a),a)
    c = np.sqrt(np.diagonal(b))
    c = np.expand_dims(c,axis=1)
    d = np.matmul(c,np.transpose(c))
    e = np.divide(b,d)
    return e


def from_fMRI(signal, applyFilters = True):
    eTS = edge_ts(signal)
    return edgets2edgecorr(eTS)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
