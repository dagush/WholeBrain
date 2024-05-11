# --------------------------------------------------------------------------------------
# Implements Dunn's Index
# https://en.wikipedia.org/wiki/Dunn_index
# --------------------------------------------------------------------------------------
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist, squareform


# --------------------------------------------------------------------------------------
# Translated from Joana Cabral's code:
# https://github.com/juanitacabral/LEiDA
# Translation by Gustavo Patow
# --------------------------------------------------------------------------------------
def dunns(clusters_number, distM, ind):
    # Dunn's index for clustering compactness and separation measurement
    # dunns(clusters_number,distM,ind)
    # clusters_number = Number of clusters
    # distM = Dissimilarity matrix
    # ind   = Indexes for each data point aka cluster to which each data point belongs

    # First, inter cluster distance, \delta(C_i, C_j)
    denominator = np.array([])
    for i2 in range(clusters_number):
        indi = (ind == i2)  # parens no needed, but... ;-)
        indj = (ind != i2)
        x = indi
        y = indj
        # let's use some broadcasting to get the submatrix representing all elements of
        # cluster i2 vs all elements NOT in cluster i2
        temp = distM[np.ix_(x,y)]
        denominator = np.concatenate([denominator, temp.flatten()])
    min_delta = np.min(denominator)

    # Now, cluster size or diameter \Delta(C_i)
    neg_obs = np.zeros(distM.shape)
    for ix in range(clusters_number):
        indxs = (ind == ix)
        neg_obs[np.ix_(indxs,indxs)] = 1
    dem = neg_obs * distM
    max_Delta = np.max(dem)  # calculates the maximum distance (the version proposed by Dunn)

    DI = min_delta / max_Delta
    return DI


def dunns_orig(points, labels):
    k = len(np.unique(labels))
    distM_fcd = squareform(pdist(points, metric='euclidean'))
    dunn_score = dunns(k, distM_fcd, labels)
    return dunn_score


# --------------------------------------------------------------------------------------
# dunn_fast
# From https://github.com/PSYMARKER/leida-python/blob/master/pyleida/clustering/_clustering.py
# --------------------------------------------------------------------------------------
def dunn_fast(points, labels):
    """
    Compute the Dunn index.

    Params:
    ----------
    points : ndarray with shape (N_samples,N_features).
        Observations/samples.

    labels : ndarray with shape (N_samples).
        Labels of each observation in 'points'.
    """
    def _delta_fast(ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]

        return np.min(values)

    def _big_delta_fast(ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]
        # values = values[np.nonzero(values)]

        return np.max(values)

    distances = cosine_distances(points)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)]) * 1_000_000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = _delta_fast((labels == ks[k]), (labels == ks[l]), distances)

        big_deltas[k] = _big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas) / np.max(big_deltas)
    return di