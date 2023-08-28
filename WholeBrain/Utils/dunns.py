# --------------------------------------------------------------------------------------
# Implements Dunn's Index
# https://en.wikipedia.org/wiki/Dunn_index
#
# Translated from Joana Cabral's code:
# https://github.com/juanitacabral/LEiDA
# Translation by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np

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
