# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes different measures (distances) between marices...
#
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from scipy import stats
from numpy import linalg as LA

print("Going to use a matrix distance measure...")


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# From [Deco2019]: Comparing empirical and simulated FCD.
# We measure KS distance between the upper triangular elements of the empirical and simulated FCD matrices
# (accumulated over all participants).
# ...
# The KS distance quantifies the maximal difference between the cumulative distribution functions of the 2 samples.
class KolmogorovSmirnovStatistic:
    def name(self):
        return "KS"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        # We do not care if they are the same length or not...
        if M1.ndim == 1 or M1.ndim == 2:
            return True
        else:
            return False

    def dist(self, FCD1, FCD2):  # FCD similarity
        d, pvalue = stats.ks_2samp(FCD1.flatten(), FCD2.flatten())
        return d


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class pearsonDissimilarity:
    def name(self):
        return "Pearson"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        if M1.shape != M2.shape:
            return False
        if M1.ndim == 1 or M1.ndim == 2:
            return True

    def pearson_r(self, x, y):
        """Compute Pearson correlation coefficient between two arrays."""
        # Compute correlation matrix
        xflat = x.flatten()
        yflat = y.flatten()
        corr_mat = np.corrcoef(xflat, yflat[:len(xflat)])  # They SHOULD be the same, but... ;-)
        # Return entry [0,1]
        return corr_mat[0,1]

    def dist(self, FC1, FC2):  # FC Disimilarity
        N = FC1.shape[0]
        if FC1.ndim == 1:
            ca = (1-self.pearson_r(FC1, FC2))/2.  # Correlation between both FC
        else:
            Isubdiag = np.tril_indices(N, k=-1)
            ca = (1-self.pearson_r(FC1[Isubdiag], FC2[Isubdiag]))/2.  # Correlation between both FC
        return ca


# --------------------------------------------------------------------------
# Manasij Venkatesh, Joseph Jaja, Luiz Pessoa, Comparing functional connectivity matrices: A geometry-aware
# approach applied to participant identification, NeuroImage, Volume 207, 2020, 116398, ISSN 1053-8119,
# DOI: 10.1016/j.neuroimage.2019.116398.
#
# Code from:
# from https://github.com/makto-toruk/FC_geodesic
# --------------------------------------------------------------------------
class geodesicDistance:
    def __init__(self, eig_thresh=10**(-3)):
        self.FC1 = None
        self.FC2 = None
        self.eig_thresh = eig_thresh

    def name(self):
        return "Geo"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        if M1.shape != M2.shape:
            return False
        if M1.ndim == 2:  # Only for matrices...
            return M1.shape[0] == M1.shape[1]  # They must be same sizes
        else:
            return False

    def geodesic(self):
        '''
        dist = sqrt(trace(log^2(M)))
        M = Q_1^{-1/2}*Q_2*Q_1^{-1/2}
        '''
        # compute Q_1^{-1/2} via eigen value decomposition
        u, s, _ = LA.svd(self.FC1, full_matrices=True)

        ## lift very small eigen values
        for ii, s_ii in enumerate(s):
            if s_ii < self.eig_thresh:
                s[ii] = self.eig_thresh

        '''
        since FC1 is in S+, u = v, u^{-1} = u'
        FC1 = usu^(-1)
        FC1^{1/2} = u[s^{1/2}]u'
        FC1^{-1/2} = u[s^{-1/2}]u'
        '''
        FC1_mod = u @ np.diag(s**(-1/2)) @ np.transpose(u)
        M = FC1_mod @ self.FC2 @ FC1_mod

        '''
        trace = sum of eigenvalues;
        np.logm might have round errors,
        implement using svd instead
        '''
        _, s, _ = LA.svd(M, full_matrices=True)

        return np.sqrt(np.sum(np.log(s)**2))

    def dist(self, Q1, Q2):
        self.FC1 = Q1
        self.FC2 = Q2
        return self.geodesic()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class L1:
    def name(self):
        return "L1"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        return True

    def dist(self, M1, M2):
        # The default value of
        # the ord parameter in numpy.linalg.norm is 2, so change it to 1.
        return np.linalg.norm(M1-M2, ord=1)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class L2:
    def name(self):
        return "L2"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        return True

    def dist(self, M1, M2):
        # This works because the Euclidean distance is the l2 norm, and the default value of
        # the ord parameter in numpy.linalg.norm is 2.
        return np.linalg.norm(M1-M2)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
ERROR_VALUE = 10
dist = None

# @jit(nopython=True)
def distance(FCD1, FCD2):  # FCD similarity, convenience function
    if not (np.isnan(FCD1).any() or np.isnan(FCD2).any()):  # No problems, go ahead!!!
        return dist(FCD1, FCD2)
    else:
        return ERROR_VALUE

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------eof
