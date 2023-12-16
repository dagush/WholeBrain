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
# Base class for distance measures
# --------------------------------------------------------------------------
class distMeasure:
    ERROR_VALUE = np.NaN

    def distance(self, A, B):
        if not (np.isnan(A).any() or np.isnan(B).any()):  # No problems, go ahead!!!
            return self._dist(A, B)
        else:
            return distMeasure.ERROR_VALUE

    def _dist(self, A, B):
        raise Exception('undefined dist measure!!!')
    
    def findMinMax(self, arrayValues):
        return np.min(arrayValues), np.argmin(arrayValues)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# From [Deco2019]: Comparing empirical and simulated FCD.
# We measure KS distance between the upper triangular elements of the empirical and simulated FCD matrices
# (accumulated over all participants).
# ...
# The KS distance quantifies the maximal difference between the cumulative distribution functions of the 2 samples.
class KolmogorovSmirnovStatistic(distMeasure):
    def name(self):
        return "KS"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        # We do not care if they are the same length or not...
        if M1.ndim == 1 or M1.ndim == 2:
            return True
        else:
            return False

    def _dist(self, FCD1, FCD2):  # FCD similarity
        d, pvalue = stats.ks_2samp(FCD1.flatten(), FCD2.flatten())
        return d


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This is here only for legacy reasons. Better use the dissimilarity...
# pearsonSimilarity computes the Pearson corr between the two observables
# Pearson correlation takes a value from −1 (perfect negative correlation) to +1 (perfect positive correlation)
# with the value of zero being no correlation between X and Y.
# (from https://www.sciencedirect.com/science/article/abs/pii/B9780128147610000046)
class pearsonSimilarity(distMeasure):
    def name(self):
        return "PearsonSimilarity"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        if M1.shape != M2.shape:
            return False
        if M1.ndim == 1 or M1.ndim == 2:
            return True

    def _pearson_r(self, x, y):
        """Compute Pearson correlation coefficient between two arrays."""
        # Compute correlation matrix
        xflat = x.flatten()
        yflat = y.flatten()
        corr_mat = np.corrcoef(xflat, yflat[:len(xflat)])  # They SHOULD be the same, but... ;-)
        # Return entry [0,1]
        return corr_mat[0,1]

    def _dist(self, FC1, FC2):
        N = FC1.shape[0]
        if FC1.ndim == 1:
            ca = self._pearson_r(FC1, FC2)  # Correlation between both FC
        else:
            Isubdiag = np.tril_indices(N, k=-1)
            ca = self._pearson_r(FC1[Isubdiag], FC2[Isubdiag])  # Correlation between both FC
        return ca

    def findMinMax(self, arrayValues):
        return np.max(arrayValues), np.argmax(arrayValues)


class pearsonDissimilarity(pearsonSimilarity):
    def name(self):
        return "PearsonDissimilarity"

    def _dist(self, FC1, FC2):  # FC Disimilarity
        pearson = super()._dist(FC1,FC2)
        ca = (1-pearson)/2.  # Correlation between both FC
        return ca



# --------------------------------------------------------------------------
# Manasij Venkatesh, Joseph Jaja, Luiz Pessoa, Comparing functional connectivity matrices: A geometry-aware
# approach applied to participant identification, NeuroImage, Volume 207, 2020, 116398, ISSN 1053-8119,
# DOI: 10.1016/j.neuroimage.2019.116398.
#
# Code from:
# from https://github.com/makto-toruk/FC_geodesic
# --------------------------------------------------------------------------
class geodesicDistance(distMeasure):
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

    def _dist(self, Q1, Q2):
        self.FC1 = Q1
        self.FC2 = Q2
        return self.geodesic()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class L1(distMeasure):
    def name(self):
        return "L1"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        return True

    def _dist(self, M1, M2):
        # The default value of the ord parameter in numpy.linalg.norm is 2, so change it to 1.
        return np.linalg.norm(M1-M2, ord=1)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class L2(distMeasure):
    def name(self):
        return "L2"

    def check(self, M1, M2):  # Method to verify the matrices satisfy the dist requirements
        return True

    def _dist(self, M1, M2):
        # This works because the Euclidean distance is the l2 norm, and the default value of
        # the ord parameter in numpy.linalg.norm is 2.
        return np.linalg.norm(M1-M2)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ACCUMULATORS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

class observableAccumulator:
    def init(self, S, N):
        raise Exception('observable accumulator not defined')


class averagingAccumulator(observableAccumulator):
    def init(self, S, N):
        return np.zeros((S, N, N))

    def accumulate(self, FCs, nsub, signal):
        FCs[nsub] = signal
        return FCs

    def postprocess(self, FCs):
        return np.squeeze(np.mean(FCs, axis=0))


class concatenatingAccumulator(observableAccumulator):
    def init(self, S, N):
        return np.array([], dtype=np.float64)

    def accumulate(self, FCDs, nsub, signal):
        FCDs = np.concatenate((FCDs, signal))  # Compute the FCD correlations
        return FCDs

    def postprocess(self, FCDs):
        return FCDs  # nothing to do here

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------eof
