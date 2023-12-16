# ==========================================================================
# ==========================================================================
# ==========================================================================
#
# Coupling functions
#
# Adapted from TVB: The activity (state-variables) that have been propagated
# over the long-range Connectivity pass through these functions before
# entering the equations (Model > dfun()) describing the local dynamics.
# The state-variable vector for the $k$-th node or region in the network can be expressed as:
# Derivative = Noise + Local dynamics + Coupling(time delays).
#
#
# ==========================================================================
# ==========================================================================
# ==========================================================================
import numpy as np
# from numba import jit
from numba import int32, double    # import the types
from numba.experimental import jitclass


# -----------------------------------------------------------------------------
# ----------------- Instantaneous Direct Coupling -----------------------------
# -----------------------------------------------------------------------------
@jitclass([('SC', double[:, :])])
class instantaneousDirectCoupling:
    def __init__(self):
        self.SC = np.empty((1,1))

    def setParms(self, SC):
        self.SC = SC

    def couple(self, x):
        return self.SC @ x


# -----------------------------------------------------------------------------
# ----------------- Delayed Direct Coupling -----------------------------------
# -----------------------------------------------------------------------------
# Some implementation details:
#
# We cannot use and advanced kind of indexing in NUMBA because:
#   NumbaTypeError: Multi-dimensional indices are not supported, but works well in regular numpy
# So this code will work in Python, but in NUMBA it will NOT WORK:
# result = np.zeros(N)
# time_idx = (self.index - 1 - self.delays + self.horizon) % self.horizon
# delayed = self.history[np.arange(N), time_idx]
# for inode in np.arange(N):
#     result[inode] = np.sum(self.SC[inode, :] * delayed[:, inode])
#
# Now, the next try was to mix computations, but:
#   NumbaTypeError: Using more than one non-scalar array index is unsupported.
# Again, this works well in regular numpy. So this code will NOT WORK:
# result = np.zeros(N)
# delayed = np.zeros((N,N))
# for inode in np.arange(N):
#     time_idx = (self.index - 1 - self.delays[:, inode] + self.horizon) % self.horizon
#     delayed[:, inode] = self.history[np.arange(N), time_idx]
#     result[inode] = np.sum(self.SC[inode, :] * delayed[:, inode])
#
# This part ended up being identical to TVB C++ implementation, see
# https://github.com/neich/tvb-cpp/blob/main/src/tvb-cpp/simulator/coupling.cpp
# here, we simply make the loops explicitl... :-(
#
# Finally, a mathematical considerationL the last part in the main iteration is the same as:
# resultTest = np.diagonal(self.SC @ delayed)
# But I believe this should be slower...
# Finalyy, for testing purposes only, without delays we get
# resultWODelays = self.SC @ x

@jitclass([('SC', double[:, :]),
           ('delays', int32[:, :]),
           ('history', double[:, :]),
           ('index', int32),
           ('horizon', int32),
           ])
class delayedDirectCoupling:
    def __init__(self):
        self.SC = np.empty((1,1))

    def setParms(self, SC, timeDelays, dt):
        self.SC = SC
        # Convert the time delays between regions in physical units into an array of linear
        # indices into the history attribute. Taken from:
        # https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_library/tvb/datatypes/connectivity.py
        self.delays = np.rint(timeDelays / dt).astype(np.int32)
        self.horizon = self.delays.max() + 1
        # Now, create space for the history data...
        self.history = np.zeros((SC.shape[0], self.horizon), dtype=double)
        self.index = 0

    def initConstantPast(self, prev_x):
        for i in range(self.horizon):
            self.history[:, i] = prev_x

    def couple(self, x):
        N = x.size
        self.history[:, self.index] = x
        result = np.zeros(N)
        delayed = np.zeros((N,N))
        for inode in np.arange(N):
            for inode_from in np.arange(N):
                time_idx = (self.index - 1 - self.delays[inode_from, inode] + self.horizon) % self.horizon
                delayed[inode_from, inode] = self.history[inode_from, time_idx]
            result[inode] = np.sum(self.SC[inode, :] * delayed[:, inode])
        # Finally, let's update the index for the next iteration...
        self.index = (self.index + 1) % self.horizon
        return result

    @property
    def hist(self):
        return self.history


# ===========================================================================================================
# Debug code only...
# ===========================================================================================================
if __name__ == '__main__':
    from numba import jit

    SC = np.ones((3,3), dtype=np.float64)
        # np.array([[0.,1.,2.],
        #            [1.,0.,3],
        #            [2.,3.,0.]], dtype=np.float64)
    delays = np.array([[0.,1.,2.],
                       [1.,0.,3],
                       [2.,3.,0.]])/5.  # np.zeros((3,3))  # np.ones((3,3))/2.

    # couplingObj = instantaneousDirectCoupling(SC)
    couplingObj = delayedDirectCoupling(SC, delays, 0.1)

    # Uncomment this to test how it works inside a jitted function...
    # @jit(nopython=True)
    # def test(a, b, coupling):
    #     # coupling = instantaneousDirectCoupling(SC)
    #     couple = coupling.couple(b)
    #     return a * couple

    for i in np.arange(10):
        coupl = couplingObj.couple(np.ones(3).astype(np.float64)*i)
        # res = test(2., np.ones(3).astype(np.float64)*i, couplingObj)  # This is for the jit function call
        print(f'\n------\ni: {i}\ncoupl:\n{coupl}')
        # print(f'\n------\nres: {res}')
        print(f'History:\n{couplingObj.hist}')
    print('Done!')

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF