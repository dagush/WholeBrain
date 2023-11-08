# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Utils for integration schemes
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from WholeBrain.Utils.randn2 import randn2
from numba import jit


# Matlab compatible definitions
# --------------------------------------------------------------------------
MatlabCompatibility = False
if MatlabCompatibility:
    def randn(vars,N):
        ra = randn2(vars,N)
        return ra #.reshape(-1)
else:
    from numpy.random import randn


# Clamping values
# --------------------------------------------------------------------------
clamping = True
maxClamp = 1.
minClamp = 0.
@jit(nopython=True)
def doClamping(simVariables):
    if clamping:
        simVariables = np.where(simVariables > maxClamp, maxClamp, simVariables)  # clamp values to 0..1
        simVariables = np.where(simVariables < minClamp, minClamp, simVariables)
    return simVariables
