import numpy as np
from numba import jit

@jit(nopython=True)
def isClose(a, b, rtol=1e-05, atol=1e-08,):
    result = np.absolute(a - b) <= (atol + rtol * np.absolute(b))
    return result


@jit(nopython=True)
def isInt(a):
    result = isClose(np.ceil(a), a) or isClose(np.floor(a), a)
    return result


@jit(nopython=True)
def isZero(a):
    result = isClose(a, 0.)
    return result

# ======================EOF
