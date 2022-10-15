# ==========================================================================
# ==========================================================================
# ==========================================================================
# a set of different external stimuli to add to our simulations...
#
# By Gustavo Patow, heavily "inspired" by PyRates
import numpy as np

print("Going to use a constant external stimulus...")

N = 1
onset = 30.0
amp = 0.0
def stimulus(t):
    if t < onset: return np.zeros(N)  # nothing before the onset
    # we start just at the onset: t-onset is our initial time
    return amp * np.ones(N)


# ==========================================================================
# ==========================================================================
# ==========================================================================
