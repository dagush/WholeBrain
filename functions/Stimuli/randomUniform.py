# ==========================================================================
# ==========================================================================
# ==========================================================================
# a set of different external stimuli to add to our simulations...
#
# By Gustavo Patow, heavily "inspired" by PyRates
import numpy as np

print("Going to use an external uniform.random (uniform) stimulus...")

onset = 30.0
ampLo = 0.0
ampHi = 1.0
def stimulus(t):
    if t < onset: return 0.  # nothing before the onset
    # we start just at the onset: t-onset is our initial time
    return np.random.uniform(ampLo, ampHi)


# ==========================================================================
# ==========================================================================
# ==========================================================================
