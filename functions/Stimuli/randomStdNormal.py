# ==========================================================================
# ==========================================================================
# ==========================================================================
# a set of different external stimuli to add to our simulations...
#
# By Gustavo Patow, heavily "inspired" by PyRates
import numpy as np

print("Going to use an external uniform.randn (standard normal) stimulus...")

N = 1
onset = 30.0
mu = 0.0
sigma = 1.0
def stimulus(t):
    if t < onset: return np.zeros(N)  # nothing before the onset
    # we start just at the onset: t-onset is our initial time
    return mu + np.random.randn(N) * sigma


# ==========================================================================
# ==========================================================================
# ==========================================================================
