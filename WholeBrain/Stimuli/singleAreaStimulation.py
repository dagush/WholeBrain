# ==========================================================================
# ==========================================================================
# ==========================================================================
# a set of different external stimuli to add to our simulations...
#
# By Gustavo Patow, heavily "inspired" by TVB
import numpy as np

print("Going to use an external 'Single Area Stimulation' stimulus...")

# N = nothing at this moment....
onset = 30.0
termination = 42.0
amp = 1.0
N = 1
seed = 1
def stimulus(t):
    if t < onset or t > termination: return np.zeros(N)  # nothing before the onset
    # we start just at the onset: t-onset is our initial time
    Istim = np.zeros(N)
    Istim[seed] = amp
    return Istim


# ==========================================================================
# ==========================================================================
# ==========================================================================
