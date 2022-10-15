# ==========================================================================
# ==========================================================================
# ==========================================================================
# a set of different external stimuli to add to our simulations...
#
# By Gustavo Patow, heavily "inspired" by TVB
import numpy as np

print("Going to use an external PulseTrain stimulus...")

# N = nothing at this moment....
onset = 30.0
T = 42.0
tau = 13.0
amp = 1.0
def stimulus(t):
    if t < onset: return np.zeros(N)  # nothing before the onset
    # we start just at the onset: t-onset is our initial time
    return amp if np.mod(t-onset, T) < tau else 0.


# ==========================================================================
# ==========================================================================
# ==========================================================================
