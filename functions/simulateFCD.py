# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#
#  Computes simulations with the Dynamic Mean Field Model (DMF):
#
#  Code written by Gustavo Deco gustavo.deco@upf.edu 2017
#  Reviewed by Josephine Cruzat and Joana Cabral
#
#  Translated to Python by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from functions import DynamicMeanField
from functions import BOLDHemModel
from functions import FCD


# Set General Model Parameters
# ============================================================================
dtt = 1e-3  # Sampling rate of simulated neuronal activity (seconds)
dt  = 0.1

TR = 2            # Sampling rate of saved simulated BOLD (seconds)
Tmax = 220        # Number of timepoints in each fMRI session
Tmaxneuronal = int((Tmax+10)*(TR/dtt))  # Number of simulated time points

N_windows = len(range(0,190,3))  # This shouldn't be done like this in Python!!!


# ============================================================================
# simulates the neuronal activity + BOLD + FCD for NumSubjects subjects
# ============================================================================
def simulate(NumSubjects, C, we):
    N=C.shape[0]
    cotsampling = np.zeros([NumSubjects, int(N_windows*(N_windows-1)/2)])

    for nsub in range(NumSubjects):
        print('Subject:',nsub)
        DynamicMeanField.initBookkeeping(N, Tmaxneuronal)
        DynamicMeanField.simulate(dt, Tmaxneuronal, C, we)
        neuro_act = DynamicMeanField.curr_rn

        #### BOLD
        # Friston BALLOON-WINDKESSEL MODEL
        BOLDHemModel.dt = dtt
        T = neuro_act.shape[0]*dtt  # Total time in seconds
        n_t = BOLDHemModel.computeRequiredVectorLength(T)
        BOLD_act = np.zeros([n_t,N])
        for nnew in range(N):
            B = BOLDHemModel.Model_Stephan2007(T,neuro_act[:,nnew])
            BOLD_act[:,nnew] = B
        step = int(np.round(TR/dtt))
        bds = BOLD_act[::step, :]

        # Compute the Hilbert phase from BOLD signals
        # and then, the FCD correlations.
        cotsampling[nsub,:] = FCD.FCD(bds.T)

    return cotsampling
