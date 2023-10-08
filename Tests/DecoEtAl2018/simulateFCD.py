# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the simulation for the Dynamic Mean Field Model (DMF)
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
import WholeBrain.Observables.swFCD as FCD
simModel = None

Tmax = 220.        # Number of timepoints in each fMRI session
windowSize = FCD.windowSize
N_windows = len(np.arange(0,Tmax-windowSize,3))  # This shouldn't be done like this in Python!!!


# ============================================================================
# ============================================================================
# ============================================================================
# simulates the neuronal activity + BOLD + FCD for ONE subject
# ============================================================================
def simSingleSubjectFCD():
    simModel.Tmax = Tmax
    bds = simModel.simulateSingleSubject()
    cotsampling = FCD.from_fMRI(bds.T)  # Compute the FCD correlations
    return cotsampling


# ============================================================================
# simulates the neuronal activity + BOLD + FCD for NumSubjects subjects
# ============================================================================
def simulate(NumSubjects):
    #N=C.shape[0]
    cotsampling = np.zeros([NumSubjects, int(N_windows*(N_windows-1)/2)])

    for nsub in range(NumSubjects):
        print('Subject:', nsub)
        # Compute the FCD correlations.
        cotsampling[nsub,:] = simSingleSubjectFCD()

    return cotsampling


# ============================================================================
# ============================================================================
# ============================================================================EOF
