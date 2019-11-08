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
from functions import Integrator_EulerMaruyama as integrator
from functions import BOLDHemModel_Stephan2007 as BOLDModel
# from functions import BOLDHemModel_Stephan2008 as BOLDModel
from functions import FCD

print("Going to use Functional Connectivity Dynamics (FCD)...")

# Set General Model Parameters
# ============================================================================
dtt = 1e-3  # Sampling rate of simulated neuronal activity (seconds)
dt  = 0.1

TR = 2.            # Sampling rate of saved simulated BOLD (seconds)
Tmax = 220.        # Number of timepoints in each fMRI session
Tmaxneuronal = int((Tmax+10.)*(TR/dtt))  # Number of simulated time points

N_windows = len(range(0,190,3))  # This shouldn't be done like this in Python!!!

# ============================================================================
# simulates the neuronal activity + BOLD for one subject
# ============================================================================
def computeSubjectSimulation(C, N):
    integrator.neuronalModel.initBookkeeping(N, Tmaxneuronal)
    integrator.simulate(dt, Tmaxneuronal, C)
    curr_xn, curr_rn = integrator.neuronalModel.returnBookkeeping()
    neuro_act = curr_rn
    return neuro_act


def computeSubjectBOLD(neuro_act, areasToSimulate=None):
    if not areasToSimulate:
        N = neuro_act.shape[1]
        areasToSimulate = range(N)
    else:
        N = len(areasToSimulate)
    #### BOLD
    # Friston BALLOON-WINDKESSEL MODEL
    BOLDModel.dt = dtt
    T = neuro_act.shape[0]*dtt  # Total time in seconds
    n_t = BOLDModel.computeRequiredVectorLength(T)
    BOLD_act = np.zeros([n_t,N])
    for nnew,area in enumerate(areasToSimulate):
        B = BOLDModel.BOLDModel(T,neuro_act[:,area])
        BOLD_act[:,nnew] = B
    step = int(np.round(TR/dtt))
    bds = BOLD_act[step-1::step, :]
    return bds


def simulateSingleSubject(C):
    N=C.shape[0]
    neuro_act = computeSubjectSimulation(C, N)
    bds = computeSubjectBOLD(neuro_act)
    return bds

# ============================================================================
# simulates the neuronal activity + BOLD + FCD for NumSubjects subjects
# ============================================================================
def simulate(NumSubjects, C):
    #N=C.shape[0]
    cotsampling = np.zeros([NumSubjects, int(N_windows*(N_windows-1)/2)])

    for nsub in range(NumSubjects):
        print('Subject:', nsub)

        bds = simulateSingleSubject(C)

        # Compute the FCD correlations.
        cotsampling[nsub,:] = FCD.FCD(bds.T)
        pass

    return cotsampling
