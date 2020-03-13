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
import functions.Integrator_EulerMaruyama as integrator
import functions.BOLDHemModel_Stephan2007 as BOLDModel
# from functions import BOLDHemModel_Stephan2008 as BOLDModel
import functions.FCD as FCD

print("Going to use Functional Connectivity Dynamics (FCD)...")

# Set General Model Parameters
# ============================================================================
dtt = 1e-3  # Sampling rate of simulated neuronal activity (seconds)
dt  = 0.1

TR = 2.            # Sampling rate of saved simulated BOLD (seconds)
Tmax = 220.        # Number of timepoints in each fMRI session
Tmaxneuronal = int((Tmax+10.)*(TR/dtt))  # Number of simulated time points
def recomputeTmaxneuronal():  # if we need a different Tmax or TR or any other var, just use this function to rebuild Tmaxneuronal
    global Tmaxneuronal
    Tmaxneuronal = int((Tmax+10.)*(TR/dtt))
    print("New Tmaxneuronal={}".format(Tmaxneuronal))


windowSize = FCD.windowSize
N_windows = len(np.arange(0,Tmax-windowSize,3))  # This shouldn't be done like this in Python!!!


# ============================================================================
# simulates the neuronal activity + BOLD for one subject
# ============================================================================
warmUpFactor = 10.
def computeSubjectSimulation(C, N, warmup):
    integrator.neuronalModel.SC = C
    integrator.neuronalModel.initBookkeeping(N, Tmaxneuronal)
    if warmup:
        integrator.warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=Tmaxneuronal/warmUpFactor)
    else:
        integrator.simulate(dt, Tmaxneuronal)
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


# ============================================================================
# simulates the neuronal activity + BOLD for ONE subject
# ============================================================================
def simulateSingleSubject(C, warmup=False):
    N=C.shape[0]
    neuro_act = computeSubjectSimulation(C, N, warmup)
    bds = computeSubjectBOLD(neuro_act)
    return bds


# ============================================================================
# simulates the neuronal activity + BOLD + FCD for ONE subject
# ============================================================================
def simSingleSubjectFCD(C, warmup=False):
    bds = simulateSingleSubject(C, warmup=warmup)
    cotsampling = FCD.FCD(bds.T)  # Compute the FCD correlations
    return cotsampling


# ============================================================================
# simulates the neuronal activity + BOLD + FCD for NumSubjects subjects
# ============================================================================
def simulate(NumSubjects, C, warmup=False):
    #N=C.shape[0]
    cotsampling = np.zeros([NumSubjects, int(N_windows*(N_windows-1)/2)])

    for nsub in range(NumSubjects):
        print('Subject:', nsub)
        # Compute the FCD correlations.
        cotsampling[nsub,:] = simSingleSubjectFCD(C, warmup=warmup)

    return cotsampling
