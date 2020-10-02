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
integrator = None  # import functions.Integrator_EulerMaruyama as integrator
BOLDModel = None  # import functions.BOLDHemModel_Stephan2007 as Stephan2007 # import functions.BOLDHemModel_Stephan2008 as Stephan2008
import functions.swFCD as FCD

print("Going to use Functional Connectivity Dynamics (FCD)...")

# Set General Model Parameters
# ============================================================================
dtt = 1e-3  # Sampling rate of simulated neuronal activity (seconds)
dt  = 0.1

TR = 2.            # Sampling rate of saved simulated BOLD (seconds)
Tmax = 220.        # Number of timepoints in each fMRI session
Toffset = 10.
Tmaxneuronal = int((Tmax+Toffset)*(TR/dtt))  # Number of simulated time points
def recomputeTmaxneuronal():  # if we need a different Tmax or TR or any other var, just use this function to rebuild Tmaxneuronal
    global Tmaxneuronal
    Tmaxneuronal = int((Tmax+Toffset)*(TR/dtt))
    print("New Tmaxneuronal={}".format(Tmaxneuronal))


windowSize = FCD.windowSize
N_windows = len(np.arange(0,Tmax-windowSize,3))  # This shouldn't be done like this in Python!!!


# ============================================================================
# simulates the neuronal activity + BOLD for one subject
# ============================================================================
warmUpFactor = 10.
def computeSubjectSimulation(C, N, warmup):
    integrator.neuronalModel.SC = C
    # integrator.initBookkeeping(N, Tmaxneuronal)
    if warmup:
        currObsVars = integrator.warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=Tmaxneuronal/warmUpFactor)
    else:
        currObsVars = integrator.simulate(dt, Tmaxneuronal)
    # currObsVars = integrator.returnBookkeeping()  # curr_xn, curr_rn
    neuro_act = currObsVars[:,1,:]  # curr_rn
    return neuro_act


# kkcounter = 0
def computeSubjectBOLD(neuro_act, areasToSimulate=None):
    # global kkcounter
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
        # if nnew == 14 and kkcounter == 21-1:
        # kkcounter += 1
        # print(f'here: {kkcounter} ({nnew})')
        B = BOLDModel.BOLDModel(T,neuro_act[:,area])
        # B2 = BOLDModel.BOLDModel(T,neuro_act[:,area])
        # B3 = BOLDModel.BOLDModel(T,neuro_act[:,area])
        BOLD_act[:,nnew] = B

        # array_has_nan = np.isnan(np.sum(B))  # code to check whether we have a nan in the arrays...
        # array2_has_nan = np.isnan(np.sum(B2))  # code to check whether we have a nan in the arrays...
        # array3_has_nan = np.isnan(np.sum(B3))  # code to check whether we have a nan in the arrays...
        # if array_has_nan: # or array2_has_nan or array3_has_nan:
        #     print(f"NAN!!! ({nnew})")
        #     B2 = BOLDModel.BOLDModel(T,neuro_act[:,area])

    step = int(np.round(TR/dtt))
    bds = BOLD_act[step-1::step, :]
    return bds


def simulateSingleSubject(C, warmup=False):
    N=C.shape[0]
    neuro_act = computeSubjectSimulation(C, N, warmup)
    bds = computeSubjectBOLD(neuro_act)
    return bds


# ============================================================================
# ============================================================================
# ============================================================================
# simulates the neuronal activity + BOLD + FCD for ONE subject
# ============================================================================
def simSingleSubjectFCD(C, warmup=False):
    bds = simulateSingleSubject(C, warmup=warmup)
    cotsampling = FCD.from_fMRI(bds.T)  # Compute the FCD correlations
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


# ============================================================================
# ============================================================================
# ============================================================================EOF
