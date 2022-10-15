# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the simulation plus the BOLD model
#  This is a combination of the integrator (and model) evaluation together
#  with the BOLD model computation.
#
# --------------------------------------------------------------------------
#  by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
integrator = None  # import WholeBrain.Integrator_EulerMaruyama as integrator
BOLDModel = None  # import WholeBrain.BOLDHemModel_Stephan2007 as Stephan2007 # import WholeBrain.BOLDHemModel_Stephan2008 as Stephan2008
# import WholeBrain.Observables.swFCD as FCD

# Set General Model Parameters
# ============================================================================
dtt = 1e-3  # Sampling rate of simulated neuronal activity (seconds)
            # note: 1e-3 is the length of a millisecond, in seconds,
            # so basically this is a milliseconds to seconds conversion factor
            # and 1/dtt is a seconds to milliseconds conversion factor...
dt  = 0.1

TR = 2.            # Sampling rate of recorded BOLD simulation (seconds)
Tmax = 220.        # Number of (useful) time-points in each fMRI session
                   # each time-point is separated by TR seconds => Tmax * TR is the total length, in seconds
Toffset = 10.
Tmaxneuronal = int((Tmax+Toffset)*(TR/dtt))  # Number of simulated time points (in milliseconds)
def recomputeTmaxneuronal():  # if we need a different Tmax or TR or any other var, just use this function to rebuild Tmaxneuronal
    global Tmaxneuronal
    Tmaxneuronal = int((Tmax+Toffset)*(TR/dtt))
    print("New Tmaxneuronal={}".format(Tmaxneuronal))


# ============================================================================
# simulates the neuronal activity + BOLD for one subject
# ============================================================================
warmUp = False
warmUpFactor = 10.
def computeSubjectSimulation():
    # integrator.neuronalModel.SC = C
    # integrator.initBookkeeping(N, Tmaxneuronal)
    if warmUp:
        currObsVars = integrator.warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=Tmaxneuronal/warmUpFactor)
    else:
        currObsVars = integrator.simulate(dt, Tmaxneuronal)
    # currObsVars = integrator.returnBookkeeping()  # curr_xn, curr_rn
    neuro_act = currObsVars[:,1,:]  # curr_rn
    return neuro_act


def computeSubjectBOLD(neuro_act, areasToSimulate=None):
    if not areasToSimulate:
        N = neuro_act.shape[1]
        areasToSimulate = range(N)
    else:
        N = len(areasToSimulate)
    #### BOLD
    # Friston BALLOON-WINDKESSEL MODEL
    BOLDModel.dt = dtt  # BOLD integration time = 1 millisecond = 1e-3 seconds
    T = neuro_act.shape[0] * dtt  # Total time in seconds
    n_t = BOLDModel.computeRequiredVectorLength(T)
    BOLD_act = np.zeros([n_t,N])
    for nnew,area in enumerate(areasToSimulate):
        B = BOLDModel.BOLDModel(T,neuro_act[:,area])
        BOLD_act[:,nnew] = B

    step = int(np.round(TR/dtt))  # each step is the length of the TR, in milliseconds
    bds = BOLD_act[step-1::step, :]
    return bds


def simulateSingleSubject():
    # N=C.shape[0]
    neuro_act = computeSubjectSimulation()
    bds = computeSubjectBOLD(neuro_act)
    return bds

# ============================================================================
# ============================================================================
# ============================================================================EOF
