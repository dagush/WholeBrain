# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
#  Computes the simulation (without a BOLD model, as in the case of the Hopf model)
#  In the end, this is a simple wrapper around the integrator (and the model)
#  and does not do much, except for a couple of minor details (i.e., warmup)...
# ---------------------------------------------------------------------------------
#  by Gustavo Patow
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
import numpy as np
integrator = None

# Set General Model Parameters
# ============================================================================
dtt = 1e-3  # Sampling rate of simulated neuronal activity (seconds)
            # note: 1e-3 is the length of a millisecond, in seconds,
            # so basically this is a milliseconds to seconds conversion factor
dt  = 0.1

TR = 2.            # Sampling rate of saved simulated BOLD (seconds)
t_min = TR/dt      # Skip t_min first samples

Tmax = 220.        # Number of (useful) time-points in each fMRI session
                   # each time-point is separated by TR seconds => Tmax * TR is the total length, in seconds
Toffset = 10.      # Number of initial time-points to skip
# We simulate Tmax+Toffset time-points with the idea of extracting Tmax useful time-points.
Tmaxneuronal = int((Tmax+Toffset)*(TR/dtt))  # Number of simulated time points
def recomputeTmaxneuronal():  # if we need a different Tmax or TR or any other var, just use this function to rebuild Tmaxneuronal
    global Tmaxneuronal, t_min
    t_min = TR/dt
    Tmaxneuronal = int((Tmax+Toffset)*(TR/dtt))
    print("New Tmaxneuronal={}".format(Tmaxneuronal))


# =======================================================================================
# Simulates the neuronal activity directly for one subject (useful for Hopf-like models)
# Calls the integrator to integrate with dt up to Tmaxneuronal sample points.
# =======================================================================================
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


# =======================================================================================
# Calls the function to simulate the neuronal activity directly for one subject,
# then SUBSAMPLE it (at rate TR/dtt) for BOLD computations
# =======================================================================================
def simulateSingleSubject():
    # N=C.shape[0]
    BOLD_act = computeSubjectSimulation()
    # now, (sub)sample the BOLD signal to obtain the final fMRI signal
    n_min = int(np.round(t_min / dtt))
    step = int(np.round(TR/dtt))
    # No need for a BOLD simulation, the result of the model directly can be used as BOLD signal
    # Discard the first n_min samples
    bds = BOLD_act[n_min+step-1::step, :]
    return bds


# ============================================================================
# ============================================================================
# ============================================================================EOF
