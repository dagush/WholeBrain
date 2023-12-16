# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Implementation of a generic Integrator
#
# By Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from WholeBrain.Utils import numTricks as iC
from numba import jit


import WholeBrain.Integrators.integr_utils as integr_utils
verbose = False
neuronalModel = None
integrationScheme = None
coupling = None


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    # # initBookkeeping.recompile()
    integr_utils.doClamping.recompile()
    neuronalModel.recompileSignatures()
    recordBookkeeping.recompile()
    integrationScheme.integrationStep.recompile()
    pass


# Functions to convert the stimulus from a function to an array
# --------------------------------------------------------------------------
stimuli = None  # To add some stimuli, if needed...
allStimuli = None
def initStimuli(dt, Tmaxneuronal):
    global allStimuli
    tValues = np.arange(0, Tmaxneuronal, dt)
    if stimuli is not None:
        allStimuli = np.array(list(map(stimuli.stimulus, tValues)))
    else:
        allStimuli = np.zeros(len(tValues))


# bookkeeping vars & methods -> Just forward them to the neuronal model we are using...
# ==========================================================================
# ==========================================================================
# ==========================================================================
# Bookkeeping variables of interest...
# --------------------------------------------------------------------------
ds = 1  # downsampling stepsize
# # @jit(nopython=True)
def initBookkeeping(N, tmax):
    # global curr_xn, curr_rn, nn
    # global curr_obsVars
    # curr_xn = np.zeros((int(tmax), N))
    # curr_rn = np.zeros((int(tmax), N))
    obsVars = neuronalModel.numObsVars()
    timeElements = int(tmax/ds) + 1  # the last +1 because of isClose roundings...
    return np.zeros((timeElements, obsVars, N))


@jit(nopython=True)
def recordBookkeeping(t, obsVars, curr_obsVars):
    # global curr_obsVars
    if iC.isInt(t/ds):
        nn = int(np.round(t/ds))  # it is an int-ish...
        curr_obsVars[nn,:,:] = obsVars[:,:]
    return curr_obsVars


# # @jit(nopython=True)
def integrationLoop(dt, Tmaxneuronal, simVars, doBookkeeping, curr_obsVars, coupling):
    # Variables:
    # dt = integration time step in milliseconds
    # Tmaxneuronal = total time to integrate in milliseconds
    for t in np.arange(0, Tmaxneuronal, dt):
        stimulus = allStimuli[int(t / dt)]
        simVars_obsVars = integrationScheme.integrationStep(simVars, dt, coupling, stimulus)
        simVars = simVars_obsVars[0]; obsVars = simVars_obsVars[1]  # cannot use unpacking in numba...
        if doBookkeeping:
            curr_obsVars = recordBookkeeping(t, obsVars, curr_obsVars)
    return simVars, curr_obsVars


# # @jit(nopython=True)
def integrate(dt, Tmaxneuronal, simVars, doBookkeeping = True):
    # numSimVars = simVars.shape[0]
    recompileSignatures()
    N = simVars.shape[1]  # N = neuronalModel.SC.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    curr_obsVars = initBookkeeping(N, Tmaxneuronal)
    integrResult = integrationLoop(dt, Tmaxneuronal, simVars, doBookkeeping, curr_obsVars, neuronalModel.couplingOp)
    return integrResult


# ==========================================================================
# ==========================================================================
# ==========================================================================
def simulate(dt,  # integration step, in milliseconds
             Tmaxneuronal  # integration length, in milliseconds
             ):
    if verbose:
        print("Simulating...", flush=True)
    N = neuronalModel.getParm('SC').shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    initStimuli(dt, Tmaxneuronal)
    simVars, obsVars = integrate(dt, Tmaxneuronal, simVars)
    return obsVars


def warmUpAndSimulate(dt,  # integration step, in milliseconds
                      Tmaxneuronal,  # integration length, in milliseconds
                      TWarmUp=10000  # Warmpup lenth, in milliseconds
                      ):
    N = neuronalModel.getParm('SC').shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    if verbose:
        print("Warming Up...", end=" ", flush=True)
    initStimuli(dt, TWarmUp)
    simVars, obsVars = integrate(dt, TWarmUp, simVars, doBookkeeping=False)
    if verbose:
        print("and simulating!!!", flush=True)
    initStimuli(dt, Tmaxneuronal)
    simVars, obsVars = integrate(dt, Tmaxneuronal, simVars, doBookkeeping=True)
    return obsVars

# ==========================================================================
# ==========================================================================
# ========================================================================== --EOF