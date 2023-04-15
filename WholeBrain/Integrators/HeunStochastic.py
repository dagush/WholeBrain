# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Implementation of the Heun Stochastic Integrator
# Based on the code from TVB:
#     It is a simple example of a predictor-corrector method. It is also known as
#     modified trapezoidal method, which uses the Euler method as its predictor.
#     And it is also a implicit integration scheme.
#
#     [1] Kloeden and Platen, Springer 1995, *Numerical solution of stochastic
#         differential equations.
#     [2] Riccardo Mannella, *Integration of Stochastic Differential Equations
#         on a Computer*, Int J. of Modern Physics C 13(9): 1177--1194, 2002.
#
#          From [2]_:
#         .. math::
#             X_i(t) = X_i(t-1) + dX(X_i(t)/2 + dX(X_i(t-1))) dt + g_i(X) Z_1
#         in our case, :math:`noise = Z_1`
#         See page 1180.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from WholeBrain.Utils.randn2 import randn2
from WholeBrain.Utils import numTricks as iC
from numba import jit

print("Going to use the Heun Integrator...")

neuronalModel = None  # To be able to choose the model externally...
verbose = True


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    # # initBookkeeping.recompile()
    neuronalModel.recompileSignatures()
    recordBookkeeping.recompile()
    integrationStep.recompile()
    pass


# Matlab compatible definitions
# --------------------------------------------------------------------------
MatlabCompatibility = False
if MatlabCompatibility:
    def randn(vars,N):
        ra = randn2(vars,N)
        return ra #.reshape(-1)
else:
    from numpy.random import randn


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


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Heun Stochastic Integration
# --------------------------------------------------------------------------
sigma = 0.01
clamping = True
@jit(nopython=True)
def integrationStep(simVars, dt, stimulus):  #, curr_obsVars, doBookkeeping):
    def doClamping(simVariables):
        if clamping:
            simVariables = np.where(simVariables > 1., 1., simVariables)  # clamp values to 0..1
            simVariables = np.where(simVariables < 0., 0., simVariables)
        return simVariables

    numSimVars = simVars.shape[0]; N = simVars.shape[1]
    dvars_obsVars = neuronalModel.dfun(simVars, stimulus)
    dvars = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...

    noise = np.sqrt(dt) * sigma * randn(numSimVars,N)

    inter = simVars + dt * dvars + noise
    inter = doClamping(inter)

    dvars_obsVars = neuronalModel.dfun(inter, stimulus)
    dvars2 = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...
    dX = (dvars + dvars2) * dt / 2.0

    simVars = simVars + dX + noise
    simVars = doClamping(simVars)

    return simVars, obsVars


# # @jit(nopython=True)
def integrationLoop(dt, Tmaxneuronal, simVars, doBookkeeping, curr_obsVars):
    # Variables:
    # dt = integration time step in milliseconds
    # Tmaxneuronal = total time to integrate in milliseconds
    for t in np.arange(0, Tmaxneuronal, dt):
        stimulus = allStimuli[int(t / dt)]
        simVars_obsVars = integrationStep(simVars, dt, stimulus)
        simVars = simVars_obsVars[0]; obsVars = simVars_obsVars[1]  # cannot use unpacking in numba...
        if doBookkeeping:
            curr_obsVars = recordBookkeeping(t, obsVars, curr_obsVars)
    return simVars, curr_obsVars


# # @jit(nopython=True)
def integrate(dt, Tmaxneuronal, simVars, doBookkeeping = True):
    # numSimVars = simVars.shape[0]
    N = simVars.shape[1]  # N = neuronalModel.SC.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    curr_obsVars = initBookkeeping(N, Tmaxneuronal)
    return integrationLoop(dt, Tmaxneuronal, simVars, doBookkeeping, curr_obsVars)


# ==========================================================================
# ==========================================================================
# ==========================================================================
def simulate(dt, Tmaxneuronal):
    if verbose:
        print("Simulating...", flush=True)
    N = neuronalModel.getParm('SC').shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    initStimuli(dt, Tmaxneuronal)
    simVars, obsVars = integrate(dt, Tmaxneuronal, simVars)
    return obsVars


def warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=10000):
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


# ======================================================================
# Debug/test code
# To use it, comment the @jit(nopython=True) line at integrationStep
# Otherwise you'll get weird numba errors
# ======================================================================
if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt

    class dummyNeuronalModel:
        def __init__(self):
            pass
        # We will use the differential equation y"(t) = -y(t).
        # This is written as
        #     y" = v' -> y' = v
        #                v' = -y
        def dfun(self, simVars, p):
            y = simVars[0]
            v = simVars[1]
            dy = v
            dv = -y
            return np.stack((dy,dv)), np.stack((y, v))
        def recompileSignatures(self):
            pass
        def numObsVars(self):
            return 2

    neuronalModel = dummyNeuronalModel()

    # The analytic solution is y = sin(t) because we initialize at 0.0.
    def asol(t):
        return math.sin(t)

    # Note that we are setting a CONSTANT sigma of 0.5, so it's better to use something like a sinus to visualize
    # its effects...
    clamping = False
    sigma = 0.05
    dt = 0.1
    ds = dt
    Tmax = 20.0
    y0 = np.array([[0.0], [1.0]])  # y0=0, v0=cos(0)=1

    initStimuli(dt, Tmax)

    simVars, obsVars = integrate(dt, Tmax, y0)

    t = np.arange(0.0, Tmax, dt)
    yasol = np.vectorize(asol)
    plt.plot(t,obsVars[:,0,:].flatten()[:-1],'r-',label="Heun Stochastic's")
    plt.plot(t,yasol(t),'b-', label='analytical')
    plt.legend()
    plt.show()

# ==========================================================================
# ==========================================================================
# ========================================================================== --EOF
