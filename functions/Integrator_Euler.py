
# Implementation of the Euler Integrator
# Based on the code from the paper:
#
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
#
#
# --------------------------------------------------------------------------
import numpy as np
from functions.randn2 import randn2

neuronalModel = None  # To be able to choose the model externally...
stimuli = None        # To add some stimuli, if needed...

print("Going to use the Euler deterministic Integrator...")

verbose = True


# Matlab compatible definitions
# --------------------------------------------------------------------------
def randn(N):
    ra = randn2(N)
    return ra.reshape(-1, 1)


# # bookkeeping vars & methods -> Just forward them to the neuronal model we are using...
# # -------------------------------------------------------------------------------------
# def initBookkeeping(N, tmax):
#     neuronalModel.initBookkeeping(N, tmax)
#
#
# def returnBookkeeping():
#     return neuronalModel.returnBookkeeping()


# Euler Integration
# --------------------------------------------------------------------------
clamping = True
simVars = []
sigma = 0.01
def integrate(dt, Tmaxneuronal, doBookkeeping = True):
    global simVars
    N = neuronalModel.SC.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    for t in np.arange(0, Tmaxneuronal, dt):
        if stimuli:
            stimulus = stimuli.stimulus(t)
        else:
            stimulus = 0.
        dvars = neuronalModel.dfun(simVars, stimulus)
        for varPos, dvar in enumerate(dvars):
            var = simVars[varPos]
            var = var + dt * dvar  # Euler integration for S^E (9).
            if clamping:
                var[var > 1] = 1.  # clamp values to 0..1
                var[var < 0] = 0.
            simVars[varPos] = var

        if doBookkeeping:
            neuronalModel.recordBookkeeping(t)


def simulate(dt, Tmaxneuronal):
    global simVars
    if verbose:
        print("Simulating...", flush=True)
    N = neuronalModel.SC.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    integrate(dt, Tmaxneuronal)


def warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp = 10000):
    global simVars
    N = neuronalModel.SC.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    if verbose:
        print("Warming Up...", end=" ", flush=True)
    integrate(dt, TWarmUp, doBookkeeping=False)
    if verbose:
        print("and simulating!!!", flush=True)
    integrate(dt, Tmaxneuronal, doBookkeeping=True)

