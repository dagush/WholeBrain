# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Implementation of the Euler-Maruyama Integrator
# Based on the code from the paper:
#
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
#
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from WholeBrain.Integrators.integr_utils import *
from numba import jit

print("Going to use the Euler integraation scheme...")

neuronalModel = None  # To be able to choose the model externally...


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Euler Integration
# --------------------------------------------------------------------------
# sigma = 0.01
# @jit(nopython=True)
def integrationStep(simVars, dt, coupling, stimulus):
    # numSimVars = simVars.shape[0]; N = simVars.shape[1]
    dvars_obsVars = neuronalModel.dfun(simVars, coupling, stimulus)
    dvars = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...
    simVars = simVars + dt * dvars  # Euler integration for S^E (9).
    return simVars, obsVars


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
