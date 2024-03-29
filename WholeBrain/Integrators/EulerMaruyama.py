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

print("Going to use the Euler-Maruyama Integrator...")

neuronalModel = None  # To be able to choose the model externally...


sigma = 0.01
# def buildNoise():  # for heterogeneous sigma
#     global sigma
#     if type(sigma) is not np.ndarray:
#         sigma = np.array(sigma).reshape(-1,)


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Euler-Maruyama Integration
# --------------------------------------------------------------------------
@jit(nopython=True)
def integrationStep(simVars, dt, coupling, stimulus):  #, curr_obsVars, doBookkeeping):
    numSimVars = simVars.shape[0]; N = simVars.shape[1]
    dvars_obsVars = neuronalModel.dfun(simVars, coupling, stimulus)
    dvars = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...
    # sigma[:,np.newaxis] * randn(numSimVars, 5)  # for heterogeneous sigma
    simVars = simVars + dt * dvars + np.sqrt(dt) * sigma * randn(numSimVars,N)  # Euler-Maruyama integration. -> change to @ if heterogeneous sigma...
    return simVars, obsVars


# ==========================================================================
# ==========================================================================
# ========================================================================== --EOF
