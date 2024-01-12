# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
#
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#             How local excitation-inhibition ratio impacts the whole brain dynamics
#             J. Neurosci., 34 (2014), pp. 7886-7898
# ==========================================================================
import numpy as np
from numba import jit

import WholeBrain.Integrators.integr_utils as utils

print("Going to use the Dynamic Mean Field (DMF) neuronal model...")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    phie.recompile()
    phii.recompile()
    dfun.recompile()
    pass


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) Model Constants
# --------------------------------------------------------------------------
taon = 100.
taog = 10.
gamma_e = 0.641
gamma_i = 1.
J_NMDA = 0.15       # [nA] NMDA current
I0 = 0.382   #.397  # [nA] overall effective external input
Jexte = 1.
Jexti = 0.7
w = 1.4
we = 2.1        # Global coupling scaling (G in the paper)
SC = None       # Structural connectivity (should be provided externally)


# --------------------------------------------------------------------------
# Simulation variables
# @jit(nopython=True)
def initSim(N):
    Se = 0.001 * np.ones(N)  # Initialize sn (S^E in the paper)
    Si = 0.001 * np.ones(N)  # Initialize sg (S^I in the paper)
    return np.stack((Se, Si))




# --------------------------------------------------------------------------
# Variables of interest, needed for bookkeeping tasks...
# Se, excitatory synaptic activity
# re, excitatory firing rate
# @jit(nopython=True)
def numObsVars():  # Returns the number of observation vars used, here xn and rn
    return 2


# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global we, J, SC
    if 'we' in modelParms:
        we = modelParms['we']
    if 'G' in modelParms:  # I've made this mistake too many times...
        we = modelParms['G']
    if 'J' in modelParms:
        J = modelParms['J']
    if 'SC' in modelParms:
        SC = modelParms['SC']


def getParm(parmName):
    if 'we' in parmName or 'G' in parmName:  # I've made this mistake too many times...
        return we
    if 'J' in parmName:
        return J
    if 'be' in parmName:
        return be
    if 'ae' in parmName:
        return ae
    if 'SC' in parmName:
        return SC
    return None


# -----------------------------------------------------------------------------
# ----------------- Dynamic Mean Field (a.k.a., reducedWongWang) --------------
# -----------------------------------------------------------------------------

# ----------------- Coumpling ----------------------
from WholeBrain.Models.Couplings import instantaneousDirectCoupling
couplingOp = instantaneousDirectCoupling()  # The only one who knows the coupling operation is the model itself!!!


# ----------------- Model ----------------------
# transfer WholeBrain:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae = 310.  # [nC^{-1}], g_E in the paper
be = 125.  # = g_E * I^{(E)_{thr}} in the paper = 310 * .403 [nA] = 124.93
de=0.16
@jit(nopython=True)
def phie(x):
    # in the paper this was g_E * (I^{(E)_n} - I^{(E)_{thr}})
    # Here, we distribute as g_E * I^{(E)_n} - g_E * I^{(E)_{thr}}, thus...
    y = (ae*x-be)
    # if (y != 0):
    return y/(1.-np.exp(-de*y))
    # else:
    #     return 0


# transfer function: inhibitory
ai = 615.  # [nC^{-1}], g_I in the paper
bi = 177.  # = g_I * I^{(I)_{thr}} in the paper = 615 * .288 [nA] = 177.12
di=0.087
@jit(nopython=True)
def phii(x):
    # in the paper this was g_I * (I^{(I)_n} - I^{(I)_{thr}}).
    # Apply same distributing as above...
    y = (ai*x-bi)
    # if (y != 0):
    return y/(1.-np.exp(-di*y))
    # else:
    #     return 0

# transfer WholeBrain used by the simulation...
He = phie
Hi = phii


@jit(nopython=True)
def dfun(simVars, coupling, I_external):
    Se = simVars[0]; Si = simVars[1]  # should be [Se, Si] = simVars
    Se = utils.doClamping(Se); Si = utils.doClamping(Si)  # Clamping, needed in Deco2014 model and derivatives...

    coupl = coupling.couple(Se)
    Ie = I0 * Jexte + w * J_NMDA * Se + we * J_NMDA * coupl - J * Si + I_external  # Eq for I^E (5). I_external = 0 => resting state condition.
    Ii = I0 * Jexti + J_NMDA * Se - Si  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
    re = He(Ie)  # Calls He(Ie). r^E = H^E(I^E) in the paper (7)
    ri = Hi(Ii)  # Calls Hi(Ii). r^I = H^I(I^I) in the paper (8)
    dSe = -Se / taon + (1. - Se) * gamma_e * re/1000.  # divide by 1000 because re is in Hz = 1/second, we need milliseconds!
    dSi = -Si / taog + gamma_i * ri/1000.
    return np.stack((dSe, dSi)), np.stack((Ie, re))


# ==========================================================================
# ==========================================================================
# ==========================================================================
