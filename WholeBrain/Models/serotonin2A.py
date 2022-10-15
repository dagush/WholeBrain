# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#
#  Computes simulations with the Dynamic Mean Field Model (DMF) using
#  Feedback Inhibitory Control (FIC) and Regional Drug Receptor Modulation (RDRM):
#
#  - the optimal coupling (we=2.1) for fitting the placebo condition
#  - the optimal neuromodulator gain for fitting the LSD condition (wge=0.2)
#
#
# Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
# Gustavo Deco, Josephine Cruzat, Joana Cabral, Gitte M. Knudsen, Robin L. Carhart-Harris, Peter C. Whybrow, Nikos K.
# Logothetis, and Morten L. Kringelbach, Current Biology 28, 3065–3074, October 8, 2018
#
#  Code written by Gustavo Deco gustavo.deco@upf.edu 2017
#  Reviewed by Josephine Cruzat and Joana Cabral
#
#  Translated to Python by Gustavo Patow
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from numba import jit
import WholeBrain.Models.DynamicMeanField as DMF

print("Going to use the serotonin 2A receptor (5-HT_{2A}R) transfer WholeBrain!")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    phie.recompile(); phii.recompile()


# Regional Drug Receptor Modulation (RDRM) constants for their transfer WholeBrain:
# serotonin 2A receptor (5-HT_{2A}R): the neuronal gain function of the model is modulated
# by the 5-HT_{2A}R density
# --------------------------------------------------------------------------
Receptor = 0
wgaini = 0.
wgaine = 0.


# transfer WholeBrain:
# --------------------------------------------------------------------------
# transfer function: excitatory
# in the paper, the wgaini is always set to 0, so in practice only wgaine is used... (0=Placebo, 0.2=LSD)
ae = 310.  # [nC^{-1}], g_E in the paper
be = 125.  # = g_E * I^{(E)_{thr}} in the paper = 310 * .403 [nA] = 124.93
de = 0.16
@jit(nopython=True)
def phie(x):
    # in the paper this was g_E * (I^{(E)_n} - I^{(E)_{thr}}) * (1 - receptor * gain_E)
    # Here, we distribute the first part as g_E * I^{(E)_n} - g_E * I^{(E)_{thr}}, thus...
    y = (ae*x-be)*(1+Receptor*wgaine)  # for LSD
    # if (y != 0):
    return y/(1-np.exp(-de*y))
    # else:
    #     return 0


# transfer function: inhibitory
# in the paper, the wgaini is always set to 0, so in practice only wgaine is used... (wgaini=0)
ai = 615  # [nC^{-1}], g_I in the paper
bi = 177  # = g_I * I^{(I)_{thr}} in the paper = 615 * .288 [nA] = 177.12
di = 0.087
@jit(nopython=True)
def phii(x):
    # in the paper this was g_I * (I^{(I)_n} - I^{(I)_{thr}}), without a
    # receptor part because inhibitory connections are not changed.
    # Apply same distributing as above...
    y = (ai*x-bi)*(1+Receptor*wgaini)  # for LSD
    # if (y != 0):
    return y/(1-np.exp(-di*y))
    # else:
    #     return 0


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Standard setup WholeBrain
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Simulation variables
# @jit(nopython=True)
def initSim(N):
    return DMF.initSim(N)

# --------------------------------------------------------------------------
# Variables of interest, needed for bookkeeping tasks...
# @jit(nopython=True)
def numObsVars():  # Returns the number of observation vars used, here xn and rn
    return DMF.numObsVars()

# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global wgaine, wgaini
    if 'S_E' in modelParms:
        wgaine = modelParms['S_E']
    if 'S_I' in modelParms:
        wgaini = modelParms['S_I']
    DMF.setParms(modelParms)


def getParm(parmList):
    if 'S_E' in parmList:
        return wgaine
    if 'S_I' in parmList:
        return wgaini
    return DMF.getParm(parmList)


# ----------------- Call the Dynamic Mean Field (a.k.a., reducedWongWang) ----------------------
@jit(nopython=True)
def dfun(simVars, I_external):
    return DMF.dfun(simVars, I_external)


DMF.He = phie
DMF.Hi = phii


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
