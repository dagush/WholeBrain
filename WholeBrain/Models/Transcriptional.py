# ==========================================================================
# ==========================================================================
# ==========================================================================
# This is the model for the transcriptional variations in excitatory and
# inhibitory receptor (E:I) gene expression, which is an extension of
# the Dynamic Mean Field (DMF) model [Deco_2014]:
#
#  Taken from the code (slurm.sbatch_genes_balanced_G_optimization.m) from:
#  [DecoEtAl_2021] Gustavo Deco, Kevin Aquino, Aurina Arnatkeviciute, Stuart Oldham, Kristina Sabaroedin,
#  Nigel Rogasch, Morten L. Kringelbach, and Alex Fornito, "Dynamical consequences of regional heterogeneity
#  in the brain’s transcriptional landscape", 2021, biorXiv
#
#
# The original DMF model:
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#             How local excitation-inhibition ratio impacts the whole brain dynamics
#             J. Neurosci., 34 (2014), pp. 7886-7898
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
import numpy as np
from numba import jit
import WholeBrain.Models.DynamicMeanField as DMF


print("Going to use the Dynamic Mean Field (DMF) neuronal model...")


def recompileSignatures():
    DMF.recompileSignatures()
    dfun.recompile()


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Transfer WholeBrain:
# --------------------------------------------------------------------------
# Parameters for the the scaling between regional biological measures of
# heterogeneity, R_i, and the effective gain within a region
alpha = 0.
beta = 0.
ratio = 0.

# transfer function: excitatory
ae = 310.  # [nC^{-1}], g_E in the paper
be = 125.  # = g_E * I^{(E)_{thr}} in the paper = 310 * .403 [nA] = 124.93
de = 0.16
@jit(nopython=True)
def phie_gain(x):
    gain=1+alpha+beta*ratio
    y = (ae*x-be)*gain
    return y/(1-np.exp(-de*y))


# transfer function: inhibitory
ai = 615  # [nC^{-1}], g_I in the paper
bi = 177  # = g_I * I^{(I)_{thr}} in the paper = 615 * .288 [nA] = 177.12
di = 0.087
@jit(nopython=True)
def phii_gain(x):
    # Apply same distributing as above...
    gain=1+alpha+beta*ratio
    y = (ai*x-bi)*gain
    return y/(1-np.exp(-di*y))


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
# Set the parameters for this model (and pass the rest to the DMF)
def setParms(modelParms):
    global alpha, beta
    if 'alpha' in modelParms:
        alpha = modelParms['alpha']
    if 'beta' in modelParms:
        beta = modelParms['beta']
    DMF.setParms(modelParms)


def getParm(parmList):
    if 'alpha' in parmList:
        return alpha
    if 'beta' in parmList:
        return beta
    return DMF.getParm(parmList)


# ----------------- Call the Dynamic Mean Field (a.k.a., reducedWongWang) ----------------------
@jit(nopython=True)
def dfun(simVars, I_external):
    return DMF.dfun(simVars, I_external)


DMF.He = phie_gain
DMF.Hi = phii_gain
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
