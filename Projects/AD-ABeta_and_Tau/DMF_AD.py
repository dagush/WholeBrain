# ==========================================================================
# ==========================================================================
# ==========================================================================
# Model for AD simulations.
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# Adapted Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
#
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#             How local excitation-inhibition ratio impacts the whole brain dynamics
#             J. Neurosci., 34 (2014), pp. 7886-7898
#
# Modifyied by Gustavo Patow for Alzheimer's Disease simulations
# ==========================================================================
import numpy as np
from numba import jit

print("Going to use the AD Dynamic Mean Field (adDMF) neuronal model...")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    phie.recompile()
    phii.recompile()
    dfun.recompile()


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) Model Constants
# --------------------------------------------------------------------------
taon = 100.
taog = 10.
gamma_e = 0.641 / 1000
gamma_i = 1. / 1000.
J_NMDA = 0.15       # [nA] NMDA current
I0 = 0.382  #.397  # [nA] overall effective external input
Jexte = 1.
Jexti = 0.7
w = 1.4
we = 2.1        # Global coupling scaling (G in the paper)
SC = None       # Structural connectivity (should be provided externally)


# ==========================================================================
# AD param initialization...
# --------------------------------------------------------------------------
M_e = 1.    # WARNING: In general, this must be initialized outside!
M_i = 1.
Abeta = None
Tau = None

def set_AD_Burden(coefs):
    global M_e, M_i
    # First, the excitatory contribution
    M_e = (1 + coefs[0] + coefs[1] * Abeta) * (1 + coefs[2] + coefs[3] * Tau)
    # Now, the inhibitory contribution
    M_i = (1 + coefs[4] + coefs[5] * Abeta) * (1 + coefs[6] + coefs[7] * Tau)

    recompileSignatures()


# --------------------------------------------------------------------------
# Simulation variables
# @jit(nopython=True)
def initSim(N):
    sn_sg = 0.001 * np.ones((2, N))  # Here we initialize with np.ones, but other models use np.zeros...
    return sn_sg


J = None    # WARNING: In general, J must be initialized outside!
def initJ(N):  # A bit silly, I know...
    global J
    J = np.ones(N)


# --------------------------------------------------------------------------
# Variables of interest, needed for bookkeeping tasks...
# xn = None  # xn, excitatory synaptic activity
# rn = None  # rn, excitatory firing rate
# @jit(nopython=True)
def numObsVars():  # Returns the number of observation vars used, here xn and rn
    return 3


# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global we, J, SC, M_e, M_i
    if 'we' in modelParms:
        we = modelParms['we']
    if 'J' in modelParms:
        J = modelParms['J']
    if 'SC' in modelParms:
        SC = modelParms['SC']
    if 'M_e' in modelParms:
        M_e = modelParms['M_e']
    if 'M_i' in modelParms:
        M_i = modelParms['M_i']
    if 'coefs' in modelParms:
        set_AD_Burden(modelParms('coefs'))


def getParm(parmList):
    if 'we' in parmList:
        return we
    if 'J' in parmList:
        return J
    if 'be' in parmList:
        return be
    if 'ae' in parmList:
        return ae
    if 'SC' in parmList:
        return SC
    return None


# -----------------------------------------------------------------------------
# ----------------- Dynamic Mean Field (a.k.a., reducedWongWang) --------------
# -----------------------------------------------------------------------------

# ----------------- Coumpling ----------------------
# @jit(nopython=True)
# def instantaneousDirectCouplingOp(x):
#     return SC @ x      # coupling = SC @ sn
couplingOp = None


# --------------------------------------------------------------------------
# transfer WholeBrain:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae=310.
be=125.
de=0.16
@jit(nopython=True)
def phie(x):
    y = M_e * (ae*x-be)  # M_e will be set by function set_AD_Burden
    return y/(1.-np.exp(-de*y))


# transfer function: inhibitory
ai=615.
bi=177.
di=0.087
@jit(nopython=True)
def phii(x):
    y = M_i * (ai*x-bi)  # This is for Modality A in our study...
                         # M_i will be set by function set_AD_Burden
    return y/(1.-np.exp(-di*y))


# transfer WholeBrain used by the simulation...
He = phie
Hi = phii


@jit(nopython=True)
def dfun(simVars, coupling, I_external):
    sn = simVars[0]; sg = simVars[1]  # should be [sn, sg] = simVars
    # This is the regular DMF behaviour
    coupl = coupling.couple(sn)
    xn = I0 * Jexte + w * J_NMDA * sn + we * J_NMDA * coupl - J * sg + I_external  # Eq for I^E (5). I_external = 0 => resting state condition.
    xg = I0 * Jexti + J_NMDA * sn - sg  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
    rn = He(xn)  # Calls He(xn). r^E = H^E(I^E) in the paper (7)
    rg = Hi(xg)  # Calls Hi(xg). r^I = H^I(I^I) in the paper (8)
    dsn = -sn / taon + (1. - sn) * gamma_e * rn
    dsg = -sg / taog + rg * gamma_i
    return np.stack((dsn, dsg)), np.stack((xn, rn, rg))

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
