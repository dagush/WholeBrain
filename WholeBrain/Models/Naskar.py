# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) with local inhibitory plasticity for the Feedback Inhibition Control
#
#  Implemented from:
#  [NaskarEtAl_2018] Amit Naskar, Anirudh Vattikonda, Gustavo Deco,
#      Dipanjan Roy, Arpan Banerjee; Multiscale dynamic mean field (MDMF)
#      model relates resting-state brain dynamics with local cortical
#      excitatory–inhibitory neurotransmitter homeostasis.
#      Network Neuroscience 2021; 5 (3): 757–782.
#      DOI: https://doi.org/10.1162/netn_a_00197
#
# Based on the works by
# [VogelsEtAl_] T. P. Vogels et al., Inhibitory Plasticity Balances Excitation and Inhibition in
#      Sensory Pathways and Memory Networks.Science334,1569-1573(2011).
#      DOI: 10.1126/science.1211095
# [HellyerEtAl_] Peter J. Hellyer, Barbara Jachs, Claudia Clopath, Robert Leech, Local inhibitory
#      plasticity tunes macroscopic brain dynamics and allows the emergence of functional brain
#      networks, NeuroImage,  Volume 124, Part A, 1 January 2016, Pages 85-95
#      DOI: 10.1016/j.neuroimage.2015.08.069
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#      How local excitation-inhibition ratio impacts the whole brain dynamics
#      J. Neurosci., 34 (2014), pp. 7886-7898
#
# By Facundo Faragó and Gustavo Doctorovich
# November 2023
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
t_glu = 7.46    # concentration of glutamate
t_gaba = 1.82   # concentration of GABA
We = 1.0        # scaling of external input current to excitatory population
Wi = 0.7        # scaling of external input current to inhibitory population
alfa_e = 0.072  # forward rate constant for NMDA gating
alfa_i = 0.53   # forward rate constant for GABA gating
B_e = 0.0066    # ms^-1  backward rate constant for NMDA gating
B_i = 0.18      # ms^-1  backward rate constant for GABA gating
J_NMDA = 0.15   # [nA] NMDA current
I0 = 0.382      #.397  # [nA] overall effective external input
gamma = 1.      # Learning rate
w = 1.4         # weight for recurrent self-excitation in each excitatory population
rho = 3         # target-firing rate of the excitatory population is maintained at the 3 Hz

SC = None       # Structural connectivity (should be provided externally)


# transfer WholeBrain:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae = 310.  # [nC^{-1}], g_E in the paper
be = 125.  # = g_E * I^{(E)_{thr}} in the paper = 310 * .403 [nA] = 124.93
de = 0.16
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
di = 0.087
@jit(nopython=True)
def phii(x):
    # in the paper this was g_I * (I^{(I)_n} - I^{(I)_{thr}}).
    # Apply same distributing as above...
    y = (ai*x-bi)
    # if (y != 0):
    return y/(1.-np.exp(-di*y))
    # else:
    #     return 0


# --------------------------------------------------------------------------
# transfer WholeBrain used by the simulation...
He = phie
Hi = phii

# --------------------------------------------------------------------------
# Simulation variables
# @jit(nopython=True)
def initSim(N):
    sn = 0.001 * np.ones(N)  # Initialize sn (S^E in the paper)
    sg = 0.001 * np.ones(N)  # Initialize sg (S^I in the paper)
    J = 1. * np.ones(N)
    return np.stack((sn,sg,J))


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
    global we, SC
    if 'we' in modelParms:
        we = modelParms['we']
    if 'G' in modelParms:  # I've made this mistake too many times...
        we = modelParms['G']
    if 'SC' in modelParms:
        SC = modelParms['SC']


def getParm(parmList):
    if 'we' in parmList or 'G' in parmList:  # I've made this mistake too many times...
        return we
    if 'be' in parmList:
        return be
    if 'ae' in parmList:
        return ae
    if 'SC' in parmList:
        return SC
    return None


# --------------------------------------------------------------------------
# ----------------------------- Coumpling ----------------------------------
# --------------------------------------------------------------------------
import WholeBrain.Models.Couplings as Couplings
couplingOp = Couplings.instantaneousDirectCoupling()  # The only one who knows the coupling operation is the model itself!!!


# -------------------------------------------------------------------------------------
# ----------------- Dynamic Mean Field (a.k.a., reducedWongWang) ----------------------
# -------------------------------------------------------------------------------------
@jit(nopython=True)
def dfun(simVars, coupling, I_external):
    Se = simVars[0]; Si = simVars[1]; J = simVars[2]
    Se = utils.doClamping(Se); Si = utils.doClamping(Si)  # Clamping, needed in Deco2014 model and derivatives...

    couple = coupling.couple(Se)  # (SC @ Se)
    Ie = We*I0 + w * J_NMDA * Se + we * J_NMDA * couple - J * Si + I_external  # Eq for I^E (5). I_external = 0 => resting state condition.
    Ii = Wi*I0 + J_NMDA * Se - Si  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
    re = He(Ie)  # Calls He(Ie). r^E = H^E(I^E) in the paper (7)
    ri = Hi(Ii)  # Calls Hi(Ii). r^I = H^I(I^I) in the paper (8)
    dSe = -Se*B_e + alfa_e * t_glu * (1.-Se) * re/1000.  # divide by 1000 because we need milliseconds!
    dSi = -Si*B_i + alfa_i * t_gaba * (1.-Si) * ri/1000.
    dJ = gamma * ri/1000. * (re-rho)/1000.  # local inhibitory plasticity
    return np.stack((dSe, dSi, dJ)), np.stack((Ie, re, J))


# ==========================================================================
# ==========================================================================
# ==========================================================================
