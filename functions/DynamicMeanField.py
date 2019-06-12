
# Implementation of the Dynamic Mean Field model
#
# serotonin 2A receptor (5-HT_{2A}R)
# the neuronal gain function of the model is modulated
# by the 5-HT_{2A}R density
#
# From
# Whole-Brain Multimodal Neuroimaging Model Using Serotonin Receptor Maps Explains Non-linear Functional Effects of LSD
# Gustavo Deco, Josephine Cruzat, Joana Cabral, Gitte M. Knudsen, Robin L. Carhart-Harris, Peter C. Whybrow, Nikos K.
# Logothetis, and Morten L. Kringelbach
# Current Biology 28, 3065–3074
# October 8, 2018
#
# --------------------------------------------------------------------------

import numpy as np
import scipy.io as sio
from randn2 import randn2


# Matlab compatible definitions
# --------------------------------------------------------------------------
def randn(N):
    ra = randn2(N)
    return ra.reshape(-1, 1)


# Model Constants
# --------------------------------------------------------------------------
taon = 100
taog = 10
gamma = 0.641
sigma = 0.01
JN = 0.15
I0 = 0.382  ##397;
Jexte = 1.
Jexti = 0.7
w = 1.4


# Some working constants
# --------------------------------------------------------------------------
ds = 1  # downsampling stepsize


# transfer functions:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae=310
be=125
de=0.16
def phie(x):
    y = (ae*x-be)
    # if (y != 0):
    return y/(1-np.exp(-de*y))
    # else:
    #     return 0

# transfer function: inhibitory
ai=615
bi=177
di=0.087
def phii(x):
    y = (ai*x-bi)
    # if (y != 0):
    return y/(1-np.exp(-di*y))
    # else:
    #     return 0

# transfer functions used by the simulation...
He = phie
Hi = phii

# bookkeeping vars & methods
# --------------------------------------------------------------------------
curr_xn = None
curr_rn = None
nn = 0


def initBookkeeping(N, tmax):
    global curr_xn, curr_rn, nn
    curr_xn = np.zeros((tmax, N))
    curr_rn = np.zeros((tmax, N))
    nn = 0


def resetBookkeeping():
    global nn
    nn = 0


def recordBookkeeping(t, xn, rn):
    global curr_xn, curr_rn, nn
    if np.mod(t, ds) == 0:
        # print(t,ds,nn)
        curr_xn[nn] = xn.T - 125 / 310  # record currm_i = xn-be/ae (i.e., I_i^E-b_E/a_E in the paper) for each i (1 to N)
        # Ut[:, nn] = xn[0:N]  #excitatory synaptic activity
        curr_rn[nn] = rn.T
        # Rt[:, nn] = rn       #excitatory firing rate
        nn = nn + 1


# Euler Integration + DMF
# --------------------------------------------------------------------------
# Simulation variables
sn = None
sg = None
J = None    # WARNING: In general, J must be initialized outside!
def initSim(N):
    global sn, sg
    sn = 0.001 * np.ones((N, 1))  # Initialize sn (S^E in the paper)
    sg = 0.001 * np.ones((N, 1))  # Initialize sg (S^I in the paper)


def initJ(N):  # A bit silly, I know...
    global J
    J = np.ones((N, 1))


def simulate(dt, Tmaxneuronal, C, we):
    global sn, sg
    N = C.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    initSim(N)
    for t in np.arange(0, Tmaxneuronal, dt):
        xn = I0 * Jexte + w * JN * sn + we * JN * C @ sn - J * sg  # Eq for I^E (5). I_external = 0 => resting state condition.
        xg = I0 * Jexti + JN * sn - sg  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
        rn = He(xn)  # Calls He(xn). r^E = H^E(I^E) in the paper (7)
        rg = Hi(xg)  # Calls Hi(xg). r^I = H^I(I^I) in the paper (8)
        sn = sn + dt * (-sn / taon + (1 - sn) * gamma * rn / 1000) + np.sqrt(dt) * sigma * randn(N)  # Euler integration for S^E (9). Why sqrt(dt)?
        sn[sn > 1] = 1  # clamp values to 0..1
        sn[sn < 0] = 0
        sg = sg + dt * (-sg / taog + rg / 1000) + np.sqrt(dt) * sigma * randn(N)  # Euler integration for S^I (10). Why sqrt(dt)?
        sg[sg > 1] = 1  # clamp values to 0..1
        sg[sg < 0] = 0

        recordBookkeeping(t, xn, rn)


