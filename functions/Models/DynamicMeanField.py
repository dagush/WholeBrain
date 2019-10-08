# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
# ==========================================================================
import numpy as np

print("Going to use the Dynamic Mean Field (DMF) model...")

# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) Model Constants
# --------------------------------------------------------------------------
taon = 100.
taog = 10.
gamma = 0.641
JN = 0.15       # [nA] NMDA current
I0 = 0.382  ##397;
Jexte = 1.
Jexti = 0.7
w = 1.4
we = 2.1        # Global coupling scaling

# transfer functions:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae=310.
be=125.
de=0.16
def phie(x):
    y = (ae*x-be)
    # if (y != 0):
    return y/(1.-np.exp(-de*y))
    # else:
    #     return 0

# transfer function: inhibitory
ai=615.
bi=177.
di=0.087
def phii(x):
    y = (ai*x-bi)
    # if (y != 0):
    return y/(1.-np.exp(-di*y))
    # else:
    #     return 0

# transfer functions used by the simulation...
He = phie
Hi = phii

# --------------------------------------------------------------------------
# Simulation variables
J = None    # WARNING: In general, J must be initialized outside!
def initSim(N):
    sn = 0.001 * np.ones((N, 1))  # Initialize sn (S^E in the paper)
    sg = 0.001 * np.ones((N, 1))  # Initialize sg (S^I in the paper)
    return [sn, sg]

def initJ(N):  # A bit silly, I know...
    global J
    J = np.ones((N, 1))


# Variables of interest, needed for bookkeeping tasks...
xn = None
rn = None


# ----------------- Dynamic Mean Field (a.k.a., reducedWongWang) ----------------------
def dfun(simVars, C, I_external):
    global xn, rn
    [sn, sg] = simVars
    xn = I0 * Jexte + w * JN * sn + we * JN * C @ sn - J * sg + I_external  # Eq for I^E (5). I_external = 0 => resting state condition.
    xg = I0 * Jexti + JN * sn - sg  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
    rn = He(xn)  # Calls He(xn). r^E = H^E(I^E) in the paper (7)
    rg = Hi(xg)  # Calls Hi(xg). r^I = H^I(I^I) in the paper (8)
    dsn = -sn / taon + (1. - sn) * gamma * rn / 1000.
    dsg = -sg / taog + rg / 1000.
    return [dsn, dsg]


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Bookkeeping variables of interest...
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


ds = 1  # downsampling stepsize
def recordBookkeeping(t):
    global curr_xn, curr_rn
    global nn
    if np.mod(t, ds) == 0:
        # print(t,ds,nn)
        curr_xn[nn] = xn.T - be/ae  # 125. / 310.  # record currm_i = xn-be/ae (i.e., I_i^E-b_E/a_E in the paper) for each i (1 to N)
        # Ut[:, nn] = xn[0:N]  #excitatory synaptic activity
        curr_rn[nn] = rn.T
        # Rt[:, nn] = rn       #excitatory firing rate
        nn = nn + 1


def returnBookkeeping():
    return curr_xn, curr_rn


# ==========================================================================
# ==========================================================================
# ==========================================================================
