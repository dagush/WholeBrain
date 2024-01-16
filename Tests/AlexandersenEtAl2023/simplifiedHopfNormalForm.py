# ==========================================================================
# ==========================================================================
# ==========================================================================
# Simplified Hopf normal form
#
# Original code by Christoffer Alexandersen
#
# [Alexandersen 2023] Alexandersen Christoffer G., de Haan Willem, Bick Christian and Goriely Alain (2023)
# A multi-scale model explains oscillatory slowing and neuronal hyperactivity in Alzheimer’s disease
# J. R. Soc. Interface
# https://doi.org/10.1098/rsif.2022.0607
#
# The dynamic equations were taken from [Alexandersen 2023]:
# {\dot z}_i=\mathcal F(z_i)+\kappa{\textstyle\text{tanh}}{(\sum_{\:j=1}^Nw_{ij}x_j(t-\tau_{ij}))},
# \mathcal F(z_i)=z_i{(\lambda-\frac{x_i^2}{a_i^2}-\frac{y_i^2}{b_i^2})}-\omega_iy_i{(\frac{a_i}{b_i})}+\mathrm i\:\omega_ix_i{(\frac{b_i}{a_i})}.
#
#
# refactored by Gustavo Patow
#
# ==========================================================================
# ==========================================================================
import numpy as np
from numba import jit

import WholeBrain.Models.Couplings as Couplings

print("Going to use the simplified Hopf normal form model...")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    dfun.recompile()
    pass

# ==========================================================================
# Simplified Hopf normal form Constants
# --------------------------------------------------------------------------
# Values taken from [Alexandersen 2023]
decay = None
# decay0 = 0
# decay1 = 1
baseDecay = -0.01
# inter_mat = None  # set to np.ones((N,N)) at initSim(N)
baseH = 5
h = None

a = -0.5        # Local bifurcation parameter
b = 0.5
omega = 0.3     # Angular frequency [Hz]
G = 1.          # Coupling constant, kappa in the original code
SC = None       # Structural Connectivity (should be provided externally)
delays = None   # Time delays
y0 = None

# --------------------------------------------------------------------------
# Simulation variables
initialValueX = 0.001
initialValueY = 0.001
def initSim(N):
    global decay, h  # SCT, inter_mat
    # SCT = SC.T
    # inter_mat = np.ones((N,N))
    decay = baseDecay * np.ones(N)
    h = np.ones(N) * baseH

    if y0 is not None:
        x = y0[0::2]  # remember that values are interleaved
        y = y0[1::2]
    else:
        x = initialValueX * np.ones(N)  # Initialize x
        y = initialValueY * np.ones(N)  # Initialize y
    return np.stack((x, y))


# Variables of interest, needed for bookkeeping tasks...
# x = y = None
def numObsVars():  # Returns the number of observation vars used, here xn and rn
    return 2


# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global G, SC, a, b, omega, y0, delays, baseDecay, baseH
    if 'we' in modelParms:
        G = modelParms['we']
    if 'SC' in modelParms:
        SC = modelParms['SC']
    if 'a' in modelParms:
        a = modelParms['a']
    if 'b' in modelParms:
        b = modelParms['b']
    if 'omega' in modelParms:
        omega = modelParms['omega']
    if 'y0' in modelParms:
        y0 = modelParms['y0']
    if 'delays' in modelParms:
        delays = modelParms['delays']
    if 'baseDecay' in modelParms:
        baseDecay = modelParms['baseDecay']
    if 'baseH' in modelParms:
        baseH = modelParms['baseH']


def getParm(parmName):
    if 'we' in parmName:
        return G
    if 'SC' in parmName:
        return SC
    if 'delays' in parmName:
        return delays
    return None


# -----------------------------------------------------------------------------
# ----------------- Simplified Hopf bifurcation model ----------------------
# -----------------------------------------------------------------------------

# ----------------- Coupling ----------------------
from WholeBrain.Models.Couplings import instantaneousDirectCoupling  # The only one who knows the coupling operation is the model itself!!!
couplingOp = instantaneousDirectCoupling()


# ----------------- Model ----------------------
@jit(nopython=True)
def dfun(simVars, coupling, I_external):
    x = simVars[0]; y = simVars[1]

    afferent_input = G * coupling.couple(x)  # np.dot(inter_mat * SCT, x)

    # dynamics of node k
    dx = decay * x - omega * y * (a/b) - x * (x**2/a**2 + y**2/b**2) + h * np.tanh(afferent_input)
    dy = decay * y + omega * x * (b/a) - y * (x**2/a**2 + y**2/b**2)

    return np.stack((dx, dy)), np.stack((x, y))


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF