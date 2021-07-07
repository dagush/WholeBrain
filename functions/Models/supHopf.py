# ==========================================================================
# ==========================================================================
# ==========================================================================
# Normal form of a supercritical Hopf bifurcation
#
# General neural mass model known as the normal form of a Hopf bifurcation
# (also known as Landau-Stuart Oscillators), which is the canonical model
# for studying the transition from noisy to oscillatory dynamics.
#
#     .. [Kuznetsov_2013] Kuznetsov, Y.A. *Elements of applied bifurcation theory.* Springer Sci & Business
#     Media, 2013, vol. 112.
#
#     .. [Deco_2017]  Deco, G., Kringelbach, M.L., Jirsa, V.K. et al.
#     The dynamics of resting fluctuations in the brain: metastability and its
#     dynamical cortical core. Sci Rep 7, 3095 (2017).
#     https://doi.org/10.1038/s41598-017-03073-5
#
#
# The supHopf model describes the normal form of a supercritical Hopf bifurcation in Cartesian coordinates. This
# normal form has a supercritical bifurcation at $a = 0$ with a the bifurcation parameter in the model. So for
# $a < 0$, the local dynamics has a stable fixed point, and for $a > 0$, the local dynamics enters in a
# stable limit cycle.
#
# The dynamic equations were taken from [Deco_2017]:
#
#         \dot{x}_{i} &= (a_{i} - x_{i}^{2} - y_{i}^{2})x_{i} - omega{i}y_{i} \\
#         \dot{y}_{i} &= (a_{i} - x_{i}^{2} - y_{i}^{2})y_{i} + omega{i}x_{i}
#
#     where a is the local bifurcation parameter and omega the angular frequency.
#
# ==========================================================================
# ==========================================================================
import numpy as np
from numba import jit

print("Going to use the supercritical Hopf bifurcation neuronal model...")

def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    dfun.recompile()
    # pass

# ==========================================================================
# supercritical Hopf bifurcation Constants
# --------------------------------------------------------------------------
# Values taken from [Deco_2017]
a = -0.5        # Local bifurcation parameter
omega = 0.3    # Angular frequency [Hz]
G = 1.          # Coupling constant
beta = 0.02     # Gaussian noise standard deviation
SC = None       # Structural connectivity (should be provided externally)
SCT = None      # Transposed Structural connectivity (should be provided externally)
ink = None      # Convenience parameters: SCT.sum(axis=1)

# --------------------------------------------------------------------------
# Simulation variables
def initSim(N):
    global SCT, ink
    SCT = SC.T
    ink = SCT.sum(axis=1)
    x = 0.001 * np.zeros(N)  # Initialize x
    y = 0.001 * np.zeros(N)  # Initialize y
    return np.array([x, y])


# Variables of interest, needed for bookkeeping tasks...
# x = y = None
def numObsVars():  # Returns the number of observation vars used, here xn and rn
    return 2


# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global G, SC, a
    if 'we' in modelParms:
        G = modelParms['we']
    if 'SC' in modelParms:
        SC = modelParms['SC']
    if 'a' in modelParms:
        a = modelParms['a']


def getParm(parmList):
    if 'we' in parmList:
        return G
    return None


# ----------------- supercritical Hopf bifurcation model ----------------------
@jit(nopython=True)
def dfun(simVars, p):  # p is the stimulus...?
    # global x, y
    N = SC.shape[0]
    x = simVars[0]; y = simVars[1]
    noiseX = np.random.normal(0, beta, N)
    noiseY = np.random.normal(0, beta, N)
    xcoup = np.dot(SCT,x) - ink * x  # sum(Cij*xi) - sum(Cij)*xj
    ycoup = np.dot(SCT,y) - ink * y  #
    # suma = wC*z - sumC.*z
    # zz = z(:,end:-1:1)  # <- flipped z, because (x.*x + y.*y)
    # dz = a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma  # <- using complex numbers z instead of x and y...
    dx = (a - x**2 - y**2) * x - omega * y + G * xcoup + noiseX
    dy = (a - x**2 - y**2) * y + omega * x + G * ycoup + noiseY
    return np.stack((dx, dy)), np.stack((x, y))


# ==========================================================================
# ==========================================================================
# ==========================================================================
