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

print("Going to use the supercritical Hopf bifurcation neuronal model...")

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
    return [x, y]

# Variables of interest, needed for bookkeeping tasks...
x = y = None

# ----------------- supercritical Hopf bifurcation model ----------------------
def dfun(simVars, p):  # p is the stimulus...?
    global x, y
    N = SC.shape[0]
    [x, y] = simVars
    noiseX = np.random.normal(0, beta, N)
    noiseY = np.random.normal(0, beta, N)
    xcoup = np.dot(SCT,x) - ink * x  # sum(Cij*xi) - sum(Cij)*xj
    ycoup = np.dot(SCT,y) - ink * y  #
    dx = (a - x**2 - y**2) * x - omega * y + G * xcoup + noiseX
    dy = (a - x**2 - y**2) * y + omega * x + G * ycoup + noiseY
    return [dx, dy]


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Bookkeeping variables of interest...
# --------------------------------------------------------------------------
curr_x = None
curr_y = None
nn = 0


def initBookkeeping(N, tmax):
    global curr_x, curr_y, nn
    curr_x = np.zeros((int(tmax/ds), N))
    curr_y = np.zeros((int(tmax/ds), N))
    nn = 0


def resetBookkeeping():
    global nn
    nn = 0


ds = 1  # downsampling stepsize
def recordBookkeeping(t):
    global curr_x, curr_y, nn
    t2 = int(t * 100000)
    ds2 = int(ds * 100000)
    if np.mod(t2, ds2) == 0:
        # print(t,ds,nn)
        curr_x[nn] = x
        curr_y[nn] = y
        nn = nn + 1


def returnBookkeeping():
    return curr_x, curr_y


# ==========================================================================
# ==========================================================================
# ==========================================================================
