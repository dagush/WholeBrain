# ==========================================================================
# ==========================================================================
# ==========================================================================
# Normal form of a supercritical Hopf bifurcation
#
# General neural mass model known as the normal form of a Hopf bifurcation
# (also known as Landau-Stuart Oscillators), which is the canonical model
# for studying the transition from noisy to oscillatory dynamics.
#
#     .. [Kuznetsov_2013] Kuznetsov, Y.A. "Elements of applied bifurcation theory", Springer Sci & Business
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
from numba import int32, double   # import the types
from numba.experimental import jitclass

print("Going to use the supercritical Hopf bifurcation neuronal model...")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    dfun.recompile()
    pass


# ==========================================================================
# supercritical Hopf bifurcation Constants
# --------------------------------------------------------------------------
# Values taken from [Deco_2017]
a = -0.5        # Local bifurcation parameter
omega = 0.3     # Angular frequency [Hz]
G = 1.          # Coupling constant
SC = None       # Structural Connectivity (should be provided externally)
SCT = None      # Transposed Structural Connectivity (we initialize it at initSim(N))
ink = None      # Convenience parameters: sum_i(Cij) = SCT.sum(axis=1) (we initialize it at initSim(N))
conservative = True  # Select between Conservative and Non-conservative (remove the x_j - x_i dependence, i.e., ink=0)


# --------------------------------------------------------------------------
# Simulation variables
initialValueX = 0.001
initialValueY = 0.001
def initSim(N):
    global SCT, ink
    SCT = SC.T
    if conservative:
        ink = SCT.sum(axis=1)   # Careful: component 2 in Matlab is component 1 in Python
    else:
        ink = 0
    x = initialValueX * np.ones(N)  # Initialize x
    y = initialValueY * np.ones(N)  # Initialize y
    return np.array([x, y])


# Variables of interest, needed for bookkeeping tasks...
# x = y = None
def numObsVars():  # Returns the number of observation vars used, here xn and rn
    return 2


# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global G, SC, a, omega
    if 'we' in modelParms:
        G = modelParms['we']
    if 'G' in modelParms:
        G = modelParms['G']
    if 'SC' in modelParms:
        SC = modelParms['SC']
        initSim(SC.shape[0])
    if 'a' in modelParms:
        a = modelParms['a']
    if 'omega' in modelParms:
        omega = modelParms['omega']


def getParm(parmList):
    if 'we' in parmList or 'G' in parmList:
        return G
    if 'SC' in parmList:
        return SC
    return None


# -----------------------------------------------------------------------------
# ----------------- supercritical Hopf bifurcation model ----------------------
# -----------------------------------------------------------------------------

# ----------------- Coupling ----------------------
@jitclass([('SCT', double[:, :]),
           ('ink', double[:])])
class instantaneousDifferenceCoupling:
    def __init__(self):
        self.SCT = np.empty((1,1))
        self.ink = np.empty((1))

    def setParms(self, SC):
        self.SCT = SC.T
        self.ink = SCT.sum(axis=1)   # Careful: component 2 in Matlab is component 1 in Python

    def couple(self, x):
        return np.dot(self.SCT, x) - self.ink * x


couplingOp = instantaneousDifferenceCoupling()


# ----------------- Model ----------------------
@jit(nopython=True)
def dfun(simVars, coupling, I_external):
    x = simVars[0]; y = simVars[1]
    pC = I_external + 0j
    # --------------------- From Gus' original code:
    # First, we need to compute the term (in pseudo-LaTeX notation):
    #       G Sum_i SC_ij (x_i - x_j) =
    #       G (Sum_i SC_ij x_i + Sum_i SC_ij x_j) =
    #       G ((Sum_i SC_ij x_i) + (Sum_i SC_ij) x_j)   <- adding some unnecessary parenthesis.
    # This is implemented in Gus' code as:
    #       wC = we * Cnew;  # <- we is G in the paper, Cnew is SC -> wC = G * SC
    #       sumC = repmat(sum(wC, 2), 1, 2);  # <- for sum Cij * xj == sum(G*SC,2)
    # Thus, we have that:
    #       suma = wC*z - sumC.*z                 # this is sum(Cij*xi) - sum(Cij)*xj, all multiplied by G
    #            = G * SC * z - sum(G*SC,2) * z   # Careful, component 2 in Matlab is component 1 in Python...
    #            = G * (SC*z - sum(SC,2)*z)
    # And now the rest of it...
    # Remember that, in Gus' code,
    #       omega = repmat(2*pi*f_diff',1,2);
    #       omega(:,1) = -omega(:,1);
    # so here I will call omega(1)=-omega, and the other component as + omega
    #       zz = z(:,end:-1:1)  # <- flipped z, because (x.*x + y.*y)     # Thus, this zz vector is (y,x)
    #       dz = a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma               # original formula in the code, using complex numbers z instead of x and y...
    #          = zz * omega   +  z  * (a -  z.* z  - zz.* zz) + suma =    # I will be using vector notation here to simplify ASCII formulae... ;-)
    #          = (y)*(-omega) + (x) * (a - (x)*(x) - (y)*(y)) + suma      # here, (x)*(x) should actually be (x) * (x,y)
    #          =  x *(+omega)    y          y * y     x * x               #        y   y                     (y)
    # ---------------------
    # Calculate the input to nodes due to couplings
    xcoup = coupling.couple(x)  # np.dot(SCT,x) - ink * x  # this is sum(Cij*xi) - sum(Cij)*xj
    ycoup = coupling.couple(y)  # np.dot(SCT,y) - ink * y  #
    # Integration step
    dx = (a - x**2 - y**2) * x - omega * y + G * xcoup + pC.real
    dy = (a - x**2 - y**2) * y + omega * x + G * ycoup + pC.imag
    return np.stack((dx, dy)), np.stack((x, y))


# ==========================================================================
# ==========================================================================
# ==========================================================================
