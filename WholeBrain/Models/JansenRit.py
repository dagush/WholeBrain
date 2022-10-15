# ==========================================================================
# ==========================================================================
# ==========================================================================
#
# The Jansen and Rit is a biologically inspired mathematical framework
# originally conceived to simulate the spontaneous electrical activity of
# neuronal assemblies, with a particular focus on alpha activity, for instance,
# as measured by EEG. Later on, it was discovered that in addition to alpha
# activity, this model was also able to simulate evoked potentials.
#
#     .. [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
#         visual evoked potential generation in a mathematical model of
#         coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.
#
#     .. [J_1993] Jansen, B., Zouridakis, G. and Brandt, M., *A
#         neurophysiologically-based mathematical model of flash visual evoked
#         potentials*
#
# The dynamic equations were taken from [JR_1995]:
#
#      \dot{y_0} &= y_3 \\
#      \dot{y_3} &= A a\,S[y_1 - y_2] - 2a\,y_3 - 2a^2\, y_0 \\
#      \dot{y_1} &= y_4\\
#      \dot{y_4} &= A a \,[p(t) + \alpha_2 J S[\alpha_1 J\,y_0]+ c_0]
#                  -2a\,y - a^2\,y_1 \\
#      \dot{y_2} &= y_5 \\
#      \dot{y_5} &= B b (\alpha_4 J\, S[\alpha_3 J \,y_0]) - 2 b\, y_5
#                  - b^2\,y_2 \\
#      S[v] &= \frac{2\, \nu_{max}}{1 + \exp^{r(v_0 - v)}}
#
#  can be any arbitrary function, including white noise or
#  random numbers taken from a uniform distribution, representing a pulse
#  density with an amplitude varying between 120 and 320
#
#  For Evoked Potentials, a transient component of the input,
#  representing the impulse density attribuable to a brief visual input is
#  applied. Time should be in seconds.
#
#      p(t) = q\,(\frac{t}{w})^n \, \exp{-\frac{t}{w}} \\
#      q = 0.5 \\
#      n = 7 \\
#      w = 0.005 [s]
#
# ==========================================================================
# ==========================================================================
import numpy as np
from numba import jit

print("Going to use the Jansen-Rit neuronal model...")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    # initSim.recompile()
    sigm.recompile()
    dfun.recompile()
    pass


# ==========================================================================
# Jansen and Rit Model Constants
# --------------------------------------------------------------------------
# Values taken from [JR_1995]
A = 3.25        # Maximum amplitude of EPSP [mV]. Also called average synaptic gain.
a = 100.        # Time constant of passive membrane and all other spatially distributed delays in the
                # dendritic network [ms^-1]. Also called average synaptic time constant. (== 1./10e-3)
B = 22          # Maximum amplitude of IPSP [mV]. Also called average synaptic gain.
b = 50.         # Time constant of passive membrane and all other spatially distributed delays in the
                # dendritic network [ms^-1]. Also called average synaptic time constant. (== 1./20e-3)
v0 = 6.0        # Firing threshold (PSP) for which a 50% firing rate is achieved. In other words, it is the value of
                # the average membrane potential corresponding to the inflection point of the sigmoid [mV]. The usual
                # value for this parameter is 6.0.
e_0 = 2.5       # Determines the maximum firing rate of the neural population [s^-1].
r = 0.56        # Steepness of the sigmoidal transformation [mV^-1].
C = 135.0       # Average number of synapses between populations.
a_1 = 1.0       # C1 = a_1 * C. Average probability of synaptic contacts in the feedback excitatory loop.
a_2 = 0.8       # C2 = a_2 * C. Average probability of synaptic contacts in the slow feedback excitatory loop.
a_3 = 0.25      # C3 = a_3 * C. Average probability of synaptic contacts in the feedback inhibitory loop.
a_4 = 0.25      # C4 = a_4 * C. Average probability of synaptic contacts in the slow feedback inhibitory loop.
we = 2.1
SC = None       # Structural connectivity (should be provided externally)

# --------------------------------------------------------------------------
# Simulation variables
# @jit(nopython=True)
def initSim(N):
    # y0 = 0.001 * np.zeros(N)  # Initialize y0
    # y1 = 0.001 * np.zeros(N)  # Initialize y1
    # y2 = 0.001 * np.zeros(N)  # Initialize y2
    # y3 = 0.001 * np.zeros(N)  # Initialize y3
    # y4 = 0.001 * np.zeros(N)  # Initialize y4
    # y5 = 0.001 * np.zeros(N)  # Initialize y5
    # return [y0, y1, y2, y3, y4, y5]
    y0_5 = np.zeros((6, N))  # Here we initialize with np.zeros, but other models use np.ones
    return y0_5


# Variables of interest, needed for bookkeeping tasks...
# v = None
# @jit(nopython=True)
def numObsVars():
    return 1


# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global we, C, SC, A, B, a, b, ds
    if 'we' in modelParms:
        we = modelParms['we']
    if 'SC' in modelParms:
        SC = modelParms['SC']
    if 'C' in modelParms:
        C = modelParms['C']
    if 'A' in modelParms:
        A = modelParms['A']
    if 'B' in modelParms:
        B = modelParms['B']
    if 'a' in modelParms:
        a = modelParms['a']
    if 'b' in modelParms:
        b = modelParms['b']
    if 'ds' in modelParms:
        ds = modelParms['ds']


def getParm(parmList):
    if 'we' in parmList:
        return we
    if 'SC' in parmList:
        return SC
    return None


# ----------------- Jansen and Rit model ----------------------
@jit(nopython=True)
def sigm(y):
    return 2.0 * e_0 / (1.0 + np.exp(r * (v0 - y)))

@jit(nopython=True)
def dfun(simVars, p):  # p is the stimulus
    # global v
    y0 = simVars[0]; y1=simVars[1]; y2=simVars[2]; y3=simVars[3]; y4=simVars[4]; y5=simVars[5]
    v = y1 - y2
    dy0 = y3
    dy3 = A * a * sigm(y1-y2) - 2.0 * a * y3 - a**2 * y0
    dy1 = y4
    dy4 = A * a * (p + we * SC @ sigm(v) + a_2*C * sigm(a_1*C*y0)) - 2.0 * a * y4 - a**2 * y1
    dy2 = y5
    dy5 = B * b * (a_4*C * sigm(a_3*C*y0)) - 2.0 * b * y5 - b**2 * y2
    return np.stack((dy0, dy1, dy2, dy3, dy4, dy5)), np.stack((v,))


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
