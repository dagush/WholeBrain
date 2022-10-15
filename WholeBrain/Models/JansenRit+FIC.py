# ==========================================================================
# ==========================================================================
# ==========================================================================
# A Modification of the basic Jansen-Rit model to include a Feedback Inhibition Control Mechanism:
#
#     .. [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
#         visual evoked potential generation in a mathematical model of
#         coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.
#
#     ..  G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#         How local excitation-inhibition ratio impacts the whole brain dynamics
#         J. Neurosci., 34 (2014), pp. 7886-7898
#         http://www.jneurosci.org/content/34/23/7886.long
#
# The dynamic equations were taken from [JR_1995]:
#
#      \dot{y_0} &= y_3 \\
#      \dot{y_3} &= A a\,S[y_1 - J_i y_2] - 2a\,y_3 - 2a^2\, y_0 \\
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

print("Going to use the Jansen-Rit + FIC neuronal model...")

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
C = np.identity(1)  # Structural connectivity

# --------------------------------------------------------------------------
# Simulation variables
def initSim(N):
    y0 = 0.001 * np.zeros(N)  # Initialize y0
    y1 = 0.001 * np.zeros(N)  # Initialize y1
    y2 = 0.001 * np.zeros(N)  # Initialize y2
    y3 = 0.001 * np.zeros(N)  # Initialize y3
    y4 = 0.001 * np.zeros(N)  # Initialize y4
    y5 = 0.001 * np.zeros(N)  # Initialize y5
    return [y0, y1, y2, y3, y4, y5]


J = None    # WARNING: In general, J must be initialized outside!
def initJ(N):  # A bit silly, I know...
    global J
    J = np.ones(N)


# Variables of interest, needed for bookkeeping tasks...
v = None

# ----------------- Jansen and Rit model ----------------------
def sigm(y):
    return 2.0 * e_0 / (1.0 + np.exp(r * (v0 - y)))

def dfun(simVars, p):  # p is the stimulus
    global v
    [y0, y1, y2, y3, y4, y5] = simVars
    # V is the variable of interest and it is y1 - y2
    v = y1 - y2
    # excitatory pyramidal cells
    dy0 = y3
    dy3 = A * a * sigm(y1-y2) - 2.0 * a * y3 - a**2 * y0
    # excitatory stellate cells
    dy1 = y4
    dy4 = A * a * (p + we * C @ sigm(v) + a_2*C * sigm(a_1*C*y0)) - 2.0 * a * y4 - a**2 * y1
    # inhibitory cells
    dy2 = y5
    dy5 = B * b * (J*a_4*C * sigm(a_3*C*y0)) - 2.0 * b * y5 - b**2 * y2
    return [dy0, dy1, dy2, dy3, dy4, dy5]


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Bookkeeping variables of interest...
# --------------------------------------------------------------------------
curr_v = None
nn = 0


def initBookkeeping(N, tmax):
    global curr_v, nn
    curr_v = np.zeros((int(tmax/ds), N))
    nn = 0


def resetBookkeeping():
    global nn
    nn = 0


ds = 1  # downsampling stepsize
def recordBookkeeping(t):
    global curr_v, nn
    t2 = int(t * 100000)
    ds2 = int(ds * 100000)
    if np.mod(t2, ds2) == 0:
        # print(t,ds,nn)
        curr_v[nn] = v
        nn = nn + 1


def returnBookkeeping():
    return curr_v


# ==========================================================================
# ==========================================================================
# ==========================================================================
