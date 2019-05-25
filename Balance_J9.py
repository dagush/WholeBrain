#
#
# This prog. optimizes the strengh of the feedback inhibition of the FIC model
# for varying global couplings (G)
# Saves the steady states and the feedback inhibition (J).
#
#
# For an isolated node, an input to the excitatory pool equal to I_i^E - b_E/a_E = -0.026;
# i.e., slightly inhibitory dominated, leads to a firing rate equal to 3.0631 Hz.
# Hence, in the large-scale model of interconnected brain areas,
# we aim to constraint in each brain area (i) the local feedback inhibitory weight Ji such
# that I_i^E - b_E/a_E = -0.026 is fulfilled (with a tolerance of +-0.005).
# To achieve this, we apply following procedure: we simulate during 5000 steps
# the system of stochastic differential DMF Equations and compute the averaged level of
# the input to the local excitatory pool of each brain area,
# then we upregulate the corresponding local feedback inhibition J_i = J_i + delta;
# otherwise, we downregulate J_i = J_i - delta.
# We recursively repeat this procedure until the constraint on the input
# to the local excitatory pool is fulfilled in all N brain areas.
#
# see:
# Deco et al. (2014) J Neurosci.
# http://www.jneurosci.org/content/34/23/7886.long
#
# Adrian Ponce-Alvarez & Gustavo Patow
# --------------------------------------------------------------------------

import numpy as np
import scipy.io as sio
from randn2 import randn2


# =====================================
# Matlab compatible definitions
def randn(N):
    ra = randn2(N)
    return ra.reshape(-1, 1)

# =====================================
# some monitoring global variables:
Se_init = None
Si_init = None
JI = None
kk = 0

# =====================================
# =====================================
def JOptim(we, C):
    global Se_init, Si_init, JI, kk
    N = C.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]

    # Model's fixed parameters:
    # ----------------------------
    dt = 0.1;
    tmax = 10000
    tspan = np.arange(0, tmax + dt, dt)  # tspan=0:dt:tmax;

    taon = 100
    taog = 10
    gamma = 0.641
    sigma = 0.01
    JN = 0.15
    J = np.ones((N, 1))
    I0 = 0.382  ##397;
    Jexte = 1.
    Jexti = 0.7
    w = 1.4

    # transfer functions:
    # transfert function: excitatory
    # ------------------------------------------------
    ae = 310
    be = 125
    de = 0.16

    def He(x):
        return (ae * x - be) / (1 - np.exp(-de * (ae * x - be)))

    # transfert function: inhibitory
    # ------------------------------------------------
    ai = 615
    bi = 177
    di = 0.087

    def Hi(x):
        return (ai * x - bi) / (1 - np.exp(-di * (ai * x - bi)))

    # initialization:
    # -------------------------
    curr = np.zeros((tmax, N))
    neuro_act = np.zeros((tmax, N))
    delta = 0.02 * np.ones((N, 1))

    ### Balance (greedy algorithm)
    # note that we used stochastic equations to estimate the JIs
    # Doing that gives more stable solutions as the JIs for each node will be
    # a function of the variance.

    print()
    print("we=", we)  # display(we)
    print("  Trials:", end=" ", flush=True)

    for k in range(5000):  # 5000 trials
        sn = 0.001 * np.ones((N, 1))  # Initialize sn (S^E in the paper)
        sg = 0.001 * np.ones((N, 1))  # Initialize sg (S^I in the paper)
        nn = 0
        j = 0
        for i in range(1, tspan.size):
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
            j = j + 1
            if j == 10:
                curr[nn] = xn.T - 125 / 310  # record currm_i = xn-be/ae (i.e., I_i^E-b_E/a_E in the paper) for each i (1 to N)
                # print("       ",nn)
                nn = nn + 1
                j = 0
        print(k, end=",", flush=True)

        currm = np.mean(curr[1000:tmax, :], 0)  # takes the mean of all xn values along dimension 1...
        # This is the "averaged level of the input of the local excitatory pool of each brain area,
        # i.e., I_i^{(E)}" in the text (pp 7889, right column, subsection "FIC").
        flag = 0
        for n in range(N):
            if np.abs(
                    currm[n] + 0.026) > 0.005:  # if currm_i < -0.026 - 0.005 or currm_i > -0.026 + 0.005 (a tolerance)
                if currm[n] < -0.026:  # if currm_i < -0.026
                    J[n] = J[n] - delta[n]  # downregulate
                    delta[n] = delta[n] - 0.001
                    if delta[n] < 0.001:
                        delta[n] = 0.001
                else:  # if currm_i >= -0.026 (in the paper, it reads =)
                    J[n] = J[n] + delta[n]  # upregulate
            else:
                flag = flag + 1
        if flag == N:
            print('Out!!!', flush=True)
            break

    # ==========================
    # Some monitoring info: initialization
    Se_init[:, kk] = sn[:, 0]  # Store steady states S^E (after many iterations/simulations)
    Si_init[:, kk] = sg[:, 0]  # Store steady states S^I
    JI[:, kk] = J[:, 0]  # Store feedback inhibition values J_i

    return J


