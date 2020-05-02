# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This prog. optimizes the strengh of the feedback inhibition of the FIC model
# for a given global coupling (G)
# Returns the feedback inhibition (J) (and the the steady states if wanted).
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
# G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
# How local excitation-inhibition ratio impacts the whole brain dynamics
# J. Neurosci., 34 (2014), pp. 7886-7898
# http://www.jneurosci.org/content/34/23/7886.long
#
# Adrian Ponce-Alvarez. Refactoring (& Python translation) by Gustavo Patow
# --------------------------------------------------------------------------
import numpy as np
from pathlib import Path
import scipy.io as sio
integrator = None  # functions.Integrator_EulerMaruyama

veryVerbose = False
verbose = True

print("Going to use the Balanced J9 (FIC) mechanism...")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    # print("\n\nRecompiling signatures!!!")
    integrator.recompileSignatures()


def updateJ(N, tmax, delta, curr, J):
    tmin = 1000 if (tmax>1000) else int(tmax/10)
    currm = np.mean(curr[tmin:tmax, :], 0)  # takes the mean of all xn values along dimension 1...
    # This is the "averaged level of the input of the local excitatory pool of each brain area,
    # i.e., I_i^{(E)}" in the text (pp 7889, right column, subsection "FIC").
    flag = 0
    if veryVerbose: print()
    if veryVerbose: print("[", end='')
    for n in range(N):
        if np.abs(currm[n] + 0.026) > 0.005:  # if currm_i < -0.026 - 0.005 or currm_i > -0.026 + 0.005 (a tolerance)
            if currm[n] < -0.026:  # if currm_i < -0.026
                J[n] = J[n] - delta[n]  # down-regulate
                delta[n] = delta[n] - 0.001
                if delta[n] < 0.001:
                    delta[n] = 0.001
                if veryVerbose: print("v", end='')
            else:  # if currm_i >= -0.026 (in the paper, it reads =)
                J[n] = J[n] + delta[n]  # up-regulate
                if veryVerbose: print("^", end='')
        else:
            flag = flag + 1
            if veryVerbose: print("-", end='')
    if veryVerbose: print("]")
    return flag


# =====================================
# =====================================
# Computes the optimum of the J_i for a given structural connectivity matrix C and
# a coupling coefficient G, which should be set externally directly at the neuronal model.
def JOptim(C, warmUp = False):
    N = C.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]

    # simulation fixed parameters:
    # ----------------------------
    dt = 0.1
    tmax = 10000

    # integrator.neuronalModel.we = we
    integrator.neuronalModel.initJ(N)

    # initialization:
    # -------------------------
    integrator.neuronalModel.SC = C
    # integrator.initBookkeeping(N, tmax)
    delta = 0.02 * np.ones(N)

    if verbose:
        print()
        print("we=", integrator.neuronalModel.we)  # display(we)
        print("  Trials:", end=" ", flush=True)

    ### Balance (greedy algorithm)
    # note that we used stochastic equations to estimate the JIs
    # Doing that gives more stable solutions as the JIs for each node will be
    # a function of the variance.
    bestJ = np.ones(N); bestJCount = -1; bestTrial = -1
    for k in range(5000):  # 5000 trials
        # integrator.resetBookkeeping()
        Tmaxneuronal = int((tmax+dt))  # (tmax+dt)/dt, but with steps of 1 unit...
        recompileSignatures()
        if warmUp:
            curr_xn = integrator.warmUpAndSimulate(dt, Tmaxneuronal)[:,0,:]  # take the xn component of the observation variables...
        else:
            curr_xn = integrator.simulate(dt, Tmaxneuronal)[:,0,:]  # take the xn component of the observation variables...
        if verbose: print(k, end='', flush=True)

        currm = curr_xn - integrator.neuronalModel.be/integrator.neuronalModel.ae  # be/ae==125./310. Records currm_i = xn-be/ae (i.e., I_i^E-b_E/a_E in the paper) for each i (1 to N)
        flagJ = updateJ(N, tmax, delta, currm, integrator.neuronalModel.J)
        if verbose: print("({})".format(flagJ), end='', flush=True)
        if flagJ > bestJCount:
            bestJCount = flagJ
            bestJ = integrator.neuronalModel.J
            bestTrial = k
            if verbose: print(' New min!!!', end='', flush=True)
        if flagJ == N:
            if verbose: print('Out !!!', flush=True)
            break
        else:
            if verbose: print(', ', end='', flush=True)

    if verbose: print("\nFinal (we={}): {} trials, with {}/{} nodes solved at trial {}\n".format(integrator.neuronalModel.we, k, bestJCount, N, bestTrial))
    if verbose: print('DONE!') if flagJ == N else print('FAILED!!!')
    return bestJ, bestJCount


# =====================================
# =====================================
# Auxiliary function to simplify work: if it was computed, load it. If not, compute (and save) it!
baseName = "Data_Produced/test_J_Balance_we{}.mat"
def Balance_J9(we, C, warmUp = False):
    fileName = baseName.format(we)
    # ==== J is calculated this only once, then saved
    integrator.neuronalModel.we = we
    if not Path(fileName).is_file():
        print("Computing {} !!!".format(fileName))
        bestJ, nodeCount = JOptim(C, warmUp=warmUp)  # This is the Feedback Inhibitory Control
        sio.savemat(fileName, {'we': we, 'J': integrator.neuronalModel.J})
    else:
        # ==== J can be calculated only once and then load J_Balance J
        print("Loading {} !!!".format(fileName))
        bestJ = sio.loadmat(fileName)['J']
    integrator.neuronalModel.J = bestJ.flatten()

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF