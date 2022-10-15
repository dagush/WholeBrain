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
# [DecoEtAl2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#     How local excitation-inhibition ratio impacts the whole brain dynamics
#     J. Neurosci., 34 (2014), pp. 7886-7898
#     http://www.jneurosci.org/content/34/23/7886.long
#
# Adrian Ponce-Alvarez. Refactoring (& Python translation) by Gustavo Patow
# --------------------------------------------------------------------------
import numpy as np
from pathlib import Path
import scipy.io as sio
import multiprocessing as mp
from WholeBrain.Utils.decorators import loadOrCompute

integrator = None  # WholeBrain.Integrator_EulerMaruyama

veryVerbose = False
verbose = True

print("Going to use the Balanced J9 (FIC) mechanism...")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    # print("\n\nRecompiling signatures!!!")
    integrator.recompileSignatures()


min_largest_distance = np.inf
slow_factor = 1.0
def updateJ_N(N, tmax, delta, curr, J):  # 2nd version of updateJ
    tmin = 1000 if (tmax>1000) else int(tmax/10)
    currm = np.mean(curr[tmin:tmax, :], 0)  # takes the mean of all xn values along dimension 1...
    # This is the "averaged level of the input of the local excitatory pool of each brain area,
    # i.e., I_i^{(E)}" in the text (pp 7889, right column, subsection "FIC").
    flag = 0
    if veryVerbose: print()
    if veryVerbose: print("[", end='')
    # ===========================================================
    global min_largest_distance, slow_factor
    distance = np.full((N,), 10.0)
    num_above_error = 0
    largest_distance = 0
    # total_error = 0.0
    Si = np.zeros((N,))
    for i in range(N):
        # ie_100 = curr[i]  # d_raw[-100:-1, 1, i, 0]  # I_e
        # ie = currm[i]  # np.average(ie_100)
        d = currm[i] + 0.026  # ie - be_ae + 0.026
        distance[i] = d
        d_abs = abs(d)
        if largest_distance < d_abs:
            largest_distance = d_abs
        # Si[i] = np.average(d_raw[-100:-1, 2, i, 0])  # S_i
        # error_i = d*d
        # error[i] = error_i
        # total_error += error_i

    if largest_distance < min_largest_distance:
        min_largest_distance = largest_distance
    else:
        slow_factor *= 0.5

    for i in range(N):
        d = distance[i]  # currm[i] + 0.026
        d_abs = np.abs(d)
        if d_abs > 0.005:  # if currm_i < -0.026 - 0.005 or currm_i > -0.026 + 0.005 (a tolerance)
            num_above_error += 1
            delta_i = slow_factor * d_abs / 0.1  # Si[i]  # 0.003 * abs(d + 0.026) / 0.026
            if delta_i < 0.005:
                delta_i = 0.005
            delta[i] = np.sign(d) * delta_i
        else:
            delta[i] = 0.0
        J[i] = J[i] + delta[i]
    if veryVerbose: print("]")
    return N - num_above_error


def updateJ(N, tmax, delta, curr, J):  # This is the original method by Gus, from the paper...
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
use_N_algorithm = True
def JOptim(N, warmUp = False):
    # simulation fixed parameters:
    # ----------------------------
    dt = 0.1
    tmax = 10000

    # initialization:
    # -------------------------
    delta = 0.02 * np.ones(N)
    # A couple of initializations, needed only for updateJ_2
    global min_largest_distance, slow_factor; min_largest_distance = np.inf; slow_factor = 1.0

    if verbose:
        print("  Trials:", end=" ", flush=True)

    ### Balance (greedy algorithm)
    # note that we used stochastic equations to estimate the JIs
    # Doing that gives more stable solutions as the JIs for each node will be
    # a function of the variance.
    currJ = np.ones(N)
    bestJ = np.ones(N); bestJCount = -1; bestTrial = -1
    for k in range(5000):  # 5000 trials
        # integrator.resetBookkeeping()
        Tmaxneuronal = int((tmax+dt))  # (tmax+dt)/dt, but with steps of 1 unit...
        integrator.neuronalModel.setParms({'J': currJ})
        recompileSignatures()
        if warmUp:
            curr_xn = integrator.warmUpAndSimulate(dt, Tmaxneuronal)[:,0,:]  # take the xn component of the observation variables...
        else:
            curr_xn = integrator.simulate(dt, Tmaxneuronal)[:,0,:]  # take the xn component of the observation variables...
        if verbose: print(k, end='', flush=True)

        currm = curr_xn - integrator.neuronalModel.getParm('be')/integrator.neuronalModel.getParm('ae')  # be/ae==125./310. Records currm_i = xn-be/ae (i.e., I_i^E-b_E/a_E in the paper) for each i (1 to N)
        if use_N_algorithm:
            flagJ = updateJ_N(N, tmax, delta, currm, currJ)  # Nacho's method... ;-)
        else:
            flagJ = updateJ(N, tmax, delta, currm, currJ)  # Gus' method, the one from [DecoEtAl2014]

        if verbose: print("({})".format(flagJ), end='', flush=True)
        if flagJ > bestJCount:
            bestJCount = flagJ
            bestJ = currJ
            bestTrial = k
            if verbose: print(' New min!!!', end='', flush=True)
        if flagJ == N:
            if verbose: print('Out !!!', flush=True)
            break
        else:
            if verbose: print(', ', end='', flush=True)

    if verbose: print("Final (we={}): {} trials, with {}/{} nodes solved at trial {}".format(integrator.neuronalModel.getParm('we'), k, bestJCount, N, bestTrial))
    if verbose: print('DONE!') if flagJ == N else print('FAILED!!!')
    return bestJ, bestJCount


# =====================================
# =====================================
# Auxiliary WholeBrain to simplify work: if it was computed, load it. If not, compute (and save) it!
@loadOrCompute
def Balance_J9(we, N, warmUp=False): # Computes (and sets) the optimized J for Feedback Inhibition Control [DecoEtAl2014]
    print("we={} (use {} optim)".format(we, "N" if use_N_algorithm else "A"))
    integrator.neuronalModel.setParms({'we': we})
    bestJ, nodeCount = JOptim(N, warmUp=warmUp)  # This is the Feedback Inhibitory Control
    integrator.neuronalModel.setParms({'J': bestJ.flatten()})
    return {'we': we, 'J': bestJ.flatten()}


def Balance_AllJ9(C, WEs,  # wStart=0, wEnd=6+0.001, wStep=0.05,
                  baseName=None,
                  parallel=False):
    # all tested global couplings (G in the paper):
    # integrator.neuronalModel.setParms({'SC': C})
    N = C.shape[0]
    result = {}
    # if not parallel:
    for we in WEs:  # iterate over the weight range (G in the paper, we here)
        balance = Balance_J9(we, N, baseName.format(np.round(we, decimals=2)))['J'].flatten()
        result[we] = {'we': we, 'J': balance}
    return result

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
