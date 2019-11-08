# ================================================================================================================
#
# This prog. tests varios aspects of the DMF model...
#
# see:
# [DecoEtAl2014] Gustavo Deco, Adrián Ponce-Alvarez, Patric Hagmann, Gian Luca Romani, Dante Mantini and Maurizio
#           Corbetta, "How Local Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics" (2014), Journal of
#           Neuroscience 4 June 2014, 34 (23) 7886-7898; DOI: https://doi.org/10.1523/JNEUROSCI.5068-13.2014
#
# by Gustavo Patow
# ================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import functions.Models.DynamicMeanField as DMF
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = DMF
# import functions.Balance_J9 as Balance_J9
# Balance_J9.integrator = integrator


np.random.seed(42)  # Fix the seed for debug purposes...

# =================================================================================
# Sensibility calibration test... repeat the SAME experiment over and over again,
# and make a histogram out of the results. It should look like a gaussian...
# =================================================================================
# Results with 10000 samples:
def testMultipleTimes(trials, we, C):
    targetFreq = 3.  # We want the firing rate to be at 3Hz
    def distTo3Hz(curr):
        import functions.Utils.errorMetrics as error
        tmin = 1000 if (tmax>1000) else int(tmax/10)
        currm = np.mean(curr[tmin:tmax, :], 0)  # takes the mean of all xn values along dimension 1...
        # return np.abs(np.average(f)-targetFreq)
        return error.l2(currm, targetFreq)

    def errorFunc():
        print('starting sim ({}): '.format(n), end=' ', flush=True)
        DMF.resetBookkeeping()
        Tmaxneuronal = int((tmax+dt))  # (tmax+dt)/dt, but with steps of 1 unit...
        integrator.simulate(dt, Tmaxneuronal, C)
        currm = DMF.curr_rn
        err = distTo3Hz(currm)
        print('-> l^2 error = {}'.format(err))
        return err

    print("Testing, multiple {} times...".format(trials))
    dt = 0.1
    tmax = 10000
    N = C.shape[0]
    DMF.we = we
    DMF.initJ(N)
    DMF.initBookkeeping(N, tmax)
    results = np.zeros(trials)
    for n in range(trials):
        error = errorFunc()
        results[n] = error
    avg = np.average(results)
    std = np.std(results)
    print("Average=", avg, "std=", std)
    print("Min=", np.min(results), "Max=", np.max(results))

    # the histogram of the data...
    binwidth = 0.01
    bins = np.arange(np.min(results), np.max(results) + binwidth, binwidth)
    print('bins:', bins)
    n, bins, patches = plt.hist(results, bins=bins, facecolor='g')
    print('bins:', bins, 'n:', n, 'patches:', patches)
    print('results:', results)
    plt.xlabel('error')
    plt.ylabel('Probability')
    plt.title('Histogram of errors')
    plt.text(60, .025, '$\mu$={}, $\sigma$={}'.format(avg, std))
    #plt.xlim(40, 160)
    #plt.ylim(0, 0.03)
    #plt.grid(True)
    plt.show()


# ======================================================================
# ======================================================================
# ======================================================================
# def computeAllJs(C):
#     # all tested global couplings (G in the paper):
#     wStart = 0
#     wEnd = 6 + 0.001  # 2
#     wStep = 0.05
#     wes = np.arange(wStart + wStep,
#                     wEnd,
#                     wStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of wes depends on the conectome.
#     numW = wes.size  # length(wes);
#
#     # ==========================
#     # Some monitoring info: initialization
#     N = C.shape[0]
#     JI=np.zeros((N,numW))
#     Se_init = np.zeros((N,numW))
#     Si_init = np.zeros((N,numW))
#     for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
#         J = Balance_J9.JOptim(C, we)
#         Se_init[:, kk] = integrator.simVars[0].reshape(N)  # Store steady states S^E (after many iterations/simulations) -> sn
#         Si_init[:, kk] = integrator.simVars[1].reshape(N)   # Store steady states S^I -> sg
#         JI[:,kk]=J
#
#     sio.savemat('Data_Produced/BenjiBalancedWeights-py.mat', #{'JI': JI})
#                 {'wes': wes,
#                  'JI': JI,
#                  'Se_init': Se_init,
#                  'Si_init': Si_init})  # save Benji_Balanced_weights wes JI Se_init Si_init
#
#     return JI

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
integrator.verbose = False

# Simple verification test, to check the info from the paper...
# I_e = -0.026+DMF.be/DMF.ae
# print("phie",DMF.phie(I_e))
# # result: phie 3.06308542427

# Load connectome:
# --------------------------------
CFile = sio.loadmat('Data_Raw/Human_66.mat')  # load Human_66.mat C
C = CFile['C']
# N = 1
# C = np.zeros((N,N))

testMultipleTimes(10000, 2.1, C)

# JI = computeAllJs(C)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
