# ================================================================================================================
#
# This prog. optimizes the strengh of the feedback inhibition of the FIC model
# for varying global couplings (G)
# Saves the steady states and the feedback inhibition (J).
#
# see:
# Deco et al. (2014) J Neurosci.
# http://www.jneurosci.org/content/34/23/7886.long
#
# Adapted by Gustavo Patow to the JR model
#
# Bibliography:
# [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and visual evoked potential generation in a
#           mathematical model of coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.
# [DecoEtAl2014] Gustavo Deco, Adrián Ponce-Alvarez, Patric Hagmann, Gian Luca Romani, Dante Mantini and Maurizio
#           Corbetta, "How Local Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics" (2014), Journal of
#           Neuroscience 4 June 2014, 34 (23) 7886-7898; DOI: https://doi.org/10.1523/JNEUROSCI.5068-13.2014
# [PyRates] Gast, R., Daniel, R., Moeller, H. E., Weiskopf, N. and Knoesche, T. R. (2019). “PyRates – A Python Framework
#           for rate-based neural Simulations.” bioRxiv (https://www.biorxiv.org/content/10.1101/608067v2).
# [SpieglerEtAl2013] Spiegler A1, Kiebel SJ, Atay FM, Knösche TR. (2010). "Bifurcation analysis of neural mass models:
#           Impact of extrinsic inputs and dendritic time constants."
#           Neuroimage. Sep;52(3):1041-58. doi: 10.1016/j.neuroimage.2009.12.081. Epub 2010 Jan 4.
# [DF_2003] Olivier David, Karl J. Friston, “A neural mass model for MEG/EEG:: coupling and neuronal dynamics”, NeuroImage,
#           Volume 20, Issue 3, 2003, Pages 1743-1755, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2003.07.015.
# [StefanovskiEtAl2019] Stefanovski, L., P. Triebkorn, A. Spiegler, M.-A. Diaz-Cortes, A. Solodkin, V. Jirsa,
#           R. McIntosh and P. Ritter; for the Alzheimer's disease Neuromigang Initiative (2019).
#           "Linking molecular pathways and large-scale computational modeling to assess candidate
#           disease mechanisms and pharmacodynamics in Alzheimer's disease." bioRxiv: 600205.
#           https://github.com/BrainModes/TVB_EducaseAD_molecular_pathways_TVB/blob/master/Educase_AD_study-LS-Surrogate.ipynb
#
# ================================================================================================================
import numpy as np
import bisect
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
# import importlib
# JR = importlib.import_module("functions.Models.JansenRit+FIC")
import functions.Models.JansenRit as JR
import functions.Integrator_Euler as integrator
integrator.neuronalModel = JR
integrator.clamping = False
# import functions.Balance_J9 as Balance_J9
# Balance_J9.integrator = integrator
import JR_RunSim as runJR
runJR.JR = JR

# A simple var for simplifying postprocessing selection...
runSim = runJR.runSim2

# The connectivity
Conn = None

# # [DF_2003] used a standard normal distribution...
# import functions.Stimuli.randomStdNormal as stimuli
# stimuli.N = None
# stimuli.onset = 0.
# stimuli.mu = 220.
# stimuli.sigma = 22.
# integrator.stimuli = stimuli

# In the original [JR_1995] paper, the random white noise input p(t) had an amplitude
# varying between 120 and 320 pulses per second.
import functions.Stimuli.randomUniform as stimuli
stimuli.N = None
stimuli.onset = 0.
stimuli.ampLo = 120.
stimuli.ampHi = 320.
integrator.stimuli = stimuli

# # Use a constant stimuli of 108.5/s., as in [StefanovskiEtAl2019]
# import functions.Stimuli.constant as stimuli
# # Do not set N, as it is constant...
# stimuli.onset = 0.
# stimuli.amp = 108.5  # [s^-1]
# integrator.stimuli = stimuli


# Integration parms...
dt = 5e-5
runJR.dt = dt
tmax = 20.
runJR.tmax = tmax
JR.ds = 1e-4
Tmaxneuronal = int((tmax+dt))
runJR.Tmaxneuronal = Tmaxneuronal

# =================================================================================================
# Tau_i manipulations
# =================================================================================================
# # Take the original values, so we can keep the ratio invariant...
# # With these values, the node will spike at 3.052Hz
H_e_orig = 3.25         # JR.A [mV]
tau_e_orig = 1./JR.a    # [s]
tau_e = 51e-3  # 10e-3  # 0.051 # [s]
# tau_i = 54e-3  # 14e-3  # 0.0540513
JR.A = H_e_orig*tau_e_orig/tau_e  # This uses the definition by [SpieglerEtAl2013] & [PyRates]
JR.a = 1./tau_e
# Gather information for the inhibitory computations...
H_i_orig = JR.B         # 22. [mV]
tau_i_orig = 1./JR.b    # 20e-3 [s]
def setTau_i(tau_i_ms):  # receives the value of tau_i in [ms]
    tau_i = tau_i_ms * 1e-3  # Transform from [ms] to [s]
    JR.B = H_i_orig*tau_i_orig/tau_i  # This uses the definition by [SpieglerEtAl2013] & [PyRates]
    JR.b = 1./tau_i


# =================================================================================================
# Functions to execute a JR simulation and compute its error...
# =================================================================================================
targetFreq = 3.  # We want the firing rate to be at 3Hz
def distTo3Hz(f):
    import functions.Utils.errorMetrics as error
    # return np.abs(np.average(f)-targetFreq)
    return error.l2(f, targetFreq)


def errorFunc(tau_i):
    print("errorFunc: tau_i=",tau_i, end=' ')
    setTau_i(tau_i)
    integrator.neuronalModel.resetBookkeeping()
    f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
    res = distTo3Hz(f)
    print('-> error=',res)
    return res


# =================================================================================================
# Function to measure the relation between Tau_i and the spiking freq for the JR model
# =================================================================================================
def calibrateTau_i_Values(we):
    def computeFrec(C, tau_i):
        print("compFreq: tau_i={} => ".format(tau_i), end=' ')
        setTau_i(tau_i)
        trials = 50
        resultsMin = np.zeros(trials)
        resultsMax = np.zeros(trials)
        for n in range(trials):
            print('{},'.format(n), end=' ', flush=True)
            integrator.neuronalModel.resetBookkeeping()
            f, p, raw_freqs, raw_power, raw_Data = runSim(C)
            resultsMin[n] = np.min(f)
            resultsMax[n] = np.max(f)
        avgMin = np.average(resultsMin)
        avgMax = np.average(resultsMax)
        print('-> freq =',avgMin)
        return avgMin, avgMax

    print("Computing calibrateTau_i_Values (single node, no plot)")
    JR.we = we
    N = 1
    C = np.zeros((N,N))
    stepPlot = 0.1
    tau_is = np.arange(2., 70.+stepPlot, stepPlot)  # [ms]
    freqsMin = np.zeros(len(tau_is))
    freqsMax = np.zeros(len(tau_is))
    for pos, tau_i in enumerate(tau_is):
        freqsMin[pos], freqsMax[pos] = computeFrec(C, tau_i)
    # plt.rcParams.update({'font.size': 22})
    # fig, ax = plt.subplots(1)
    # ax.plot(tau_is, freqsMin, lw=4, label='average min', color='red')
    # ax.plot(tau_is, freqsMax, lw=4, label='average max', color='red')
    # fig.suptitle(r'Plot of the frequencies vs. $\tau_i$')
    # ax.legend(loc='lower right')
    # ax.set_xlabel(r'$\tau_i$')
    # ax.set_ylabel(r'Freq')
    # ax.grid()
    # plt.show()
    sio.savemat('Data_Produced/JR-Frecs_vs_Tau_i.mat',
                {'Tau_i': tau_is,
                 'FreqsMin': freqsMin,
                 'FreqsMax': freqsMax})

# =================================================================================================
# Plotting functions for FIC control
# =================================================================================================
def runAndPlotSim(we):
    JR.we = we
    f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)

    print('finished sim: fr=', f)
    print('max freq={} at pos {}'.format(np.max(f), np.argmax(f)))
    print('min freq={} at pos {}'.format(np.min(f), np.argmin(f)))
    print('avg freq={}'.format(np.average(f)))
    print('var freq={}'.format(np.var(f)))
    print('Error={}'.format(distTo3Hz(f)))
    plt.rcParams.update({'font.size': 22})
    plt.bar(np.arange(len(f)), f)
    plt.show()

    node = np.argmin(f)
    print("Checking node {}".format(node))
    time = np.arange(0, Tmaxneuronal, JR.ds)
    lowCut = int(.1 * len(raw_Data))  # int(1./JR.ds)  # Ignore the first steps for warm-up...
    plt.plot(time[lowCut:], raw_Data[lowCut:,node], 'k', alpha=1.0)
    plt.title("Plot of the time response for node {}".format(node))
    plt.show()

    highCut = 400
    plt.bar(raw_freqs[0:highCut], raw_power[0:highCut, node])
    plt.title("First {} freqs for node {}".format(highCut, node))
    plt.show()

    print("done!")


def plotMaxMinFrecsForAllTau_i(we):
    JR.we = we
    print("Computing plotMaxMinFrecsForAllTau_i")
    stepPlot = 0.5
    tau_is = np.arange(2., 60.+stepPlot, stepPlot)  # [ms]
    minf = np.zeros(len(tau_is))
    maxf = np.zeros(len(tau_is))
    avgf = np.zeros(len(tau_is))
    varf = np.zeros(len(tau_is))
    # node = 3
    for pos, tau_i in enumerate(tau_is):
        print('starting sim: tau_i={}'.format(tau_i))
        setTau_i(tau_i)  # tau_i * np.ones(C.shape[0])
        f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
        minf[pos] = np.min(f)
        maxf[pos] = np.max(f)
        avgf[pos] = np.average(f)
        varf[pos] = np.var(f)
        node = np.argmin(f)
        print('   max freq={} at pos {}'.format(maxf[pos], np.argmax(f)))
        print('   min freq={} at pos {}'.format(minf[pos], np.argmin(f)))
        print('   min freq={} at NODE {}'.format(f[node], node))
        print('   avg freq={}'.format(avgf[pos]))
        print('   var freq={}'.format(varf[pos]))
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1)
    ax.plot(tau_is, avgf, lw=4, label='average', color='red')
    ax.plot(tau_is, minf, lw=2, label='min', color='blue')
    ax.plot(tau_is, maxf, lw=2, label='max', color='green')
    ax.fill_between(tau_is, minf, maxf, facecolor='yellow', alpha=0.5)  #,label='freq range'
    ax.axhline(targetFreq, color='magenta', linewidth=2, linestyle='--')
    ax.legend(loc='lower right')
    ax.set_xlabel(r'$\tau_i$')
    ax.set_ylabel('freq')
    ax.grid()
    plt.show()


def plotErrorForAllTau_i(we):
    JR.we = we
    print("Computing plotErrorForAllTau_i")
    stepPlot = 0.5
    tau_is = np.arange(2., 60.+stepPlot, stepPlot)  # [ms]
    errs = np.zeros(len(tau_is))
    for pos, tau_i in enumerate(tau_is):
        print('starting sim: tau_i={}'.format(tau_i))
        errs[pos] = errorFunc(tau_i)
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1)
    ax.plot(tau_is, errs, lw=4, label='average', color='red')
    fig.suptitle(r'Plot of the $l^2$ Error for all $\tau_i$')
    ax.legend(loc='lower right')
    ax.set_xlabel(r'$\tau_i$')
    ax.set_ylabel(r'$l^2$ Error')
    ax.grid()
    plt.show()


def plotErrorForTargetAreaForAllTau_i(we, tau_i, targetArea):
    def averageTrials(trials):
        results = np.zeros(trials)
        for n in range(trials):
            print('starting sim ({})'.format(n), end=' ')
            error = errorFunc(tau_i)
            results[n] = error
        avg = np.average(results)
        std = np.std(results)
        return avg

    print("Computing plotErrorForTargetAreaForAllTau_i...")
    JR.we = we
    stepOptim = 1.
    tau_is = np.arange(2., 60.+2., stepOptim)  # [ms]
    minValue = np.inf
    minPos = 0.
    errs = np.zeros(len(tau_is))
    trials = 100
    for pos, tau_i_value in enumerate(tau_is):
        print('starting sim: tau_i[{}]={}'.format(targetArea, tau_i_value))
        tau_i[targetArea] = tau_i_value
        errs[pos] = averageTrials(trials)
        if errs[pos] < minValue:
            minValue = errs[pos]
            minPos = tau_i_value
    print("\n\nFinished: minimum of {} at {}".format(minValue, minPos))
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1)
    ax.plot(tau_is, errs, lw=4, label='average', color='red')
    fig.suptitle(r'Plot of the averaged $l^2$ Error for all $\tau_i$ for area {} ({} trials)'.format(targetArea, trials))
    ax.legend(loc='lower right')
    ax.set_xlabel(r'$\tau_i$')
    ax.set_ylabel(r'$l^2$ Error')
    ax.grid()
    plt.show()


# =================================================================================================
# Optimization methods!
# =================================================================================================
# def brutefoce1DOptim(we):
#     print("Computing brutefoce1DOptim...")
#     JR.we = we
#     stepOptim = 1.
#     tau_is = np.arange(2., 60., stepOptim)  # [ms]
#     minValue = np.inf
#     minPos = 0.
#     for tau_i in tau_is:
#         print('starting sim: tau_i={}'.format(tau_i))
#         res = errorFunc(tau_i)
#         if res < minValue:
#             minValue = res
#             minPos = tau_i
#     print("\n\nFinished: {} at {}".format(minValue, minPos))
#     return minPos


# def Optim2(we, startingValue):
#     print("Computing Optim2...")
#     JR.we = we
#     # init...
#     N = Conn.shape[0]
#     integrator.neuronalModel.initBookkeeping(N, tmax)
#
#     initialValues = startingValue * np.ones(N)  # [b_defaultValue] * N
#     stepOptim = 1.
#     lowB = startingValue - stepOptim * 3.
#     upperB = startingValue + stepOptim * 3.
#     bounds = [(lowB, upperB) for _ in initialValues]
#
#     # ---------------------------------------------------------------------------
#     # Now, fit it !!!
#     # ---------------------------------------------------------------------------
#     # # Using optim.minimize > CG:
#     # import scipy.optimize as optim
#     # print("Optim with optim.minimize > CG")
#     # res = optim.minimize(errorFunc, bounds=bounds, x0=initialValues, method='CG')
#     # ---------------------------------------------------------------------------
#     # # Using optim.basinhopping
#     # import scipy.optimize as optim
#     # print("Optim with optim.basinhopping")
#     # res = optim.basinhopping(errorFunc, x0=initialValues)  # basinhopping does not support bounds...
#     # ---------------------------------------------------------------------------
#     # # Using Noisyopt: A python library for optimizing noisy functions (https://github.com/andim/noisyopt)
#     import noisyopt
#     print("Optim with noisyopt.minimizeCompass")
#     res = noisyopt.minimizeCompass(errorFunc, bounds=bounds, x0=initialValues, deltatol=0.1, paired=False)
#     # ---------------------------------------------------------------------------
#     # # Using Scikit-Optimize (https://github.com/scikit-optimize/scikit-optimize)
#     # import skopt
#     # print("Optim with skopt.gp_minimize")
#     # res = skopt.gp_minimize(errorFunc, bounds, n_calls=N*1000, x0=initialValues)
#     # print("Optim with skopt.forest_minimize")
#     # res = skopt.forest_minimize(errorFunc, bounds, n_calls=N*100, x0=initialValues)
#     # print("Optim with skopt.gbrt_minimize")
#     # res = skopt.gbrt_minimize(errorFunc, bounds, n_calls=N*100, x0=initialValues)
#
#     # ---------------------------------------------------------------------------
#     # ---------------------------------------------------------------------------
#     final_values = errorFunc(res.x)
#     print("Result:", res)
#     print("Final Value:", final_values+targetFreq)
#     return final_values


l2threshold = 3.
localFreqThreshold = 0.3  # the variance * 3
def Optim3(we, tau_i):
    def getIrredeemables(nodeFreqs, nodeTauis, minTaui):
        candidates = nodeFreqs[np.where(nodeTauis == minTaui)]
        irredeemables = targetFreq - candidates > localFreqThreshold
        return irredeemables

    def evaluate(tau_i):
        print("errorFunc: tau_i=",tau_i, end=' ')
        setTau_i(tau_i)
        integrator.neuronalModel.resetBookkeeping()
        f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
        res = distTo3Hz(f)
        print('-> error=',res)
        return f, res

    def update_Tau_i(nodeFreqs, nodeTauis, freqResponse, allTaui):
        for nodePos, nodeF in enumerate(nodeFreqs):
            delta = targetFreq-nodeF  # In general, it will always be nodeF < targetfreq...
            if np.abs(delta) > localFreqThreshold:
                if delta < 0:
                    print("delta < 0!!!")
                currentTaui = nodeTauis[nodePos]
                currentTauiPosAtSet = np.abs(allTaui - currentTaui).argmin()
                currenttargetFreq = freqResponse[currentTauiPosAtSet]
                newf = currenttargetFreq + delta
                newPos = np.abs(freqResponse - newf).argmin()
                nodeTauis[nodePos] = allTaui[newPos]

    print("Computing Optim3, customized Optim!!!")
    JR.we = we
    # init...
    N = Conn.shape[0]
    integrator.neuronalModel.initBookkeeping(N, tmax)

    # ---------------------------------------------------------------------------
    # Load 1D isolated node behaviour
    # ---------------------------------------------------------------------------
    tauAndFreqs = sio.loadmat('Data_Produced/JR-Frecs_vs_Tau_i.mat')
    allTau_i = tauAndFreqs['Tau_i'].flatten()
    allFreqResponses = tauAndFreqs['Freqs'].flatten()

    minTaui = allTau_i[np.argmax(allFreqResponses)]
    # ---------------------------------------------------------------------------
    # Now, fit it !!!
    # ---------------------------------------------------------------------------
    convergence = np.inf
    nodeFreqs, convergence = evaluate(tau_i)
    reallyPendingNodes = np.count_nonzero(nodeFreqs)
    convergenceValues = np.array([convergence])
    while not convergence < l2threshold and reallyPendingNodes > 0:
        update_Tau_i(nodeFreqs, tau_i, allFreqResponses, allTau_i)
        nodeFreqs, convergence = evaluate(tau_i)
        convergenceValues = np.append(convergenceValues, [convergence])
        irredeemables = getIrredeemables(nodeFreqs, tau_i, minTaui)
        pendingNodes = np.where(np.abs(nodeFreqs-targetFreq) > 0.2)
        pendingFreqs = nodeFreqs[pendingNodes]
        reallyPendingNodes = np.count_nonzero(pendingNodes) - np.count_nonzero(irredeemables)
        print('convergence factor: {}'.format(convergence))
        print('reallyPendingNodes:', reallyPendingNodes)
        print('pending Nodes, Freqs and Tau_i\'s:\n', np.dstack((pendingNodes, pendingFreqs, tau_i[pendingNodes])))
        print('irredeemables:', irredeemables)


    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    print("Result:", tau_i)
    print("Final Value:", convergence)
    # print("Resulting freqs:", nodeFreqs)

    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1)
    numValues = np.count_nonzero(convergenceValues)
    ax.plot(np.arange(0,numValues), convergenceValues, lw=4, color='red')
    fig.suptitle(r'Convergence of the algorithm')
    # ax.legend(loc='lower right')
    ax.set_xlabel(r'iteration')
    ax.set_ylabel(r'$l^2$ Error')
    ax.grid()
    plt.show()

    return tau_i


if __name__ == '__main__':
    integrator.verbose = False
    # -------------------------- calibrateTau_iValues -> do this once and save the file...
    # we = 300.
    # calibrateTau_i_Values(we)
    # -------------------------- Load connectome:
    print('Loading Data_Raw/Human_66.mat connectome')
    CFile = sio.loadmat('Data_Raw/Human_66.mat')  # load Human_66.mat C
    Conn = CFile['C']
    N = Conn.shape[0]
    stimuli.N = N

    # ------------------------ Check behaviour with different we
    # we = 300.
    # setTau_i(1./26.)
    # runAndPlotSim(we)
    # plotSpectrumforSeveralWe()
    # plotTimeResponseForSeveralWe('max')
    # plotMaxMinFrecsForAllWe()
    # plotSensitivityForAllWe()
    # tune_simRun_SimRun2_ForAllWe()
    # plotHist2DForAllWe()

    # ------------------------ Do a sensitivity test
    # we = 300.
    # tau_i = 49.  # [ms]
    # testMultipleTimes(1000, tau_i, we)

    # ------------------------ Check behaviour with FIC
    we = 300.
    # plotMaxMinFrecsForAllTau_i(we)
    # plotErrorForAllTau_i(we)

    # ------------------------- Optimize !!!
    we = 300.
    print("Running connectivity matrix with FIC control...")
    # tau_i_Start = brutefoce1DOptim(we)
    tau_i_Start = 54.  # 49.  # [ms]
    tau_i = tau_i_Start * np.ones(N)

    # ------ do some verifications...
    setTau_i(tau_i)
    runAndPlotSim(we)
    # worstf, worstNode = runSimAndSelectWorstNode(we)
    # print("Worst node is {} with f={}".format(worstNode, worstf))
    # print("Full tau_i:", tau_i)

    # setTau_i(tau_i)
    # plotErrorForTargetAreaForAllTau_i(we, tau_i, worstNode)
    # tau_i_Node = brutefoce1VariableOptim(we, tau_i, worstNode)
    # print("New tau_i found:", tau_i_Node)
    # tau_i[worstNode] = tau_i_Node
    # print("Full tau_i:", tau_i)
    # setTau_i(tau_i)
    # runAndPlotSim(we)

    # ------ and optimize!!!
    setTau_i(tau_i)
    tau_i_End2 = Optim3(we, tau_i)
    setTau_i(tau_i_End2)
    runAndPlotSim(we)
# ======================================================================
# ======================================================================
# ======================================================================
