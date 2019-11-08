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
import RunJansenRitSim as runJR
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


def calibrateTau_i_Values(we):
    def computeFrec(C, tau_i):
        print("compFreq: tau_i={} => ".format(tau_i), end=' ')
        setTau_i(tau_i)
        trials = 20
        results = np.zeros(trials)
        for n in range(trials):
            print('{},'.format(n), end=' ', flush=True)
            integrator.neuronalModel.resetBookkeeping()
            f, p, raw_freqs, raw_power, raw_Data = runSim(C)
            results[n] = np.min(f)
        avg = np.average(results)
        print('-> freq =',avg)
        return avg

    print("Computing calibrateTau_i_Values")
    JR.we = we
    N = 1
    C = np.zeros((N,N))
    stepPlot = 0.1
    tau_is = np.arange(2., 70.+stepPlot, stepPlot)  # [ms]
    freqs = np.zeros(len(tau_is))
    for pos, tau_i in enumerate(tau_is):
        freqs[pos] = computeFrec(C, tau_i)
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1)
    ax.plot(tau_is, freqs, lw=4, label='average', color='red')
    fig.suptitle(r'Plot of the frequencies vs. $\tau_i$')
    ax.legend(loc='lower right')
    ax.set_xlabel(r'$\tau_i$')
    ax.set_ylabel(r'Freq')
    ax.grid()
    plt.show()
    sio.savemat('Data_Produced/JR-Frecs_vs_Tau_i.mat',
                {'Tau_i': tau_is,
                 'Freqs': freqs})



# =================================================================================================
# Functions to study/observe the behavior of the system for different coupling constants we
# =================================================================================================
def runSimAndSelectWorstNode(we):
    print("\nComputing runSimAndSelectWorstNode")
    JR.we = we
    f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
    print('finished sim: fr=', f)
    print('max freq={} at pos {}'.format(np.max(f), np.argmax(f)))
    print('min freq={} at pos {}'.format(np.min(f), np.argmin(f)))
    print('avg freq={}'.format(np.average(f)))
    print('var freq={}'.format(np.var(f)))
    print('Error={}'.format(distTo3Hz(f)))
    print()
    return np.min(f), np.argmin(f)


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


def plotSpectrumforSeveralWe():
    wes = np.arange(0, 600, 100)
    plt.rcParams.update({'font.size': 22})
    fig, axs = plt.subplots(len(wes), sharex=True)
    fig.suptitle("Pyramidal cell firing rates")
    for pos, we in enumerate(wes):
        JR.we = we
        f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
        print('finished sim: we={}'.format(we))
        print('   max freq={} at pos {}'.format(np.max(f), np.argmax(f)))
        print('   min freq={} at pos {}'.format(np.min(f), np.argmin(f)))
        print('   avg freq={}'.format(np.average(f)))
        print('   var freq={}'.format(np.var(f)))
        axs[pos].bar(np.arange(len(f)), f)
        axs[pos].set_title("we = {}".format(JR.we))
    plt.show()


def plotTimeResponseForSeveralWe(minmax):
    fixedNode = 1
    wes = np.arange(0, 600, 100)
    print("going to process:", wes)
    time = np.arange(0, Tmaxneuronal, JR.ds)
    plt.rcParams.update({'font.size': 22})
    fig, axs = plt.subplots(len(wes), sharex=True)
    if not fixedNode:
        fig.suptitle("Plot of the time response for best nodes")
    else:
        fig.suptitle("Plot of the time response for node {}".format(fixedNode))
    for pos, we in enumerate(wes):
        print('starting sim: we={}'.format(we))
        JR.we = we
        f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
        if not fixedNode:
            node = np.argmax(f) if minmax == 'max' else np.argmin(f)
        else:
            node = fixedNode
        print("   Checking node {}".format(node))
        print('   max freq={} at pos {}'.format(np.max(f), np.argmax(f)))
        print('   min freq={} at pos {}'.format(np.min(f), np.argmin(f)))
        print('   avg freq={}'.format(np.average(f)))
        print('   var freq={}'.format(np.var(f)))
        lowCut = int(.1 * len(raw_Data))  # int(1./JR.ds)  # Ignore the first steps for warm-up...
        axs[pos].plot(time[lowCut:], raw_Data[lowCut:, node])
        axs[pos].set_title("we={}, Node ={}".format(JR.we, node))
    plt.show()


def plotHist2DForAllWe():
    def simWe(we):
        JR.we = we
        f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
        return f

    print("Computing plotHist2DForAllWe")
    wes = np.arange(0, 600, 30)

    print('starting sim: we={}'.format(wes[0]))
    psp_peak_freq = simWe(wes[0])
    for pos, we in enumerate(wes[1:]):
        print('starting sim: we={}'.format(we))
        f = simWe(we)
        psp_peak_freq = np.vstack((psp_peak_freq, f))

    # define colormap
    lower = plt.cm.jet(np.linspace(0,1,200))
    colors = np.vstack(([0,0,0,0],lower))
    tmap = matplotlib.colors.LinearSegmentedColormap.from_list('test', colors)

    # plot psp frequency
    x_coord = wes.repeat(Conn.shape[0])
    plt.rcParams.update({'font.size': 22})
    plt.hist2d(x_coord, psp_peak_freq.flatten(), bins=[len(wes),40], cmap=tmap,
              range=[[np.min(wes),np.max(wes)],[-1,14]] ) #, vmax=100)
    plt.colorbar(label="Number of regions")
    plt.grid()
    plt.ylabel(' Frequency in Hz')
    plt.xlabel(' global coupling ')
    plt.show()


def tune_simRun_SimRun2_ForAllWe():
    def simWe(we):
        JR.we = we
        f, p, raw_freqs, raw_power, raw_Data = runJR.runSim(Conn)
        f2, p2, raw_freqs2, raw_power2, raw_Data2 = runJR.runSim2(Conn)
        return np.average(f), np.average(f2)

    print("Computing tune_simRun_SimRun2_ForAllWe")
    wes = np.arange(0, 600, 30)
    time = np.arange(0, Tmaxneuronal, JR.ds)
    plt.rcParams.update({'font.size': 22})
    avgf = np.zeros(len(wes))
    avgf2 = np.zeros(len(wes))
    for pos, we in enumerate(wes):
        print('starting sim: we={}'.format(we))
        avgf[pos], avgf2[pos] = simWe(we)
        print('   avg freq ={}'.format(avgf[pos]))
        print('   avg freq2={}'.format(avgf2[pos]))
    fig, ax = plt.subplots(1)
    ax.plot(wes, avgf, lw=4, label='avg', color='red')
    ax.plot(wes, avgf2, lw=2, label='avg 2', color='blue')
    ax.legend(loc='upper left') # lower/upper
    ax.set_xlabel('we')
    ax.set_ylabel('freq')
    ax.grid()
    plt.show()


def plotMaxMinFrecsForAllWe():
    def simWe(we):
        JR.we = we
        f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
        return f

    print("Computing plotMaxMinFrecsForAllWe")
    wes = np.arange(0, 600, 10)
    plt.rcParams.update({'font.size': 22})
    minf = np.zeros(len(wes))
    maxf = np.zeros(len(wes))
    avgf = np.zeros(len(wes))
    varf = np.zeros(len(wes))
    for pos, we in enumerate(wes):
        print('starting sim: we={}'.format(we))
        f = simWe(we)
        node = np.argmin(f)
        minf[pos] = np.min(f)
        maxf[pos] = np.max(f)
        avgf[pos] = np.average(f)
        varf[pos] = np.var(f)
        print("   Checking node {}".format(node))
        print('   max freq={} at pos {}'.format(maxf[pos], np.argmax(f)))
        print('   min freq={} at pos {}'.format(minf[pos], np.argmin(f)))
        print('   avg freq={}'.format(avgf[pos]))
        print('   var freq={}'.format(varf[pos]))
    fig, ax = plt.subplots(1)
    ax.plot(wes, avgf, lw=4, label='average', color='red')
    ax.plot(wes, minf, lw=2, label='min', color='blue')
    ax.plot(wes, maxf, lw=2, label='max', color='green')
    ax.fill_between(wes, minf, maxf, facecolor='yellow', alpha=0.5)  #,label='freq range'
    ax.legend(loc='lower right')  # lower/upper
    ax.set_xlabel('we')
    ax.set_ylabel('freq')
    ax.grid()
    plt.show()


def plotSensitivityForAllWe():
    def simWe(we):
        JR.we = we
        integrator.neuronalModel.resetBookkeeping()
        f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
        return f

    print("Computing plotSensitivityForAllWe")
    wes = np.arange(0, 600, 10)
    numTrials = 20
    plt.rcParams.update({'font.size': 22})
    minAvg = np.zeros(len(wes))
    maxAvg = np.zeros(len(wes))
    avgAvg = np.zeros(len(wes))
    for pos, we in enumerate(wes):
        print('starting sim: we = {}'.format(we))
        localAvg = np.zeros(numTrials)
        localMin = np.zeros(numTrials)
        localMax = np.zeros(numTrials)
        localRes = np.zeros(numTrials)
        for n in range(numTrials):
            print('{},'.format(n), end=' ', flush=True)
            f = simWe(we)
            localAvg[n] = np.average(f)
            localMin[n] = np.min(f)
            localMax[n] = np.max(f)
            localRes[n] = distTo3Hz(f)
        minAvg[pos] = np.min(localAvg)
        maxAvg[pos] = np.max(localAvg)
        avgAvg[pos] = np.average(localAvg)
        print()
        print('   max ={}'.format(maxAvg[pos]))
        print('   min ={}'.format(minAvg[pos]))
        print('   avg ={}'.format(avgAvg[pos]))
    fig, ax = plt.subplots(1)
    ax.plot(wes, avgAvg, lw=4, label='average', color='red')
    ax.plot(wes, minAvg, lw=2, label='min', color='blue')
    ax.plot(wes, maxAvg, lw=2, label='max', color='green')
    ax.legend(loc='lower left')  # lower/upper
    ax.set_xlabel('we')
    ax.set_ylabel('freq')
    ax.grid()
    plt.show()


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


# =================================================================================
# Sensibility calibration test... repeat the SAME experiment over and over again,
# and make a histogram out of the results. It should look like a gaussian...
# =================================================================================
# Results with 1000 samples:
# Average= 4.472627442078614 std= 0.09824919709702232
# Min= 4.159013873023076 Max= 4.754641103347178
def testMultipleTimes(trials, tau_i, we):
    print("Testing, multiple {} times...".format(trials))
    JR.we = we
    setTau_i(tau_i)
    results = np.zeros(trials)
    for n in range(trials):
        print('starting sim ({}): tau_i={}'.format(n,tau_i))
        error = errorFunc(tau_i)
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


# =================================================================================================
# Functions for FIC control
# =================================================================================================
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
def brutefoce1DOptim(we):
    print("Computing brutefoce1DOptim...")
    JR.we = we
    stepOptim = 1.
    tau_is = np.arange(2., 60., stepOptim)  # [ms]
    minValue = np.inf
    minPos = 0.
    for tau_i in tau_is:
        print('starting sim: tau_i={}'.format(tau_i))
        res = errorFunc(tau_i)
        if res < minValue:
            minValue = res
            minPos = tau_i
    print("\n\nFinished: {} at {}".format(minValue, minPos))
    return minPos


def brutefoce1VariableOptim(we, tau_i, targetArea):
    print("Computing brutefoce1VariableOptim...")
    JR.we = we
    startingValue = tau_i[targetArea]
    amp = 10.
    steps = 40.
    tau_is = np.arange(startingValue-amp, startingValue+amp+2.*amp/steps, 2.*amp/steps)  # [ms]
    minValue = np.inf
    minPos = 0.
    for tau_i_value in tau_is:
        print('starting sim: tau_i[{}]={}'.format(targetArea, tau_i_value))
        tau_i[targetArea] = tau_i_value
        res = errorFunc(tau_i)
        if res < minValue:
            minValue = res
            minPos = tau_i_value
    print("\n\nFinished: {} at {}".format(minValue, minPos))
    return minPos


def Optim2(we, startingValue):
    print("Computing Optim2...")
    JR.we = we
    # init...
    N = Conn.shape[0]
    integrator.neuronalModel.initBookkeeping(N, tmax)

    initialValues = startingValue * np.ones(N)  # [b_defaultValue] * N
    stepOptim = 1.
    lowB = startingValue - stepOptim * 3.
    upperB = startingValue + stepOptim * 3.
    bounds = [(lowB, upperB) for _ in initialValues]

    # ---------------------------------------------------------------------------
    # Now, fit it !!!
    # ---------------------------------------------------------------------------
    # # Using optim.minimize > CG:
    # import scipy.optimize as optim
    # print("Optim with optim.minimize > CG")
    # res = optim.minimize(errorFunc, bounds=bounds, x0=initialValues, method='CG')
    # ---------------------------------------------------------------------------
    # # Using optim.basinhopping
    # import scipy.optimize as optim
    # print("Optim with optim.basinhopping")
    # res = optim.basinhopping(errorFunc, x0=initialValues)  # basinhopping does not support bounds...
    # ---------------------------------------------------------------------------
    # # Using Noisyopt: A python library for optimizing noisy functions (https://github.com/andim/noisyopt)
    import noisyopt
    print("Optim with noisyopt.minimizeCompass")
    res = noisyopt.minimizeCompass(errorFunc, bounds=bounds, x0=initialValues, deltatol=0.1, paired=False)
    # ---------------------------------------------------------------------------
    # # Using Scikit-Optimize (https://github.com/scikit-optimize/scikit-optimize)
    # import skopt
    # print("Optim with skopt.gp_minimize")
    # res = skopt.gp_minimize(errorFunc, bounds, n_calls=N*1000, x0=initialValues)
    # print("Optim with skopt.forest_minimize")
    # res = skopt.forest_minimize(errorFunc, bounds, n_calls=N*100, x0=initialValues)
    # print("Optim with skopt.gbrt_minimize")
    # res = skopt.gbrt_minimize(errorFunc, bounds, n_calls=N*100, x0=initialValues)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    final_values = errorFunc(res.x)
    print("Result:", res)
    print("Final Value:", final_values+targetFreq)
    return final_values


# def updateJ(N, tmax, delta, f, p, J):
#     # tmin = 1000 if (tmax>1000) else int(tmax/10)
#     # currm = np.mean(curr[tmin:tmax, :], 0)  # takes the mean of all xn values along dimension 1...
#     # This is the "averaged level of the input of the local excitatory pool of each brain area,
#     # i.e., I_i^{(E)}" in the text (pp 7889, right column, subsection "FIC").
#     flag = 0
#     for n in range(N):
#         print(" {}:".format(n), end='')
#         if np.abs(f[n] - targetFreq) > 0.05:  # if currm_i < -0.026 - 0.005 or currm_i > -0.026 + 0.005 (a tolerance)
#             if f[n] < targetFreq:  # if currm_i < -0.026
#                 print("v", end='')
#                 J[n] = J[n] - delta[n]  # down-regulate
#                 delta[n] = delta[n] - 0.001
#                 if delta[n] < 0.001:
#                     print("|", end='')
#                     delta[n] = 0.001
#             else:  # if currm_i >= -0.026 (in the paper, it reads =)
#                 print("^", end='')
#                 J[n] = J[n] + delta[n]  # up-regulate
#         else:
#             print("!", end='')
#             flag = flag + 1
#     print(" flag:{}".format(flag))
#     return flag == N


# def JOptim(we):
#     N = Conn.shape[0]
#
#     integrator.neuronalModel.we = we
#     integrator.neuronalModel.initJ(N)
#
#     # initialization:
#     # -------------------------
#     integrator.neuronalModel.initBookkeeping(N, tmax)
#     delta = 0.02 * np.ones(N)
#
#     print()
#     print("we=", integrator.neuronalModel.we)  # display(we)
#     # print("  Trials:", end=" ", flush=True)
#
#     ### Balance (greedy algorithm)
#     # note that we used stochastic equations to estimate the JIs
#     # Doing that gives more stable solutions as the JIs for each node will be
#     # a function of the variance.
#     for k in range(100):  # 5000 trials
#         integrator.neuronalModel.resetBookkeeping()
#         f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
#         print("trial {}=>".format(k), end=" ", flush=True)
#
#         # currm = integrator.neuronalModel.curr_xn - integrator.neuronalModel.be/integrator.neuronalModel.ae  # be/ae==125./310. Records currm_i = xn-be/ae (i.e., I_i^E-b_E/a_E in the paper) for each i (1 to N)
#         flagJ = updateJ(N, tmax, delta, f, p, integrator.neuronalModel.J)
#         if flagJ:
#             print('Out !!!', flush=True)
#             break
#
#     return integrator.neuronalModel.J

# def computeJs():
#     # all tested global couplings (G in the paper):
#     WStart = 0
#     WEnd = 2 + 0.001  # 2
#     WStep = 0.05
#     wes = np.arange(WStart + WStep,
#                     WEnd,
#                     WStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of Gs depends on the conectome.
#     numW = wes.size  # length(wes);
#
#     # np.random.seed(42)  # Fix the seed for debug purposes...
#
#     # ==========================
#     # Some monitoring info: initialization
#     N = Conn.shape[0]
#     JI=np.zeros((N,numW))
#     y0_init = np.zeros((N, numW))
#     y1_init = np.zeros((N, numW))
#     y2_init = np.zeros((N, numW))
#     y3_init = np.zeros((N, numW))
#     y4_init = np.zeros((N, numW))
#     y5_init = np.zeros((N, numW))
#     for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
#         J = Balance_J9.JOptim(we)
#         y0_init[:, kk] = integrator.simVars[0].reshape(N)
#         y1_init[:, kk] = integrator.simVars[1].reshape(N)
#         y2_init[:, kk] = integrator.simVars[2].reshape(N)
#         y3_init[:, kk] = integrator.simVars[3].reshape(N)
#         y4_init[:, kk] = integrator.simVars[4].reshape(N)
#         y5_init[:, kk] = integrator.simVars[5].reshape(N)
#         JI[:,kk]=J[:,0]
#
#     sio.savemat('Data_Produced/JansenRitBalancedWeights.mat',
#                 {'wes': wes,
#                  'JI': JI,
#                  'y0_init': y0_init,
#                  'y1_init': y1_init,
#                  'y2_init': y2_init,
#                  'y3_init': y3_init,
#                  'y4_init': y4_init,
#                  'y5_init': y5_init
#                  })  # save Benji_Balanced_weights wes JI Se_init Si_init


# ======================================================================
# ======================================================================
# ======================================================================


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
    tau_i_Start = 49.  # [ms]
    tau_i = tau_i_Start * np.ones(N)
    setTau_i(tau_i)
    # runAndPlotSim(we)
    worstf, worstNode = runSimAndSelectWorstNode(we)
    print("Worst node is {} with f={}".format(worstNode, worstf))
    print("Full tau_i:", tau_i)

    setTau_i(tau_i)
    plotErrorForTargetAreaForAllTau_i(we, tau_i, worstNode)
    # tau_i_Node = brutefoce1VariableOptim(we, tau_i, worstNode)
    # print("New tau_i found:", tau_i_Node)
    # tau_i[worstNode] = tau_i_Node
    # print("Full tau_i:", tau_i)
    # setTau_i(tau_i)
    # runAndPlotSim(we)

    # tau_i_End2 = Optim2(we, tau_i_Start)
    # setTau_i(tau_i_End2)
    # runAndPlotSim(we)
# ======================================================================
# ======================================================================
# ======================================================================
# brutefoce1DOptim:
# ------------------
# Finished: 4.193943525194505 at 49.0
# finished sim: fr= [2.00400802 2.90581162 2.90581162 1.70340681 3.15631263 2.90581162
#  3.10621242 2.65531062 3.10621242 1.75350701 3.10621242 3.15631263
#  3.10621242 2.90581162 3.10621242 1.75350701 3.15631263 2.90581162
#  2.90581162 2.90581162 1.75350701 2.70541082 1.70340681 2.90581162
#  2.00400802 2.90581162 2.90581162 3.10621242 2.65531062 3.15631263
#  2.65531062 3.15631263 2.00400802 3.05611222 2.90581162 2.90581162
#  1.75350701 3.15631263 2.90581162 3.10621242 2.75551102 3.05611222
#  1.70340681 3.05611222 2.90581162 3.05611222 2.90581162 3.05611222
#  2.40480962 3.15631263 2.90581162 2.90581162 2.90581162 3.10621242
#  2.75551102 1.70340681 2.90581162 2.10420842 2.90581162 2.90581162
#  2.90581162 2.75551102 3.10621242 2.75551102 3.15631263 2.75551102]
# max freq=3.1563126252505005 at pos 4
# min freq=1.7034068136272544 at pos 3
# avg freq=2.745642800753021
# var freq=0.21291289819647308
# Checking node 3
