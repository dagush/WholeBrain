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
import scipy.signal as signal
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

def setH_i(H_i):  # receives the value of H_i in [mV]
    JR.B = H_i

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
# Set of functions to test the JR model...
# =================================================================================================
def calibrateAndPlot_Tau_i_Values(we):
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

    print("Computing calibrateAndPlotTau_i_Values (single node)")
    JR.we = we
    N = 1
    C = np.zeros((N,N))
    stepPlot = 0.1
    tau_is = np.arange(2., 70.+stepPlot, stepPlot)  # [ms]
    freqsMin = np.zeros(len(tau_is))
    freqsMax = np.zeros(len(tau_is))
    for pos, tau_i in enumerate(tau_is):
        freqsMin[pos], freqsMax[pos] = computeFrec(C, tau_i)
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1)
    ax.plot(tau_is, freqsMin, lw=4, label='average min', color='red')
    ax.plot(tau_is, freqsMax, lw=4, label='average max', color='red')
    fig.suptitle(r'Plot of the frequencies vs. $\tau_i$')
    ax.legend(loc='lower right')
    ax.set_xlabel(r'$\tau_i$')
    ax.set_ylabel(r'Freq')
    ax.grid()
    plt.show()
    sio.savemat('Data_Produced/JR-Frecs_vs_Tau_i.mat',
                {'Tau_i': tau_is,
                 'FreqsMin': freqsMin,
                 'FreqsMax': freqsMax})


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
    print("Computing plotTimeResponseForSeveralWe")
    fixedNode = None
    wes = np.arange(0, 600, 100)
    print("going to process:", wes)
    time = np.arange(0, Tmaxneuronal, JR.ds)
    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots(len(wes), sharex=True)
    if not fixedNode:
        if minmax == 'max':
            fig.suptitle("Plot of the time response for best nodes")
        else:
            fig.suptitle("Plot of the time response for worst nodes")
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
# Functions for FIC control: Tau_i
# =================================================================================================
def plotTimeResponseForSeveralTau_i(we, minmax, tau_is):
    print("Computing plotTimeResponseForSeveralTau_i")
    JR.we = we
    fixedNode = None
    print("going to process:", tau_is)
    time = np.arange(0, Tmaxneuronal, JR.ds)
    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots(len(tau_is), sharex=True)
    if not fixedNode:
        if minmax == 'max':
            fig.suptitle("Plot of the time response for best nodes")
        else:
            fig.suptitle("Plot of the time response for worst nodes")
    else:
        fig.suptitle("Plot of the time response for node {}".format(fixedNode))
    for pos, taui in enumerate(tau_is):
        print('starting sim: Tau_i={}'.format(taui))
        setTau_i(taui)
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
        axs[pos].set_title(r'$\tau_i$={}, Node={}'.format(taui, node))
    plt.show()


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
# Functions for FIC control: H_i
# =================================================================================================
def plotTimeResponseForSeveral_H_i(minmax, we):
    print('Computing plotTimeResponseForSeveral_H_i')
    fixedNode = not (minmax == 'max' or minmax == 'min')
    JR.we = we
    startHi = 5.
    finishHi = 17.
    stepHi = 1. # (finishHi-startHi)/stepsHi
    H_is = np.arange(startHi, finishHi, stepHi)  # [mV]
    print("going to process:", H_is)
    print("      with:", minmax)
    time = np.arange(0, Tmaxneuronal, JR.ds)
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(len(H_is), sharex=True)
    if not fixedNode:
        if minmax == 'max':
            fig.suptitle("Plot of the time response for best nodes")
        else:
            fig.suptitle("Plot of the time response for worst nodes")
    else:
        fig.suptitle("Plot of the time response for node {}".format(minmax))
    for pos, H_i in enumerate(H_is):
        print('starting sim: H_i={}'.format(H_i))
        setH_i(H_i)
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
        axs[pos].set_title(r'$H_i$={}, Node ={}'.format(H_i, node))
    plt.show()


def plotAmplitudeForAll_H_i(we):
    print('Computing plotAmplitudeForAll_H_i...')
    JR.we = we
    startHi = -10.
    finishHi = 17.
    stepHi = 0.1
    H_is = np.arange(startHi, finishHi, stepHi)  # [mV]
    print("going to process:", H_is)
    ampMax = np.zeros(len(H_is))
    ampMin = np.zeros(len(H_is))
    for pos, H_i in enumerate(H_is):
        print('starting sim: H_i={}'.format(H_i))
        setH_i(H_i)
        f, p, raw_freqs, raw_power, raw_Data = runSim(Conn)
        data = signal.detrend(raw_Data, axis=0)
        ampMax[pos] = np.max(np.max(data, axis=0) - np.min(data, axis=0))
        ampMin[pos] = np.min(np.max(data, axis=0) - np.min(data, axis=0))
        print('Hi={} -> ampMin={}, ampMax={}'.format(H_i, ampMin[pos], ampMax[pos]))
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1)
    ax.plot(H_is, ampMax, lw=4, label='max', color='red')
    ax.plot(H_is, ampMin, lw=4, label='min', color='blue')
    fig.suptitle(r'Plot of the amplitudes vs. $H_i$')
    ax.legend(loc='lower left')
    ax.set_xlabel(r'$H_i$')
    ax.set_ylabel(r'Amplitude')
    ax.grid()
    plt.show()


# ======================================================================
# ======================================================================
# ======================================================================
if __name__ == '__main__':
    integrator.verbose = False
    # -------------------------- calibrateTau_iValues -> do this once and save the file...
    we = 300.
    # calibrateAndPlot_Tau_i_Values(we)
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
    # plotTimeResponseForSeveralWe('min')
    # plotMaxMinFrecsForAllWe()
    # plotSensitivityForAllWe()
    # tune_simRun_SimRun2_ForAllWe()
    # plotHist2DForAllWe()

    # ------------------------ Do a sensitivity test
    # we = 300.
    # tau_i = 49.  # [ms]
    # testMultipleTimes(1000, tau_i, we)

    # ------------------------ Check behaviour for inhibitory variables (Tau_i and H_i)
    we = 0.
    # plotMaxMinFrecsForAllTau_i(we)
    # plotErrorForAllTau_i(we)
    # setTau_i(54.)
    # plotTimeResponseForSeveral_H_i('max', we)
    # plotAmplitudeForAll_H_i(we)
    plotTimeResponseForSeveralTau_i(we, 'min', tau_is=np.arange(16,21,1))

    # -------------------------
    we = 300.
    # print("Running connectivity matrix with FIC control...")
    # tau_i_Start = brutefoce1DOptim(we)
    # tau_i_Start = 49.  # [ms]
    # tau_i = tau_i_Start * np.ones(N)
    # setTau_i(tau_i)
    # runAndPlotSim(we)
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

    # tau_i_End2 = Optim2(we, tau_i_Start)
    # setTau_i(tau_i_End2)
    # runAndPlotSim(we)
# ======================================================================
# ======================================================================
# ======================================================================
