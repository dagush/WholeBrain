# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Tests the routines in OptimBOLD. Based on the papers
#
# * K.J. Friston, L. Harrison, and W. Penny,
#   Dynamic causal modelling, NeuroImage 19 (2003) 1273–1302
# * Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
#   Comparing hemodynamic models with DCM, NeuroImage 38 (2007) 387–401
#
# Later revisited in
# * Klaas Enno Stephan, Lars Kasper, Lee M. Harrison, Jean Daunizeau, Hanneke E.M. den Ouden, Michael Breakspear, and Karl J. Friston
#   Nonlinear Dynamic Causal Models for fMRI, Neuroimage. 2008 Aug 15; 42(2): 649–662.
#
# Also, check:
# * K.J. Friston, Katrin H. Preller, Chris Mathys, Hayriye Cagnan, Jakob Heinzle, Adeel Razi, Peter Zeidman
#   Dynamic causal modelling revisited, NeuroImage 199 (2019) 730–744
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
from Utils.FIC import BalanceFIC
from Projects.OptimBOLD import optimBOLD
from WholeBrain.Models import DynamicMeanField #, serotonin2A
from Integrators import EulerMaruyama as integrator
import simulateFCD
from Utils.BOLD import BOLDHemModel_Stephan2008
from WholeBrain.Utils import errorMetrics
from Observables import BOLDFilters
import time

integrator.neuronalModel = DynamicMeanField
DynamicMeanField.we = 2.1  # Global Coupling parameter

doIndividualSim = False

neuro_act = None
def simulateActivitySubject(C,N,we):
    global neuro_act

    if not doIndividualSim:
        if neuro_act is None:
            if not Path("Data_Produced/singleSubjectActivity.mat").is_file():
                print("Computing Single Subject Activity  -> Data_Produced/singleSubjectActivity.mat!!!")
                neuro_act = simulateFCD.computeSubjectSimulation(C, N, we)
                if not doIndividualSim:
                    print("   Finished subject neuro act simulation!!!")
                sio.savemat('Data_Produced/singleSubjectActivity.mat', {'neuro_act': neuro_act})
            else:
                print("Loading Data_Produced/singleSubjectActivity.mat !!!")
                neuro_act = sio.loadmat('Data_Produced/singleSubjectActivity.mat')['neuro_act']
        else:
            pass # print("Already done Single Subject Activity !!!")
    else:
        neuro_act = simulateFCD.computeSubjectSimulation(C, N, we)
    # return neuro_act


def processSubjectForEpsilon(signal):
    global neuro_act
    if doIndividualSim:  # each subject goes with his/her own sim...
        simulateActivitySubject(C, N, we)
    results = []
    epsilons = np.arange(0,1.5,0.1)
    for epsilon in epsilons:
        print("Epsilon:", epsilon, end=' ')
        BOLDHemModel_Stephan2008.epsilon = epsilon
        bds = simulateFCD.computeSubjectBOLD(neuro_act)
        bdsT = bds.T

        norm_bdsT = normalizeSignal(signal, bdsT)
        #print("sim:  ", np.average(norm_bdsT), "+/-", np.sqrt(np.var(norm_bdsT)))
        #print("fMRFI:", np.average(signal), "+/-", np.sqrt(np.var(signal)))
        error = errorMetrics.l2(signal, norm_bdsT)
        np.append(results, [error])
        print("Error:", error)
    plt.plot(epsilons, results, 'r-')
    plt.show()


def processAllSubjectsForEpsilon(Conditions):
    if not doIndividualSim: #Let's do a one-for-all simulation (done once, re-use for ever!)
        simulateActivitySubject(C, N, we)    # Loop over conditions and subjects
    #N = C.shape[0]
    for task in range(len(Conditions)):
        print("Task:", task, "(Condition:", Conditions[task], ")")
        for s in range(Subjects):
            signal = tc_aal[s, Conditions[task]]
            print("   Subject:", s, "Condition:", Conditions[task], "Signal.shape=", signal.shape)
            processSubjectForEpsilon(signal)


def pltSubjectBOLDForEpsilon(signal):
    global neuro_act
    simulateActivitySubject(C, N, we)
    results = []
    epsilons = np.arange(0,2.1,0.2)
    for epsilon in epsilons:
        print("Epsilon:", epsilon, end=' ')
        BOLDHemModel_Stephan2008.epsilon = epsilon
        bds = simulateFCD.computeSubjectBOLD(neuro_act, N)
        bdsT = bds.T

        norm_bdsT = normalizeSignal(signal, bdsT)
        #print("sim:  ", np.average(norm_bdsT), "+/-", np.sqrt(np.var(norm_bdsT)))
        #print("fMRFI:", np.average(signal), "+/-", np.sqrt(np.var(signal)))
        error = errorMetrics.l2(signal, norm_bdsT)
        results = np.append(results, [error])
        print("Error:", error)
    plt.plot(epsilons, results, 'r-')
    plt.show()


def processBrainAreasBOLDForEpsilon(brainAreaSignal, areas):
    global neuro_act
    simulateActivitySubject(C, N, we)
    epsilons = np.arange(0,2.1,0.1)
    results = [] #np.zeros([len(epsilons), len(areas)])
    minEpsilon=0
    minValue=1e99
    for epsilon in epsilons:
        print("Epsilon:", epsilon, end=' ')
        BOLDHemModel_Stephan2008.epsilon = epsilon
        bds = simulateFCD.computeSubjectBOLD(neuro_act, N, areasToSimulate=areas)
        bdsT = bds.T

        #for area, areaID in enumerate(areas):
        norm_bdsT = normalizeSignal(brainAreaSignal, bdsT)
        #print("sim:  ", np.average(norm_bdsT), "+/-", np.sqrt(np.var(norm_bdsT)))
        #print("fMRFI:", np.average(signal), "+/-", np.sqrt(np.var(signal)))
        error = errorMetrics.l2(brainAreaSignal, norm_bdsT)
        if error < minValue:
            minEpsilon = epsilon
            minValue = error
        results.append([error])
        print("Error:", error)
    print("Minimum:", minValue, "at:", minEpsilon,)
    plt.plot(epsilons, results, 'r-')
    plt.show()

    BOLDHemModel_Stephan2008.epsilon = minEpsilon
    bds = simulateFCD.computeSubjectBOLD(neuro_act, N, areasToSimulate=[0])
    bdsT = bds.T
    norm_bdsT = normalizeSignal(brainAreaSignal, bdsT)
    interval=range(len(brainAreaSignal))
    plt.plot(interval,brainAreaSignal.flatten(), 'b-',   interval,norm_bdsT.flatten(), 'r-')
    plt.suptitle('Area:'+str(areas[0]))
    plt.show()

def pltSubjectSimulatedBOLD(BOLDSignal):
    global neuro_act
    simulateActivitySubject(C, N, we)
    bds = simulateFCD.computeSubjectBOLD(neuro_act, N)
    bdsT = bds.T
    norm_bdsT = normalizeSignal(BOLDSignal, bdsT)
    interval=range(Tmax)
    for area in range(N):
        plt.plot(interval,norm_bdsT[area].flatten())
    plt.suptitle('BOLD simulation for single subject')
    plt.show()


# =================================================================================
# Sensibility calibration test... repeat the SAME experiment over and over again,
# and make a histogram out of the results. It should look like a gaussian...
# =================================================================================
def testSingleSubjectMultipleTimes(signal, times):
    print("Testing single subject, multiple {} times...".format(times))
    results = []
    for t in range(times):
        neuro_act = simulateFCD.computeSubjectSimulation(C, N)
        bds = simulateFCD.computeSubjectBOLD(neuro_act)
        bdsT = bds.T
        sim_filt = BOLDFilters.filterBrainArea(bdsT, 0)
        sim_filt /= np.std(sim_filt)
        error = errorMetrics.l2(signal, sim_filt)
        print("Trial: {} ({}) of {}".format(t+1, t, times), "Error:", error)
        results.append([error])
    avg = np.average(results)
    std = np.std(results)
    print("Average:", avg, "std:", std)

    # the histogram of the data
    n, bins, patches = plt.hist(results/avg*10, bins=10, facecolor='g', alpha=0.75)
    plt.xlabel('error')
    plt.ylabel('Probability')
    plt.title('Histogram of errors')
    plt.text(60, .025, '$\mu$={}, $\sigma$={}'.format(avg, std))
    #plt.xlim(40, 160)
    #plt.ylim(0, 0.03)
    #plt.grid(True)
    plt.show()

# =================================================================================
# =================================================================================
def simBrainAreaForOptimVars(N, area, keysAndValues):
    global neuro_act
    simulateActivitySubject(C, N, we)
    for key, value in keysAndValues.items():
        exec('BOLDHemModel2.'+key+' = '+str(value))
    bds = simulateFCD.computeSubjectBOLD(neuro_act, areasToSimulate=[area])
    bdsT = bds.T
    sim_filt = BOLDFilters.filterBrainArea(bdsT, 0)
    sim_filt /= np.std(sim_filt)
    return sim_filt


def fitPltErrorForBrainArea(BOLDSignal, area):
    N,T = BOLDSignal.shape
    simulateActivitySubject(C, N, we)
    # popt, pcov, finalError = optimBOLD.fitBOLDBrainAreaCatchingErrors(neuro_act, BOLDSignal, area)
    # perr = np.sqrt(np.diag(pcov))
    # print("Computed minimum:", popt, "value:", finalError, "Std Dev:", perr)

    interval=range(T)
    emp_filt = BOLDFilters.filterBrainArea(BOLDSignal, area)
    emp_filt /= np.std(emp_filt)
    empSignal, = plt.plot(interval, emp_filt, 'r--', label='empirical')
    sim_filtOptim = simBrainAreaForOptimVars(N, area, {'epsilon': 0.5}) # popt[0])
    simSignalOptim, = plt.plot(interval, sim_filtOptim, 'b-', label='simulated(epsilon={})'.format(0.5)) #popt[0]))
    plt.suptitle('Optimal Sim and Empirical BOLD for area {}'.format(area))
    plt.legend(handles=[empSignal, simSignalOptim])
    plt.show()


# ==============================================
#  fits a brain area and a subject BOLD signal
# ==============================================
def fitBrainArea(BOLDSignal, area):
    # Minimize!!!!
    print("  Area {} to be processed...".format(area))
    popt, pcov, finalError = optimBOLD.fitBOLDBrainAreaCatchingErrors(neuro_act, BOLDSignal, area)
    if not finalError == np.inf:
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = pcov
    print("  Computed minimum:", optimBOLD.pairVarsAndValues(popt), "\n           Std Dev:", optimBOLD.pairVarsAndValues(perr), "\n           value:", finalError)
    print("  Counter:", optimBOLD.evalCounter)
    return popt, perr, finalError


def fitSubject(BOLDSignal, rangeToTest):
    if rangeToTest is None:
        N,T = BOLDSignal.shape
        rangeToTest = range(N)
    else:
        N = len(rangeToTest)
    results = np.zeros([N, optimBOLD.numVars])
    perrs = np.zeros([N, optimBOLD.numVars])
    errors = np.zeros(N)
    print("Going to process {} areas".format(N))
    for pos, area in enumerate(rangeToTest):
        popt, perr, finalError = fitBrainArea(BOLDSignal, area)
        results[pos,:] = popt
        perrs[pos,:] = perr
        errors[pos] = finalError
    return results, perrs, errors


# ===========================================
#  processes and saves a subject BOLD signal
# ===========================================
def processSubject(BOLDSignal, ID, rangeToTest=None):
    print("Processing Subject:", ID)
    # init...
    N,T = BOLDSignal.shape
    simulateActivitySubject(C, N, we)
    # and fit !!!
    results, stds, errors = fitSubject(BOLDSignal, rangeToTest)
    print("SavingData_Produced/subject_{}_BOLD.mat".format(ID))
    sio.savemat('Data_Produced/subject_{}_BOLD.mat'.format(ID), {'parms': results, 'stds': stds, 'errors': errors})

# =================================================================================
# Synthetic signals test
# =================================================================================
def testSyntheticSignals(N, times):
    print("Testing single subject, multiple times...")
    simulateActivitySubject(C, N, we)
    allResults = np.zeros([len(varNames),times])
    allValues = np.zeros([len(varNames),times])
    allErrors = np.zeros([times])
    for t in range(times):
        print("begin Trial: {} ({}) of {}".format(t+1, t, times))
        # First, select some var values and use them to compute the target signal
        # Order: ['epsilon', 'alpha', 'tau', 'gamma', 'kappa']
        values = initialValues * np.random.rand(len(varNames))*1.9
        allValues[:,t] = values
        keysAndValues = optimBOLD.pairVarsAndValues(values)
        print("  Target values:", keysAndValues)
        test = simBrainAreaForOptimVars(N, 0, keysAndValues)
        BOLDTarget = test.reshape((1,len(test)))
        # And now, let's find it (yes, stupid but necessary to test the algorithm).
        results, stds, error = fitBrainArea(BOLDTarget, 0)
        # store the reults
        allResults[:,t] = results
        allErrors[t] = error

        # plot it!
        print("end Trial: {} ({}) of {}".format(t+1, t, times), "Error:", error)
        interval=range(len(test))
        testFit = simBrainAreaForOptimVars(N, 0, optimBOLD.pairVarsAndValues(results))
        plt.rcParams.update({'font.size': 22})
        plt.suptitle('Original vs fitted signal (test {} of {})'.format(t+1, times))
        plt.plot(interval, test, 'r-')
        plt.plot(interval, testFit, 'b-')
        plt.show()
    avg = np.average(allErrors)
    std = np.std(allErrors)
    print("Average:", avg, "std:", std)


# ===========================================
#  tests averaging over all subjects!!!
# ===========================================
def computeSingleBrainAreaForAllSubjects(tc_aal, area, condition):
    # init...
    print("function averageSingleBrainAreaForAllSubjects ({})".format(area))
    N,T = tc_aal[0, condition].shape
    simulateActivitySubject(C, N, we)
    allResults = np.zeros([Subjects, optimBOLD.numVars])
    allErrors = np.zeros([Subjects])
    for subject in range(Subjects):
        print("Processing area {} from subject {} !!!".format(area,subject))
        BOLDData = tc_aal[subject, condition]
        # and fit !!!
        results, stds, error = fitBrainArea(BOLDData, area)
        allResults[subject,:] = results
        allErrors[subject] = error
    print("Saving Data_Produced/BOLDConstForArea{}_For_{}Subjects.mat".format(area,subject))
    sio.savemat('Data_Produced/BOLDConstForArea{}_For_{}Subjects.mat'.format(area,subject), {'allResults': allResults})
    return allResults, allErrors

def averageSingleBrainAreaForAllSubjects(tc_aal, area, condition):
    allResults, allErrors = computeSingleBrainAreaForAllSubjects(tc_aal, area, condition)
    avgResult = np.mean(allResults, axis = 0)
    avgError = np.mean(allErrors)
    print("Mean result:", avgResult)
    print("Mean error:", avgError)
    return avgResult, avgError

def checkAveragedValuesOverSubjects(tc_aal, area, condition):
    print("function checkAveragedValuesOverSubjects ({} Subjects)".format(Subjects))
    print("======================================================")
    allErrors = np.zeros([Subjects])
    avgResult, avgError = averageSingleBrainAreaForAllSubjects(tc_aal, area, condition)
    print("Average error (area by area):", avgError)
    for subject in range(Subjects):
        print("Checking area {} from subject {} !!!".format(area,subject))
        BOLDData = tc_aal[subject, condition]
        emp_filt = BOLDFilters.filterBrainArea(BOLDData, area)
        emp_filt /= np.std(emp_filt)
        sim_filtOptim = simBrainAreaForOptimVars(N, area, optimBOLD.pairVarsAndValues(avgResult))
        error = errorMetrics.l2(emp_filt, sim_filtOptim)
        print("Error computed:", error)
        allErrors[subject] = error
    finalAvgError = np.mean(allErrors)
    print("Final avg error (all areas same parms):", finalAvgError)


# ===========================================
#  fits (and plots) a brain area BOLD signal
# ===========================================
def fitAndPltBrainArea(BOLDSignal, area):
    # init...
    N,T = BOLDSignal.shape
    simulateActivitySubject(C, N, we)
    # fit !!!
    t = time.time()
    popt, perr, finalError = fitBrainArea(BOLDSignal, area)
    print(np.round_(time.time() - t, 3), 'sec elapsed')
    # print("Area {} processed:".format(area))
    # print("Computed minimum:", pairVarsAndValues(popt), "\n         Std Dev:", pairVarsAndValues(perr, "\n         value:", finalError))
    # and plot.
    interval=range(T)
    emp_filt = BOLDFilters.filterBrainArea(BOLDSignal, area)
    emp_filt /= np.std(emp_filt)
    empSignal, = plt.plot(interval, emp_filt, 'r--', label='Empirical')
    sim_filtOptim = simBrainAreaForOptimVars(N, area, optimBOLD.pairVarsAndValues(popt))
    simSignalOptim, = plt.plot(interval, sim_filtOptim, 'b-', label='Simulated')
    plt.suptitle('Optimal Sim and Empirical BOLD for area {}'.format(area))
    plt.legend(handles=[empSignal, simSignalOptim])
    plt.show()


def pltErrorForBrainAreaFitOptimVariable(BOLDSignal, area, key, minValue = -0.5, maxValue=2.6, stepValue=0.1):
    from scipy.stats.stats import pearsonr
    emp_filt = BOLDFilters.filterBrainArea(BOLDSignal, area)
    emp_filt /= np.std(emp_filt)

    N,T = BOLDSignal.shape
    global neuro_act
    simulateActivitySubject(C, N, we)
    values = np.arange(minValue,maxValue,stepValue)
    results = [] #np.zeros([len(epsilons), len(areas)])
    resultsPearson = []
    minVar=-1
    minValue=1e99
    for value in values:
        print("Var value:", value, end=' ')
        sim_filt = simBrainAreaForOptimVars(N, area, {key: value})
        # simSignal, = plt.plot(interval, sim_filt, label='simulated(epsilon={}})'.format(epsilon))
        error = errorMetrics.l2(emp_filt, sim_filt)
        errorPearson = (1-pearsonr(emp_filt, sim_filt)[0])*20
        if error < minValue:
            minVar = value
            minValue = error
        results.append([error])
        resultsPearson.append([errorPearson])
        print("Error:", error, "Pearson.r:", errorPearson)
    print("'Manual' minimum:", minValue, "at:", minVar,)
    plt.rcParams.update({'font.size': 22})
    plt.suptitle('BOLD Error for '+ key +', area ({})'.format(area))
    plt.plot(values, results, 'r-')
    plt.plot(values, resultsPearson, 'b-')
    #popt, pcov, finalError = optimBOLD.fitBOLDBrainAreaCatchingErrors(neuro_act, BOLDSignal, area)
    #perr = np.sqrt(np.diag(pcov))
    #print("Computed minimum:", popt, "value:", finalError,"Std Dev:", perr)
    lineAt = minVar  # popt[0]   or   minVar
    plt.axvline(x=lineAt, color='g', linestyle='--')
    plt.show()

    interval=range(T)
    empSignal, = plt.plot(interval, emp_filt, 'r--', label='empirical')
    sim_filtOptim = simBrainAreaForOptimVars(N, area, {key: lineAt})
    simSignalOptim, = plt.plot(interval, sim_filtOptim, 'b-', label='simulated(epsilon={})'.format(lineAt))
    plt.suptitle('Optimal Sim and Empirical ('+key+') BOLD for area {}'.format(area))
    plt.legend(handles=[empSignal, simSignalOptim])
    plt.show()


def pltSimAndEmpiricalBrainAreaForVariousVariableValues(BOLDSignal, area, varKey, subplot=False):
    N,T = BOLDSignal.shape
    interval=range(T)
    plt.rcParams.update({'font.size': 22})

    if subplot:
        plt.subplot(2, 1, 1)
        plt.title("Empirical BOLD for area {}".format(area))
    emp_filt = BOLDFilters.filterBrainArea(BOLDSignal, area)
    emp_filt /= np.std(emp_filt)
    empSignal, = plt.plot(interval, emp_filt, 'r--', label='empirical')

    if subplot:
        plt.subplot(2, 1, 2)
        plt.title("Simulated BOLD for area {}".format(area))
    sim_filt00 = simBrainAreaForOptimVars(N, area, {varKey: 0.05})  # Some vars cannot be just 0.0...
    simSignal00, = plt.plot(interval, sim_filt00, 'b-', label='simulated('+varKey+'=0.05)')

    sim_filt05 = simBrainAreaForOptimVars(N, area, {varKey: 0.5})
    simSignal05, = plt.plot(interval, sim_filt05, 'g-', label='simulated('+varKey+'=0.5)')

    sim_filt10 = simBrainAreaForOptimVars(N, area, {varKey: 1.0})
    simSignal10, = plt.plot(interval, sim_filt10, 'c-', label='simulated('+varKey+'=1.0)')

    sim_filt15 = simBrainAreaForOptimVars(N, area, {varKey: 1.5})
    simSignal15, = plt.plot(interval, sim_filt15, 'm-', label='simulated('+varKey+'=1.5)')

    plt.suptitle('Sim and Empirical BOLD for single area, '+varKey)
    if not subplot:
        plt.legend(handles=[empSignal, simSignal00, simSignal05, simSignal10, simSignal15])
    else:
        plt.legend(handles=[simSignal00, simSignal05, simSignal10, simSignal15])
        plt.tight_layout()
    # plt.legend(handles=[simSignal00, simSignal05, simSignal10, simSignal15])
    # plt.legend(handles=[empSignal, simSignal05])
    plt.show()

    # just some computations to know a little bit better this stuff
    print("l^2 Error (0.05):", errorMetrics.l2(emp_filt, sim_filt00))
    print("l^2 Error (0.5):", errorMetrics.l2(emp_filt, sim_filt05))
    print("l^2 Error (1.0):", errorMetrics.l2(emp_filt, sim_filt10))
    print("l^2 Error (1.5):", errorMetrics.l2(emp_filt, sim_filt15))
    from scipy.stats.stats import pearsonr
    print("Pearson.r (0.05):", pearsonr(emp_filt, sim_filt00))
    print("Pearson.r (0.5):", pearsonr(emp_filt, sim_filt05))
    print("Pearson.r (1.0):", pearsonr(emp_filt, sim_filt10))
    print("Pearson.r (1.5):", pearsonr(emp_filt, sim_filt15))


def pltSimulatedNeuroActAndBOLDForArea(N, T, area):
    interval=range(T)

    plt.subplot(2, 1, 1)
    simulateActivitySubject(C, N, we)
    # factor = 5.
    # step = int(np.round(simulateFCD.TR/simulateFCD.dtt))/factor
    plt.rcParams.update({'font.size': 22})
    plotSignal = signal.decimate(neuro_act[:, area], 10) # Too high frequency. Let's subsample (filtering) it!
    plotSignal = signal.decimate(plotSignal, 10)
    plotSignal = signal.decimate(plotSignal, 10)
    plotSignal = signal.decimate(plotSignal, 2) # Factor of simulateFCD.TR/simulateFCD.dtt = 2000
    # intervalNeuroAct = np.arange(0, T+10, 1/factor) #, simulateFCD.dtt/simulateFCD.TR)
    # print("Takning step:", step, "between 0 and {}".format(T+10), "(factor:{})".format(factor))
    plt.title("Neural Activity for area {}".format(area))
    # neuroSignal, = plt.plot(interval, neuro_act[::step, area], 'b-')
    neuroSignal, = plt.plot(interval, plotSignal[10:], 'b-')

    plt.subplot(2, 1, 2)
    sim_filt = simBrainAreaForOptimVars(N, area, {'epsilon': 0.5})
    plt.title("BOLD for area {}".format(area))
    simSignal, = plt.plot(interval, sim_filt, 'r-')

    plt.suptitle('Comparison of Neural Activity and BOLD for area {}'.format(area))
    #plt.legend(handles=[neuroSignal, simSignal])
    plt.show()


def pltPreprocessingBrainAreaEmpiricalBOLD(BOLDSignal):
    # I use this function to understand, step by step, what pre-processing is done
    # to the BOLD signal. As far as I can tell, the steps are:
    # * Remove any linear trend (detrend) in the data
    # * remove strong artifacts (where strong is |value| > std(detrendedBOLDSignal)
    # * Apply a 2nd order Butterworth filter
    #
    plt.rcParams.update({'font.size': 22})
    N,T = BOLDSignal.shape
    interval=range(T)
    seed = 0
    ts = signal.detrend(BOLDSignal[seed, :])
    originalsignal, = plt.plot(interval, ts, 'r-', label='detrend(signal)')

    ts[ts>3*np.std(ts)] = 3*np.std(ts)   # Remove strong artefacts
    ts[ts<-3*np.std(ts)] = -3*np.std(ts)  # Remove strong artefacts
    artifacts, = plt.plot(interval, ts, 'b-', label='artifacts removed')

    TR = 2
    k = 2                             # 2nd order butterworth filter
    flp = .04                         # lowpass frequency of filter
    fhi = 0.07                        # highpass
    fnq = 1./(2.*TR)                  # Nyquist frequency
    Wn = [flp/fnq, fhi/fnq]           # butterworth bandpass non-dimensional frequency
    bfilt, afilt = signal.butter(k,Wn, btype='band', analog=False)   # construct the filter
    signal_filt = signal.filtfilt(bfilt, afilt, ts, padlen=3*(max(len(bfilt),len(afilt))-1))  # Band pass filter. padlen modified to get the same result as in Matlab
    filteredSignal, = plt.plot(interval, signal_filt, 'g-', label='filtered')

    plt.suptitle('Step-by-step BOLD pre-processing (single area)')
    plt.legend(handles=[originalsignal, artifacts, filteredSignal])
    plt.show()


# Load Structural Connectivity Matrix
print("Loading RawData/all_SC_FC_TC_76_90_116.mat")
sc90 = sio.loadmat('RawData/all_SC_FC_TC_76_90_116.mat')['sc90']  #load LSDnew.mat tc_aal
C=sc90/np.max(sc90[:])*0.2

Subjects = 15
Conditions = [1, 4]  # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...

# load fMRI data
print("Loading RawData/LSDnew.mat")
LSDnew = sio.loadmat('RawData/LSDnew.mat')  #load LSDnew.mat tc_aal
tc_aal = LSDnew['tc_aal']
(N, Tmax) = tc_aal[1,1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time


# ==== J is calculated this only once, then saved
if not Path("J_Balance.mat").is_file():
    print("Computing Data_Produced/J_Balance !!!")
    DynamicMeanField.J= BalanceFIC.Balance_J9(we, C) # This is the Feedback Inhibitory Control
    sio.savemat('Data_Produced/J_Balance.mat', {'J': J})  # save J_Balance J
else:
    print("Loading Data_Produced/J_Balance !!!")
    # ==== J can be calculated only once and then load J_Balance J
    DynamicMeanField.J = sio.loadmat('Data_Produced/J_Balance.mat')['J']


np.random.seed(42)
subjectToSimulate = 2
areaToSimulate = 0
conditionToUse = 1 # Use 1 for Placebo, 0 for LSD...
BOLDData = tc_aal[subjectToSimulate, Conditions[conditionToUse]]
Verbose = False

# -------- Sensibility test -----------
testSingleSubjectMultipleTimes(BOLDData, 200)

# -------- Actual computation tests... ---------------
#### pltSubjectBOLD(BOLDData)
# pltPreprocessingBrainAreaEmpiricalBOLD(BOLDData)

# pltSimAndEmpiricalBrainAreaForVariousVariableValues(BOLDData, areaToSimulate, "gamma", subplot=True)

# ==========================================================================================
# Plots the error curve for a given variable ('epsilon', 'alpha', 'tau', 'gamma', 'kappa')
# ==========================================================================================
# pltErrorForBrainAreaFitOptimVariable(BOLDData, areaToSimulate, 'alpha', minValue=0.01, maxValue=2.6, stepValue=0.01)
# pltErrorForBrainAreaFitOptimVariable(BOLDData, areaToSimulate, 'gamma', minValue=0.01, maxValue=2.6, stepValue=0.01)


# pltSimulatedNeuroActAndBOLDForArea(N, Tmax, areaToSimulate)
#### pltSubjectSimulatedBOLD(BOLDData)
#### processBrainAreasBOLD(BOLDData[areaToSimulate], [areaToSimulate]) #, [0,1])
#### processAllSubjects(Conditions)

# fitAndPltBrainArea(BOLDData, areaToSimulate)
# fitPltErrorForBrainArea(BOLDData, areaToSimulate)

# ==========================================================================================
# Computed vars for a single subject...
# ==========================================================================================
# processSubject(BOLDData, subjectToSimulate, range(5,10))

# ==========================================================================================
# Computed vars for single area for all subjects, and then verifies the effect
# of using averaged values..
# ==========================================================================================
# checkAveragedValuesOverSubjects(tc_aal, areaToSimulate, Conditions[conditionToUse])

# ==========================================================================================
# Tests our method with synthetic signals, to be sure of its performance...
# ==========================================================================================
#testSyntheticSignals(N, 2)

print("Evaluations done:", optimBOLD.evalCounter)
