# --------------------------------------------------------------------------------------
# Full pipeline from:
# [StefanovskiEtAl2019] Stefanovski, L., P. Triebkorn, A. Spiegler, M.-A. Diaz-Cortes, A. Solodkin, V. Jirsa,
#           R. McIntosh and P. Ritter; for the Alzheimer's disease Neuromigang Initiative (2019).
#           "Linking molecular pathways and large-scale computational modeling to assess candidate
#           disease mechanisms and pharmacodynamics in Alzheimer's disease." bioRxiv: 600205.
# Taken from the code at:
#           https://github.com/BrainModes/TVB_EducaseAD_molecular_pathways_TVB/blob/master/Educase_AD_study-LS-Surrogate.ipynb
#
# --------------------------------------------------------------------------------------
import numpy as np
from scipy import signal, stats
import scipy.io as sio
import os, csv
from pathlib import Path
import matplotlib.pyplot as plt
import time
from functions.Utils.decorators import loadOrCompute, loadCache, vectorCache

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import functions.Utils.plotSC as plotSC
from functions.Models import Abeta_StefanovskiEtAl2019 as Abeta
# from functions.Models import JansenRit as JR
AD_modality = 'A'
if AD_modality == 'A':
    import functions.Models.AD_DMF_A as adDMF
else:
    import functions.Models.AD_DMF_B as adDMF
neuronalModel = adDMF

base_folder = "./Data_Raw/from_Ritter"
save_folder = "./Data_Produced/AD"

import functions.Integrator_EulerMaruyama
integrator = functions.Integrator_EulerMaruyama
integrator.neuronalModel = neuronalModel
integrator.verbose = False
# Integration parms...
# dt = 5e-5
# tmax = 20.
# ds = 1e-4
# Tmaxneuronal = int((tmax+dt))

import functions.BOLDHemModel_Stephan2007 as Stephan2007
import functions.simulateFCD as simulateFCD
simulateFCD.integrator = integrator
simulateFCD.BOLDModel = Stephan2007
from functions import BalanceFIC
BalanceFIC.integrator = integrator

import functions.FC as FC
import functions.swFCD as swFCD
import functions.phFCD as phFCD
import functions.indPhDyn as indPhDyn
import functions.G_optim as G_optim
G_optim.integrator = integrator
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------
verbose = True
def processBOLDSignals(BOLDsignals, distanceSettings):
    NumSubjects = len(BOLDsignals)
    N = BOLDsignals[next(iter(BOLDsignals))].shape[0]  # get the first key to retrieve the value of N = number of areas

    # First, let's create a data structure for the distance measurement operations...
    measureValues = {}
    for ds in distanceSettings:  # Initialize data structs for each distance measure
        measureValues[ds] = distanceSettings[ds][0].init(NumSubjects, N)

    # Loop over subjects
    for pos, s in enumerate(BOLDsignals):
        if verbose: print('   BOLD {}/{} Subject: {} ({}x{})'.format(pos, NumSubjects, s, BOLDsignals[s].shape[0], BOLDsignals[s].shape[1]), end='', flush=True)
        signal = BOLDsignals[s]  # LR_version_symm(tc[s])
        start_time = time.clock()

        for ds in distanceSettings:  # Now, let's compute each measure and store the results
            measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
            applyFilters = distanceSettings[ds][1]  # whether we apply filters or not...
            procSignal = measure.from_fMRI(signal, applyFilters=applyFilters)
            measureValues[ds] = measure.accumulate(measureValues[ds], pos, procSignal)

        if verbose: print(" -> computed in {} seconds".format(time.clock() - start_time))

    for ds in distanceSettings:  # finish computing each distance measure
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        measureValues[ds] = measure.postprocess(measureValues[ds])

    return measureValues


# =====================================================================================
# Methods to input AD data
# =====================================================================================
def getClassifications(subjects):
    # ============================================================================
    # This code is to check whether we have the information of the type of subject
    # They can be one of:
    # Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer Disease (AD) or Significant Memory Concern (SMC)
    # ============================================================================
    input_classification = csv.reader(open(base_folder+"/subjects.csv", 'r'))
    classification = dict(filter(None,input_classification))
    mistery = []
    for subject in subjects:
        if subject in classification:
            print('Subject {} classified as {}'.format(subject, classification[subject]))
        else:
            print('Subject {} NOT classified'.format(subject))
            mistery.append(subject)
    print("Misisng {} subjects:".format(len(mistery)), mistery)
    print()
    return classification


modality = "Amyloid" # Amyloid or Tau
def loadSubjectData(subject, correctSCMatrix=True):
    sc_folder = base_folder+'/connectomes/'+subject+"/DWI_processing"
    SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
    if correctSCMatrix:
        SCnorm = correctSC(SC)
    else:
        SCnorm = np.log(SC + 1)

    pet_path = base_folder+"/PET_loads/"+subject+"/PET_PVC_MG/" + modality
    RH_pet = np.loadtxt(pet_path+"/"+"L."+modality+"_load_MSMAll.pscalar.txt")
    LH_pet = np.loadtxt(pet_path+"/"+"R."+modality+"_load_MSMAll.pscalar.txt")
    subcort_pet = np.loadtxt(pet_path+"/"+modality+"_load.subcortical.txt")[-19:]
    abeta_burden = np.concatenate((LH_pet,RH_pet,subcort_pet))

    fMRI_path = base_folder+"/fMRI/"+subject+"/MNINonLinear/Results/Restingstate"
    series = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt")
    subcSeries = np.loadtxt(fMRI_path+"/"+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt")
    fullSeries = np.concatenate((series,subcSeries))

    return SCnorm, abeta_burden, fullSeries


normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66  # yeah, a bit redundant... ;-)
maxNodeInput66 = 0.7275543904602363
def correctSC(SC):
    N = SC.shape[0]
    logMatrix = np.log(SC+1)
    # areasSC = logMatrix.shape[0]
    # avgSC = np.average(logMatrix)
    # === Normalization ===
    # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()  # normalize to the maximum
    # finalMatrix = logMatrix * avgHuman66/avgSC * (areasHuman66*areasHuman66)/(areasSC * areasSC)  # normalize to the avg AND the number of connections...
    maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
    finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput
    return finalMatrix


def analyzeMatrix(name, C):
    max, min, avg, std, maxNodeInput, avgNodeInput = FC.characterizeConnectivityMatrix(C)
    print(name + " => Shape:{}, Max:{}, Min:{}, Avg:{}, Std:{}".format(C.shape, max, min, avg, std), end='')
    print("  => impact=Avg*#:{}".format(avg*C.shape[0]), end='')
    print("  => maxNodeInputs:{}".format(maxNodeInput), end='')
    print("  => avgNodeInputs:{}".format(avgNodeInput))


# =================================================================================
# Sensibility calibration tests... repeat the SAME experiment over and over again,
# and make a histogram out of the results. It should look like a gaussian...
# =================================================================================
def testSingleSubjectMultipleTimes(subjectName, SC, BOLDsignal, times, distanceSettings, workPath):
    print("Testing single subject, multiple {} times...".format(times))
    processedBOLDemp = processBOLDSignals({subjectName: BOLDsignal}, distanceSettings)
    N = SC.shape[0]
    results = {}
    for ds in distanceSettings:
        results[ds] = []
    for t in range(times):
        print(f"\n\n ======== trial: {t}/{times} =======")
        error = computeValues(1, SC, BOLDsignal, processedBOLDemp, distanceSettings,
                              0, 1.0,  # do not use this part...
                              f'{workPath}/temp/{subjectName}/eval_at_{0}_with_{1.0}')
        for ds in distanceSettings:
            print(f" {ds}: {error[ds]};", end='', flush=True)
            results[ds].append(error[ds])
    print("\n")
    for ds in distanceSettings:
        avg = np.average(results[ds])
        std = np.std(results[ds])
        print(f"- {ds}:Average={avg}, std={std}")

        # ---- the histogram of the data
        n, bins, patches = plt.hist(results[ds], bins=10, facecolor='g', alpha=0.75)
        plt.xlabel('error')
        plt.ylabel('Probability')
        plt.title(f'Histogram of errors ({ds})')
        plt.text(60, .025, '$\mu$={}, $\sigma$={}'.format(avg, std))

        plt.show()


# =================================================================================
# test & plot Parm Sensitivity
# Note: I am a bit sloppy with these functions, may be made more efficient... ;-)
# =================================================================================
def testParmSensitivity(subjectName, NumSimSubjects, parmPos, AvgHC, # BOLD_fullSeries,
                        processedBOLDemp, parmRange, distanceSettings,
                        workPath):
    print("Testing single parameter: {}...\n".format(parmPos))

    # N, T = BOLD_fullSeries.shape
    results = {}
    for ds in distanceSettings:
        results[ds] = []
    for value in parmRange:
        simMeasures = computeValues(NumSimSubjects, AvgHC,  # BOLD_fullSeries,
                                    processedBOLDemp, distanceSettings,
                                    parmPos, value,
                                    f'{workPath}/temp/{subjectName}/eval_at_{parmPos}_with_{np.round(value, decimals=3)}-A.mat')
        for ds in distanceSettings:
            results[ds].append(simMeasures[ds].flatten())
            print(f" - {ds}: {simMeasures[ds]};", end='', flush=True)
        print("\n")
    return results


def pltSingleParmRange(subjectName, parmPos, AvgHC, BOLD_fullSeries, distanceSettings, workPath):
    NumSimSubjects = 10
    parmRange = np.arange(0, 2.01, 0.05)
    processedBOLDemp = processBOLDSignals({subjectName: BOLD_fullSeries}, distanceSettings)
    results = testParmSensitivity(subjectName, NumSimSubjects, parmPos, AvgHC,  # BOLD_fullSeries,
                                  processedBOLDemp,
                                  parmRange, distanceSettings, workPath)

    print("\n\n#####################################################################################################")
    plt.rcParams.update({'font.size': 22})
    # ax = plt.gca()
    for ds in distanceSettings:
        # color = next(ax._get_lines.prop_cycler)['color']
        plotFCpla, = plt.plot(parmRange, results[ds]) #, color=color)
        # plt.axvline(x=WEs[optimValDist[1]], ls='--', c=color)
        plotFCpla.set_label(ds)

    print("#####################################################################################################\n\n")
    plt.title(f"Parm {parmPos} variation")
    plt.ylabel("Fitting")
    plt.xlabel(f"parm {parmPos}")
    plt.legend()
    plt.show()


def pltFullParmRange(subjectName, AvgHC, BOLD_fullSeries, distanceSettings, parmRange, workPath):
    N, T = BOLD_fullSeries.shape
    NumSimSubjects = 10
    graph = np.arange(0, N, int(N/5)).reshape((3,2))
    processedBOLDemp = processBOLDSignals({subjectName: BOLD_fullSeries}, distanceSettings)
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(graph.shape[0], graph.shape[1])
    for ix, iy in np.ndindex(graph.shape):
        print(f"computing: graph({ix}, {iy}) -> parm={graph[ix,iy]} for parm in {parmRange}")

        results = testParmSensitivity(subjectName, NumSimSubjects, graph[ix,iy], AvgHC,
                                      # BOLD_fullSeries,
                                      processedBOLDemp,
                                      parmRange, distanceSettings, workPath)

        # print("\n\n#####################################################################################################")
        # ax = plt.gca()
        for ds in distanceSettings:
            # color = next(ax._get_lines.prop_cycler)['color']
            plotFCpla, = axs[ix,iy].plot(parmRange, results[ds]) #, color=color)
            # plt.axvline(x=WEs[optimValDist[1]], ls='--', c=color)
            plotFCpla.set_label(ds)
        axs[ix,iy].set(xlabel='parm value', ylabel='Fitting')
        axs[ix,iy].set_title(f"parm {graph[ix,iy]}")
        axs[ix,iy].legend()

    # print("#####################################################################################################\n\n")
    for ax in fig.get_axes():
        ax.label_outer()
    plt.title(f"Parm variation")
    plt.show()


# =================================================================================
# analyzeSignals
# =================================================================================
def analyzeSignals(neuroAct, simBOLD, adBOLD):
    N = neuroAct.shape[0]
    print(f"   max neuro_act rate:{np.max(np.mean(neuroAct,0))}")
    # --- Max rate is around 4.5 Hz for AvgHC
    # --- Max rate around 3.4 Hz for 011_S_4547 <- this is a bit surprising!
    print(f"   max simBOLD rate:{np.max(np.mean(simBOLD,0))}")
    print(f"   max adBOLD rate:{np.max(np.mean(adBOLD,0))}")
    ks_nA_sB = np.zeros(N)
    ks_nA_ad = np.zeros(N)
    ks_sB_ad = np.zeros(N)
    for area in np.arange(N):
        nA = signal.detrend(neuroAct[area]); nA /= nA.std()
        sB = signal.detrend(simBOLD[area]); sB /= sB.std()
        aB = signal.detrend(adBOLD[area]); aB /= aB.std()
        ks_nA_sB[area], pvalue = stats.ks_2samp(nA, sB)
        ks_nA_ad[area], pvalue = stats.ks_2samp(nA, aB)
        ks_sB_ad[area], pvalue = stats.ks_2samp(sB, aB)
    print(f"   KS(neuroAct_i, simBOLD_i): max={np.max(ks_nA_sB)}, argmax={np.argmax(ks_nA_sB)}, avg={np.average(ks_nA_sB)}")
    print(f"   KS(neuroAct_i, simBOLD_i): min={np.min(ks_nA_sB)}, argmin={np.argmin(ks_nA_sB)}")
    print(f"   KS(neuroAct_i, simBOLD_i): avg={np.average(ks_nA_sB)}")

    print(f"   KS(neuroAct_i, adBOLD_i): max={np.max(ks_nA_ad)}, argmax={np.argmax(ks_nA_ad)}, avg={np.average(ks_nA_ad)}")
    print(f"   KS(neuroAct_i, adBOLD_i): min={np.min(ks_nA_ad)}, argmin={np.argmin(ks_nA_ad)}")
    print(f"   KS(neuroAct_i, adBOLD_i): avg={np.average(ks_nA_ad)}")

    print(f"   KS(simBOLD_i,  adBOLD_i): max={np.max(ks_sB_ad)}, argmax={np.argmax(ks_sB_ad)}, avg={np.average(ks_sB_ad)}")
    print(f"   KS(simBOLD_i,  adBOLD_i): min={np.min(ks_sB_ad)}, argmin={np.argmin(ks_sB_ad)}")
    print(f"   KS(simBOLD_i,  adBOLD_i): avg={np.average(ks_sB_ad)}")


def plotCompareBOLDSignals(emp, SC, subjectName):
    from scipy import signal
    import seaborn as sns

    area = 0

    N, T = emp.shape
    emp_demeaned = signal.detrend(emp, axis=1)
    nE = signal.detrend(emp_demeaned[area]); nE /= nE.std()

    neuronalModel.ad = np.ones(N)
    integrator.recompileSignatures()
    bds = simulateFCD.simulateSingleSubject(SC, warmup=False).T
    bds_demeaned = signal.detrend(bds, axis=1)
    nS = signal.detrend(bds_demeaned[area]); nS /= nS.std()

    a = emp.flatten(); a /= a.max()
    b = bds.flatten(); b /= b.max()

    # ================================================================
    # Direct plot of BOLD activity...
    plot_emp, = plt.plot(nE, label="Empirical")
    plot_sim, = plt.plot(nS, label="Simulated")
    plt.xlabel('time [s]')
    plt.ylabel(f'BOLD activity ({subjectName})')
    plt.legend()
    plt.show()

    # ================================================================
    # distribution of BOLD activity for Empirical and Simulated of our subject
    # Let's have a look at the distribution of BOLD signal for one subject in the two different conditions. Do they look different?
    plt.hist(a, density=True, bins=30, alpha=0.7, label='Empirical')
    plt.hist(b, density=True, bins=30, alpha=0.7, label='Simulated')
    plt.xlabel(f'BOLD activity ({subjectName})')
    plt.legend()
    plt.show()

    # ================================================================
    # A fancier way of representing distributions is the so called violin plot, where the colored
    # are represents the density of data (inside each violin there is small box plot.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.violinplot(data=[a, b], cut=0, orient='h', scale='width', ax=ax)  # uses seaborn violinplot
    ax.set_yticklabels(['Empirical', 'Simulated'])
    ax.set_xlabel(f'BOLD ({subjectName})')
    plt.legend()
    plt.show()

    # ================================================================
    # We can also check the distribution of the demeaned BOLD signal for one ROI. Does it look like Gaussian?
    plt.hist(nE, bins=20, density=True, label='Empirical')
    plt.hist(nS, bins=20, density=True, label='Simulated')
    plt.xlabel(f'BOLD activity ({subjectName})')
    plt.legend()
    plt.show()


    # ================================================================
    # Let's evaluate the power spectrum of the BOLD signals.
    # calculate the power spectrum average over the ROIs for a single session
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(2)
    ax = plt.gca()

    freq = np.fft.fftfreq(T, d=1./2.4)[:int(T/2)]

    specEmp = np.fft.fft(emp_demeaned, axis=1)[:,:int(T/2)]
    av_pow_spec_Emp = np.abs(specEmp).mean(0)
    a1e,a0e = np.polyfit(freq[5:100], av_pow_spec_Emp[5:100], 1) # linear fit
    color = next(ax._get_lines.prop_cycler)['color']
    plotFCpla, = axs[0].plot(freq, av_pow_spec_Emp, color=color)  # data spectrum
    axs[0].plot(freq, a0e+a1e*freq, '--', color=color)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylim(ymin=1)
    axs[0].set_xlabel('frequency')
    axs[0].set_ylabel('power')
    axs[0].set_title('Empirical')

    specSim = np.fft.fft(bds_demeaned/bds_demeaned.max(), axis=1)[:,:int(T/2)]
    av_pow_spec_Sim = np.abs(specSim).mean(0)
    a1s,a0s = np.polyfit(freq[5:100], av_pow_spec_Sim[5:100], 1) # linear fit
    color = next(ax._get_lines.prop_cycler)['color']
    plotFCpla, = axs[1].plot(freq, av_pow_spec_Sim, color=color)  # data spectrum
    axs[1].plot(freq, a0s+a1s*freq, '--', color=color)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_ylim(ymin=1)
    axs[1].set_xlabel('frequency')
    axs[1].set_ylabel('power')
    axs[1].set_title('Simulated')

    plt.suptitle(f"Power Spectrum ({subjectName})")
    for ax in fig.get_axes():
        ax.label_outer()
    # plt.legend()
    plt.show()


# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
def fit_AD_Parms(subjectName, SC, targetBOLDSeries, BOLD_length, distanceSetting, cachePath, method, modality, trials=1):
    measure = distanceSetting[0]
    applyFilters = distanceSetting[1]
    processedBOLDemp = processBOLDSignals({subjectName: targetBOLDSeries}, {'dist': distanceSetting})['dist']
    angles_emp = measure.from_fMRI(targetBOLDSeries)
    start_time = None

    @vectorCache(cachePath=cachePath)
    def func(x):
        neuronalModel.M_i = x
        integrator.recompileSignatures()
        measureValues = measure.init(trials, N)
        for i in range(trials):
            bds = simulateFCD.simulateSingleSubject(SC, warmup=False).T
            procSignal = measure.from_fMRI(bds, applyFilters=applyFilters)
            measureValues = measure.accumulate(measureValues, i, procSignal)

        # ====== Measure the results...
        # measure = distanceSetting[0]  # FC, swFCD, phFCD, ...
        # applyFilters = distanceSetting[1]  # whether we apply filters or not...
        # procSignal = measure.from_fMRI(bds, applyFilters=applyFilters)
        # ====== Now, return the residuals...
        # r = processedBOLDemp - procSignal  # this will give a vector of the length of a phCFD upper triangular matrix. Then they will be compared using the l^2 metric
        # r = measure.distance(processedBOLDemp, procSignal)  # this is a float with the KS distance between the two phFCD vectors...
        measureValues = measure.postprocess(measureValues)
        r = measure.distance(measureValues, angles_emp)
        return r

    print("Starting optimization...")
    # Setting up simulation parms

    print("\n\n##################################################################")
    print(f"#  Modality: {modality}")
    print("##################################################################\n\n")
    adDMF.ADModality = modality
    adDMF.recompileSignatures()

    simulateFCD.Tmax = BOLD_length
    simulateFCD.recomputeTmaxneuronal()

    # init...
    (N, Tmax) = targetBOLDSeries.shape
    loadCache(cachePath, 100)
    x0 = np.ones(N)
    print("\n\n##################################################################")
    print(f"#  Fitting {subjectName} with {method}!!!")
    print("##################################################################\n\n")
    fx0 = func(x0)
    print(f"Starting value: f(np.ones())={fx0}")
    start_time = time.clock()
    if method in ['trf', 'dogbox', 'lm']:
        from scipy.optimize import least_squares
        if method == 'lm':
            lossFunc = 'linear'
        else:
            lossFunc = 'soft_l1' # method='lm' supports only 'linear' loss function.  # 'linear', 'soft_l1'
        print(f'(with {lossFunc} as lossFunc)')
        res = least_squares(func, x0, method=method, loss=lossFunc, f_scale=0.02,xtol=1e-10, # args=(processedBOLDemp,),
                            verbose=2)
    elif method in ['nelder-mead', 'Powell']:
        from scipy.optimize import minimize
        res = minimize(func, x0, method=method,
                       options={'xatol': 1e-8, 'disp': True})
    elif method in ['pBrent']:
        from functions.pBrent import pBrent
        res = pBrent(func, x0=x0, N=N, a=0.01 * np.ones(N), b=2 * np.ones(N))
    print("\n\n --- TIME: {} seconds ---".format(time.clock() - start_time), flush=True)

    # Print optimization results
    if method in ['pBrent']:
        from unittest.mock import Mock
        obj = Mock()
        for k, v in res.items():
            setattr(obj, k, v)
    else:
        obj = res
    print("success!!!") if obj.success else print("failure")
    print(f"\n\nFound: f={obj.fun} at ", obj.x)
    print(f"Evaluations: {obj.nfev} with status={obj.status} => {obj.message}")
    print(f'min[f(x)])={obj.x.min()}  max[f(x)]={obj.x.max()}')

    # Save results for later use
    fileName = f'Data_Produced/AD/AD_{subjectName}_fit-{method}_{modality}.mat'
    sio.savemat(fileName, {'AD': obj.x, 'value': obj.fun})


@loadOrCompute
def computeValues(NumSimSubjects, SCnorm,  # fullBOLDSeries,
                  processedBOLDemp, distanceSettings,
                  parmPos, value):  # analyzeSignals = False):
    print(f"========== computeValues for ad[{parmPos}]={value}  (with {NumSimSubjects} subjects)")
    N = SCnorm.shape[0]
    neuronalModel.ad = np.ones(N)
    neuronalModel.ad[parmPos] = value
    integrator.recompileSignatures()

    print("   --- BEGIN TIME ---")
    simulatedBOLDs = {}
    start_time = time.clock()
    for nsub in range(NumSimSubjects):  # trials. Originally it was 20.
        print("   Simulating subject {}/{}!!!".format(nsub, NumSimSubjects))

        # bds = simulateFCD.simulateSingleSubject(AvgHC, warmup=False).T
        # --- Simple test to check the max rate at the nodes during simulation... ---
        neuro_act = simulateFCD.computeSubjectSimulation(SCnorm, N, warmup=False)
        bds = simulateFCD.computeSubjectBOLD(neuro_act)

        # if analyzeSignals: analyzeSignals(neuro_act.T, bds.T, fullBOLDSeries)

        simulatedBOLDs[nsub] = bds
    simMeasures = processBOLDSignals(simulatedBOLDs, distanceSettings)
    print("   --- TOTAL TIME: {} seconds ---".format(time.clock() - start_time))

   # ---- and now compute the final FC, FCD, ... distances !!! ----
    fitting = {}
    for ds in distanceSettings:
        fitting[ds] = distanceSettings[ds][0].distance(simMeasures[ds], processedBOLDemp[ds])

    return fitting


def AD_pipeline(subjectName,
                distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                AvgHC,
                parmRange):
    N = AvgHC.shape[0]

    # ------------------------------------------------
    # Load individual Abeta and Tau PET SUVRs
    # ------------------------------------------------
    AD_SCnorm, AD_abeta, AD_fullSeries = loadSubjectData(subjectName)
    analyzeMatrix("AD SC norm", AD_SCnorm)
    print("   # of elements in AD SCnorm connectome: {}".format(AD_SCnorm.shape))
    # processedBOLDemp = processBOLDSignals({subjectName: AD_fullSeries}, distanceSettings)

    # ------------------------------------------------
    # Configure simulation
    # ------------------------------------------------
    we = 3.15  # Result from previous preprocessing using phFCD...
    J_fileName = f'Data_Produced/AD/FICWeights-AvgHC/BenjiBalancedWeights-{we}.mat'
    neuronalModel.we = we
    integrator.neuronalModel.J = sio.loadmat(J_fileName)['J'].flatten()  # Loads the optimized J for Feedback Inhibition Control [DecoEtAl2014]
    neuronalModel.M_e = np.ones(N)
    neuronalModel.M_i = np.ones(N)
    integrator.recompileSignatures()

    # testSingleSubjectMultipleTimes(subjectName, AvgHC, AD_fullSeries, 200, distanceSettings, save_folder)
    # --------------------------------------------------------------
    # Rerults (for 200 trials):
    # - FC:Average=-0.008666260484928823, std=0.038507334179143954
    # - swFCD:Average=0.8816466081810909, std=0.0018165349768813684
    # - phFCD:Average=0.15374840820533392, std=0.013982501401817268

    # Some functions to plot and analyze AD BOLD signals...
    # --------------------------------------------------------------
    # processedBOLDemp = processBOLDSignals({subjectName: AD_fullSeries}, distanceSettings)
    # testParmSensitivity(subjectName, NumSimSubjects=10, parmPos=312, AvgHC=AvgHC,
    #                     # BOLD_fullSeries=AD_fullSeries,
    #                     processedBOLDemp=processedBOLDemp,
    #                     parmRange=testRange, distanceSettings=distanceSettings)
    # pltSingleParmRange(subjectName, 312, AvgHC, AD_fullSeries, distanceSettings, workPath=save_folder)
    # plotCompareBOLDSignals(AD_fullSeries, AvgHC, subjectName)
    # pltFullParmRange(subjectName, AvgHC=AvgHC, BOLD_fullSeries=AD_fullSeries, distanceSettings=distanceSettings, parmRange=parmRange,
    #                  workPath=save_folder)
    # =======  Only for quick load'n plot test...
    import functions.Utils.plotFitting as plotFitting
    # parmPos = 0
    # ============== Simple single parm plotting
    # valueFilePaths = save_folder+'/temp/'+subjectName+'/eval_at_'+str(parmPos)+'_with_{}-A.mat'
    # plotFitting.loadAndPlot(valueFilePaths, distanceSettings,
    #                         WEs=np.arange(0.0, 2.001, 0.01),
    #                         empFilePath=None)
    # ============== Multiple parm plotting
    # valueFilePaths = save_folder+'/temp/'+subjectName+'/eval_at_{}_with_{}-A.mat'
    # parmRanges = np.arange(0, N, int(N/5))
    # plotFitting.pltFullParmRange(f"Parm variation for {subjectName} (A)", valueFilePaths, distanceSettings,
    #                              parms=parmRanges,
    #                              parmRange=np.arange(0.0, 2.001, 0.01),
    #                              graphShape=(2,3))

    # Fit params!!!
    # --------------------------------------------------------------
    # optMethod = 'trf'  # for least_squares: 'trf', 'dogbox', 'lm'
    # optMethod = 'nelder-mead'  # for (noisy) minimize: 'nelder-mead', 'Powell'
    optMethod = 'pBrent'
    cacheFilePath = save_folder+'/temp/'+subjectName+'_valueCache_'+optMethod+'_'+AD_modality+'.mat'
    measureToUse = 'indPhDyn'
    fit_AD_Parms(subjectName, AvgHC, AD_fullSeries, AD_fullSeries.shape[1], distanceSettings[measureToUse], cacheFilePath, optMethod, AD_modality, trials=10)


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
def processRangeValues(argv):
    import getopt
    try:
        opts, args = getopt.getopt(argv,'',["wStart=","wEnd=","wStep="])
    except getopt.GetoptError:
        print('AD_pipeline.py --wStart <wStartValue> --wEnd <wEndValue> --wStep <wStepValue>')
        sys.exit(2)
    wStart = 0.00001; wEnd = 2.00001; wStep = 0.1
    for opt, arg in opts:
        if opt == '-h':
            print('AD_pipeline.py -wStart <wStartValue> -wEnd <wEndValue> -wStep <wStepValue>')
            sys.exit()
        elif opt in ("--wStart"):
            wStart = float(arg)
        elif opt in ("--wEnd"):
            wEnd = float(arg)
        elif opt in ("--wStep"):
            wStep = float(arg)
    print(f'Input values are: wStart={wStart}, wEnd={wEnd}, wStep={wStep}')
    return wStart, wEnd, wStep


visualizeAll = True
if __name__ == '__main__':
    import sys
    wStart, wEnd, wStep = processRangeValues(sys.argv[1:])
    parmRange = np.arange(wStart, wEnd, wStep)

    plt.rcParams.update({'font.size': 22})

    # ------------------------------------------------
    # Load individual classification
    # ------------------------------------------------
    subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
    classification = getClassifications(subjects)
    HCSubjects = [s for s in classification if classification[s] == 'HC']
    ADSubjects = [s for s in classification if classification[s] == 'AD']

    # ------------------------------------------------
    # Load the Avg SC matrix
    # ------------------------------------------------
    AvgHC = sio.loadmat('Data_Produced/AD/AvgHC_SC.mat')['SC']
    analyzeMatrix("AvgHC norm", AvgHC)
    print("# of elements in AVG connectome: {}".format(AvgHC.shape))

    # ------------------------------------------------
    # Simulation settings
    # ------------------------------------------------
    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True), 'indPhDyn': (indPhDyn, True)}
    ADSubject = '114_S_6039'  # ADSubjects[0]

    AD_pipeline(ADSubject, distanceSettings, AvgHC, parmRange)

    print("DONE !!!")
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
