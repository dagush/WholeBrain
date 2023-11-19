# --------------------------------------------------------------------------------------
# Full shuffling pipeline for AD subject processing
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
#
# --------------------------------------------------------------------------------------
import numpy as np
# from scipy import signal, stats
import scipy.io as sio
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time
from WholeBrain.Utils.decorators import loadOrCompute  #, loadSingleCache, loadMultipleCache, vectorCache
import dataLoader

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
from setup import *

import WholeBrain.Observables.phFCD as phFCD

fontSize = 10
import WholeBrain.Utils.p_values as p_values
p_values.fontSize = fontSize

from Utils.preprocessSignal import processEmpiricalSubjects

import functions_AD
functions_AD.neuronalModel = neuronalModel
functions_AD.integrator = integrator
functions_AD.simulateBOLD = simulateBOLD

TEST_MODE = 'TEST_ABeta_Tau'  # TEST_ABeta_Tau / TEST_Tau / TEST_ABeta
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


posShuffled = 1; posDefault = 2; posOptim = 3
def plotShuffling(result, label, yLimits = None):
    dataShuffled = [float(result[s]['shuffled']) for s in result.keys()]
    dataDef = [float(result[s]['default']) for s in result.keys()]  # result['011_S_4547']['default']
    dataOptim = [float(result[s]['optim']) for s in result.keys()]

    if TEST_MODE == 'TEST_Tau' or TEST_MODE == 'TEST_ABeta':
        dataHomogeneous = [float(result[s]['homogeneous']) for s in result.keys()]
        if TEST_MODE == 'TEST_Tau':
            middleLabel = 'Tau'
        else:
            middleLabel = 'ABeta'
        dataToTest = {'Shuffled': dataShuffled, middleLabel: dataOptim, 'Homog': dataHomogeneous, 'BEI': dataDef}
        dataLabels = list(dataToTest.keys())  #['Shuffled', middleLabel, 'Homog', 'BEI']
        p_values.plotComparisonAcrossLabels2(dataToTest,  #dataShuffled, dataOptim, dataHomogeneous, dataDef,
                                             dataLabels, graphLabel=f'Evaluation Comparison ({label})')
    elif TEST_MODE == 'TEST_ABeta_Tau':
        dataToTest = {'Shuffled': dataShuffled, 'Homogeneous': dataDef, 'Optim': dataOptim}
        dataLabels = list(dataToTest.keys())  # ['Shuffled', 'BEI', 'ABeta+Tau']
        p_values.plotComparisonAcrossLabels2(dataToTest,  #dataShuffled, dataDef, dataOptim,
                                             dataLabels, graphLabel=f'Shuffling Comparison ({label})')  #, yLimits=yLimits)
    else:
        raise NameError("Not implemented yet!")


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
        if verbose: print('   BOLD {}/{} Subject: {} ({}x{})'.format(pos, NumSubjects-1, s, BOLDsignals[s].shape[0], BOLDsignals[s].shape[1]), end='', flush=True)
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


# ========================================================================
# Do the shuffling computations
# Returns the average of the shuffling trials
# ========================================================================
# cachePath = None
# @vectorCache(filePath=cachePath)
def shufflingFunc(optParms):
    if TEST_MODE == 'TEST_ABeta_Tau':
        return functions_AD.func6D(optParms)
    elif TEST_MODE == 'TEST_ABeta':
        return functions_AD.func_full_ABeta(optParms)
    elif TEST_MODE == 'TEST_Tau':
        return functions_AD.func_full_Tau(optParms)


if TEST_MODE == 'TEST_ABeta_Tau':
    shufflingTrials = 10
else:
    shufflingTrials = 1


@loadOrCompute
def AD_Parms_test(subjectToTest, SCMatrix, targetBOLDSeries, BOLD_length, distanceSetting,
                  Abeta, tau, optimizedParms):
    # ======================= Initialization =======================
    # A few global vars to simplify parameter passing... shhh... don't tell anyone! ;-)
    # global measure, N, applyFilters, SC, angles_emp, subjectName
    functions_AD.SC = SCMatrix
    measureName = list(distanceSetting.keys())[0]
    measure = distanceSetting[measureName][0]
    functions_AD.measure = measure
    functions_AD.applyFilters = distanceSetting[measureName][1]
    print("Measuring empirical data from_fMRI...")
    outEmpFileName = save_folder+'/fNeuro_emp_AllHC.mat'
    functions_AD.processedEmp = processEmpiricalSubjects({subjectToTest: targetBOLDSeries},
                                                         distanceSettings,
                                                         outEmpFileName)[measureName]

    (N, Tmax) = targetBOLDSeries.shape
    functions_AD.N = N
    simulateBOLD.Tmax = BOLD_length
    simulateBOLD.recomputeTmaxneuronal()

    print("\n\n##################################################################")
    print(f"#  # of Shuffling Trials: {shufflingTrials}")
    print("##################################################################\n\n")
    results = np.zeros((shufflingTrials))
    for n in range(shufflingTrials):
        trialCounter = 0
        valueShuffled = measure.ERROR_VALUE
        while valueShuffled == measure.ERROR_VALUE:  # Retry until we do not get an error...
            # Statistically, it cannot hang the computer indefinitely... I hope! ;-)
            trialCounter += 1
            print(f"Times retried this one (trial {n}): {trialCounter}")
            if TEST_MODE == 'TEST_ABeta_Tau' or TEST_MODE == 'TEST_ABeta':
                np.random.shuffle(Abeta)
            DMF_AD.Abeta = Abeta
            if TEST_MODE == 'TEST_ABeta_Tau' or TEST_MODE == 'TEST_Tau':
                np.random.shuffle(tau)
            DMF_AD.Tau = tau
            valueShuffled = shufflingFunc(optimizedParms['parms'].flatten())
        print(f"Result {n}: {valueShuffled}")
        results[n] = valueShuffled

    print("\n\n##################################################################")
    print(f"#  Homogeneous Test                                              #")
    print("##################################################################\n\n")
    trialCounter = 0
    valueHomogeneous = measure.ERROR_VALUE
    while valueHomogeneous == measure.ERROR_VALUE:  # Retry until we do not get an error...
        # Statistically, it cannot hang the computer indefinitely... I hope! ;-)
        trialCounter += 1
        print(f"Times retried this one (trial {n}): {trialCounter}")
        if TEST_MODE == 'TEST_ABeta_Tau' or TEST_MODE == 'TEST_ABeta':
            AbetaAvg = np.average(Abeta)
            Abeta = np.ones(Abeta.size) * AbetaAvg
        DMF_AD.Abeta = Abeta
        if TEST_MODE == 'TEST_ABeta_Tau' or TEST_MODE == 'TEST_Tau':
            tauAvg = np.average(tau)
            tau = np.ones(tau.size) * tauAvg
        DMF_AD.Tau = tau
        valueHomogeneous = shufflingFunc(optimizedParms['parms'].flatten())

    print(f"got the following results: {results}")
    print(f"which average to {np.average(results)}")
    return {subjectToTest:{'shuffling': np.average(results), 'homogeneous': valueHomogeneous}}


# ========================================================================
# If we have an optimization result (we may not have it... yet!), let's
# shuffle the burden ABeta and Tau) and let's see how it goes...
# ========================================================================
def AD_check(subjectName,
             distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
             AvgHC):
    if TEST_MODE == 'TEST_ABeta_Tau':
        optMethod = 'gp_minimize-cheat'
    elif TEST_MODE == 'TEST_ABeta':
        optMethod = 'gp_minimize-cheat-full_ABeta'
    elif TEST_MODE == 'TEST_Tau':
        optMethod = 'gp_minimize-cheat-full_tau'
    fileName = save_folder + f'/AD_{subjectName}_fittingResult-{optMethod}.mat'
    defaultFileName = save_folder + f'/AD_{subjectName}_fittingResult-gp_minimize-cheat.mat'

    if Path(fileName).is_file():  # check if fitted parameters file exists
        defaultParms = sio.loadmat(defaultFileName)
        optimizedParms = sio.loadmat(fileName)
        print("\n\n##################################################################")
        print(f"#  Evaluating {subjectName} at shuffled burden!!!")
        print(f"#  With parms: {optimizedParms['parms']}")
        print(f"#  Default Value: {optimizedParms['default']}, OptimValue: {optimizedParms['value']}")

        N = AvgHC.shape[0]

        # ------------------------------------------------
        # Load individual Abeta and Tau PET SUVRs
        # ------------------------------------------------
        AD_SCnorm, AD_Abeta, AD_tau, AD_fullSeries = dataLoader.loadSubjectData(subjectName)
        AD_fullSeries = dataLoader.cutTimeSeriesIfNeeded(AD_fullSeries)
        # AD_Auxiliar.analyzeMatrix("AD SC norm", AD_SCnorm)
        # print("   # of elements in AD SCnorm connectome: {}".format(AD_SCnorm.shape))
        # processedBOLDemp = processBOLDSignals({subjectName: AD_fullSeries}, distanceSettings)

        # ------------------------------------------------
        # Configure simulation
        # ------------------------------------------------
        we = 3.1  # Result from previous preprocessing using phFCD...
        J_fileName = save_folder + f'/FICWeights-AvgHC/BenjiBalancedWeights-{we}.mat'
        neuronalModel.setParms({'SC': AD_SCnorm, 'we': we})  # neuronalModel.we = we
        neuronalModel.setParms({'J': sio.loadmat(J_fileName)['J'].flatten()})  # Loads the optimized J for Feedback Inhibition Control [DecoEtAl2014]
        neuronalModel.setParms({'M_e': np.ones(N)})
        neuronalModel.setParms({'M_i': np.ones(N)})
        integrator.recompileSignatures()

        # ------------------------------------------------
        # Now, the specific AD simulation
        # ------------------------------------------------
        testType = '' if TEST_MODE == 'TEST_ABeta_Tau' else '-'+TEST_MODE
        shufflefileName = save_folder + f'/AD_{subjectName}_shufflingResult-{optMethod}{testType}.mat'
        result = AD_Parms_test(subjectName, AvgHC, AD_fullSeries, AD_fullSeries.shape[1], distanceSettings,
                               AD_Abeta, AD_tau, optimizedParms,
                               shufflefileName)
        print(f"#  Result: {result}")
        print(f"Saved to: {shufflefileName}")
        print("##################################################################\n\n")
        if TEST_MODE == 'TEST_Tau' or TEST_MODE == 'TEST_ABeta':
            return {subjectName: {'shuffled': result[subjectName]['shuffling'],
                                  'default': defaultParms['default'],
                                  'optim': optimizedParms['value'],
                                  'homogeneous': result[subjectName]['homogeneous']}}
        elif TEST_MODE == 'TEST_ABeta_Tau':
            return {subjectName: {'shuffled': result[subjectName],
                                  'default': optimizedParms['default'],
                                  'optim': optimizedParms['value']}}
        else:
            raise NameError("Not implemented yet!")
    else:
        print("\n\n##################################################################")
        print(f"#  Optimized Parms NOT Found for {subjectName} at shuffled burden!!!")
        print(f"#  Looked for {fileName}")
        print("##################################################################\n\n")
        return None


# ========================================================================
# Loop over subjects to test the fitting...
# ========================================================================
def shufflingTest(subjects, distanceSettings, AvgHC):
    results = {}
    for s in subjects:
        check = AD_check(s, distanceSettings, AvgHC)
        if check is not None:
            results[s] = check[s]
    return results


visualizeAll = True
if __name__ == '__main__':
    # import sys
    # group = processParmValues(sys.argv[1:])

    plt.rcParams.update({'font.size': 12})

    # ------------------------------------------------
    # Load the Avg SC matrix
    # ------------------------------------------------
    AvgHC = sio.loadmat(save_folder + '/AvgHC_SC.mat')['SC']
    dataLoader.analyzeMatrix("AvgHC norm", AvgHC)
    print("# of elements in AVG connectome: {}".format(AvgHC.shape))

    # ------------------------------------------------
    # Simulation settings
    # ------------------------------------------------
    distanceSettings = {'phFCD': (phFCD, True)}

    # ------------------------------------------------
    # Run shuffling pipeline tests for the ADSubjects
    # ------------------------------------------------
    if TEST_MODE == 'TEST_Tau' or TEST_MODE == 'TEST_ABeta':
        testSet = ['MCI']
    elif TEST_MODE == 'TEST_ABeta_Tau':
        testSet = ['AD', 'MCI', 'HC']
    else:
        raise NameError("To be developped!")
    for label in testSet:
        setToTest = [s for s in classification if classification[s] == label]
        print(f" Running for: {label}")
        result = shufflingTest(setToTest, distanceSettings, AvgHC)
        plotShuffling(result, label, [0.0, 1.0])

    # ------------------------------------------------
    # Done !!!
    # ------------------------------------------------
    print("DONE !!!")
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
