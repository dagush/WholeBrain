# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Inspired from the code (fgain_Neuro.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from numba import jit
from functions.Utils.decorators import loadOrCompute
import time

verbose = True

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
# import functions.Models.DynamicMeanField as neuronalModel
# import functions.Models.serotonin2A as serotonin2A
# import functions.Integrator_EulerMaruyama as integrator
# integrator.neuronalModel = neuronalModel
# integrator.verbose = False
integrator = None
# import functions.BOLDHemModel_Stephan2007 as Stephan2007

import functions.Observables.FC as FC
import functions.Observables.swFCD as FCD

import functions.BalanceFIC as BalanceFIC
# BalanceFIC.integrator = integrator
# BalanceFIC.baseName = "Data_Produced/SC90/J_Balance_we{}.mat"

simulateBOLD = None

# set BOLD filter settings
# import functions.BOLDFilters as filters
# filters.k = 2                             # 2nd order butterworth filter
# filters.flp = .01                         # lowpass frequency of filter
# filters.fhi = .1                          # highpass
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# def recompileSignatures():
#     # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
#     # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
#     # print("\n\nRecompiling signatures!!!")
#     # serotonin2A.recompileSignatures()
#     integrator.recompileSignatures()


# def LR_version_symm(TC):
#     # returns a symmetrical LR version of the input matrix
#     N = TC.shape[0]  # 90 for AAL 90x90
#     odd = np.arange(0,N,2)
#     even = np.arange(1,N,2)[::-1]  # sort 'descend'
#     symLR = np.zeros((N,TC.shape[1]))
#     symLR[0:int(N/2.),:] = TC[odd,:]
#     symLR[int(N/2.):N,:] = TC[even,:]
#     return symLR


def processBOLDSignals(BOLDsignals, distanceSettings):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # distanceSettings is a dictionary of {distanceMeasureName: distanceMeasurePythonModule}
    NumSubjects = len(BOLDsignals)
    N = BOLDsignals[next(iter(BOLDsignals))].shape[0]  # get the first key to retrieve the value of N = number of areas

    # First, let's create a data structure for the distance measurement operations...
    measureValues = {}
    for ds in distanceSettings:  # Initialize data structs for each distance measure
        measureValues[ds] = distanceSettings[ds][0].init(NumSubjects, N)

    # Loop over subjects
    for pos, s in enumerate(BOLDsignals):
        if verbose: print('   BOLD {}/{} Subject: {} ({}x{})'.format(pos+1, NumSubjects, s, BOLDsignals[s].shape[0], BOLDsignals[s].shape[1]), end='', flush=True)
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


# ============== a practical way to save recomputing necessary (but lengthy) results ==========
@loadOrCompute
def processEmpiricalSubjects(BOLDsignals, distanceSettings):
    return processBOLDSignals(BOLDsignals, distanceSettings)


# ==========================================================================
# ==========================================================================
# ==========================================================================
# ---- convenience method, to parallelize the code (someday)
@loadOrCompute
def distanceForOne_G(we, C, modelParms, N, NumSimSubjects,
                     distanceSettings):  # distanceSettings is a dictionary of {name: (distance module, apply filters bool)}
    integrator.neuronalModel.setParms(modelParms)
    integrator.recompileSignatures()

    print(f"   --- BEGIN TIME @ we={we} ---")
    simulatedBOLDs = {}
    start_time = time.clock()
    for nsub in range(NumSimSubjects):  # trials. Originally it was 20.
        print("   Simulating we={} -> subject {}/{}!!!".format(we, nsub, NumSimSubjects))
        bds = simulateBOLD.simulateSingleSubject(C, warmup=False).T
        simulatedBOLDs[nsub] = bds

    dist = processBOLDSignals(simulatedBOLDs, distanceSettings)
    dist["We"] = we
    print("   --- TOTAL TIME: {} seconds ---".format(time.clock() - start_time))
    return dist


def distanceForAll_G(C, tc, modelParms, NumSimSubjects,
                     distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                     WEs,  # wStart=0.0, wEnd=6.0, wStep=0.05,
                     outFilePath=None,
                     fileNameSuffix=''):
    if verbose:
        import functions.Utils.decorators as deco
        deco.verbose = True
    NumSubjects = len(tc)
    N = tc[next(iter(tc))].shape[0]  # get the first key to retrieve the value of N = number of areas
    print('tc({} subjects): each entry has N={} regions'.format(NumSubjects, N))

    processed = processEmpiricalSubjects(tc, distanceSettings, outFilePath+'/fNeuro_emp'+fileNameSuffix+'.mat')

    # WEs = np.arange(wStart, wEnd+wStep, wStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of wes depends on the conectome.
    numWEs = len(WEs)

    fitting = {}
    for ds in distanceSettings:
        fitting[ds] = np.zeros((numWEs))

    # Model Simulations
    # -----------------
    print('\n\n ====================== Model Simulations ======================\n\n')
    for pos, we in enumerate(WEs):  # iteration over the values for G (we in this code)
        # ---- Perform the simulation of NumSimSubjects ----
        simMeasures = distanceForOne_G(we, C, modelParms[we], N, NumSimSubjects, distanceSettings,
                                       outFilePath + '/fitting_we{}.mat'.format(np.round(we, decimals=3)))

        # ---- and now compute the final FC, FCD, ... distances for this G (we)!!! ----
        print(f"{we}/{WEs[-1]}:", end='', flush=True)
        for ds in distanceSettings:
            fitting[ds][pos] = distanceSettings[ds][0].distance(simMeasures[ds], processed[ds])
            print(f" {ds}: {fitting[ds][pos]};", end='', flush=True)
        print("\n")

    print("\n\n#####################################################################################################")
    print(f"# Results (in ({WEs[0]}, {WEs[-1]})):")
    for ds in distanceSettings:
        optimValDist = distanceSettings[ds][0].findMinMax(fitting[ds])
        print(f"# Optimal {ds} = {optimValDist[0]} @ {np.round(WEs[optimValDist[1]], decimals=3)}")
    print("#####################################################################################################\n\n")

    print("DONE!!!")
    return fitting


# ==========================================================================
# ==========================================================================
# ==========================================================================
if __name__ == '__main__':
    # Load Structural Connectivity Matrix
    print("Loading Data_Raw/all_SC_FC_TC_76_90_116.mat")
    sc90 = sio.loadmat('Data_Raw/all_SC_FC_TC_76_90_116.mat')['sc90']
    C = sc90/np.max(sc90[:])*0.2  # Normalization...

    NumSubjects = 15  # Number of Subjects in empirical fMRI dataset
    print("Simulating {} subjects!".format(NumSubjects))
    Conditions = [4, 1]  # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...

    #load fMRI data
    print("Loading Data_Raw/LSDnew.mat")
    LSDnew = sio.loadmat('Data_Raw/LSDnew.mat')  #load LSDnew.mat tc_aal
    tc_aal = LSDnew['tc_aal']

    # distanceForAll_G(C, tc_aal, 'Data_Produced/error_{}.mat')
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
