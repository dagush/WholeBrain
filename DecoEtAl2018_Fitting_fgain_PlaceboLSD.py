# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Taken from the code (fgain_Placebo.m and fgain_LCD.m) from:
#
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
# from pathlib import Path
# from numba import jit
import time

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
# Setup for Serotonin 2A-based DMF simulation!!!
import functions.Models.DynamicMeanField as neuronalModel
import functions.Models.serotonin2A as serotonin2A
neuronalModel.He = serotonin2A.phie
neuronalModel.Hi = serotonin2A.phii
# ----------------------------------------------
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = neuronalModel
integrator.verbose = False
import functions.BOLDHemModel_Stephan2007 as Stephan2007
import functions.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007
import functions.simulateFCD as simulateFCD
simulateFCD.integrator = integrator
simulateFCD.BOLDModel = Stephan2007

import functions.Observables.FC as FC
import functions.Observables.swFCD as swFCD

import functions.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator
# BalanceFIC.baseName = "Data_Produced/SC90/J_Balance_we{}.mat"

import functions.G_optim as G_optim
G_optim.simulateBOLD = simulateBOLD
G_optim.integrator = integrator

# set BOLD filter settings
import functions.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .01                         # lowpass frequency of filter
filters.fhi = .1                          # highpass

PLACEBO_cond = 4; LSD_cond = 1   # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    print("\n\nRecompiling signatures!!!")
    serotonin2A.recompileSignatures()
    integrator.recompileSignatures()


def LR_version_symm(TC):
    # returns a symmetrical LR version of AAL 90x90 matrix
    odd = np.arange(0,90,2)
    even = np.arange(1,90,2)[::-1]  # sort 'descend'
    symLR = np.zeros((90,TC.shape[1]))
    symLR[0:45,:] = TC[odd,:]
    symLR[45:90,:] = TC[even,:]
    return symLR


def transformEmpiricalSubjects(tc_aal, cond, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        # transformed[s] = np.zeros(tc_aal[0,cond].shape)
        transformed[s] = LR_version_symm(tc_aal[s,cond])
    return transformed


# ==========================================================================
# ==========================================================================
# ==========================================================================
# IMPORTANT: This function was created to reproduce Deco et al.'s 2018 code for Figure 3B.
# ACtually, this only performs the fitting which gives the value of S_E (wge in the original
# code) to use for further computations (e.g., plotting Figure 3B)
def fitting_ModelParms(tc_transf, suffix):
    # Load Structural Connectivity Matrix
    print("Loading Data_Raw/all_SC_FC_TC_76_90_116.mat")
    sc90 = sio.loadmat('Data_Raw/all_SC_FC_TC_76_90_116.mat')['sc90']
    C = sc90/np.max(sc90[:])*0.2  # Normalization...

    neuronalModel.we = 2.1  # Global Coupling parameter, found in the DecoEtAl2018_Prepro_* file...

    # Load Regional Drug Receptor Map
    print('Loading Data_Raw/mean5HT2A_bindingaal.mat')
    mean5HT2A_aalsymm = sio.loadmat('Data_Raw/mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
    serotonin2A.Receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()
    serotonin2A.wgaine = 0.
    serotonin2A.wgaini = 0.

    # TCs = np.zeros((len(Conditions), NumSubjects, N, Tmax))
    # N_windows = int(np.ceil((Tmax-FCD.windowSize) / 3))  # len(range(0,Tmax-30,3))
    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True)}  #'phFCD': (phFCD, True)}

    # tc_transf_PLA = transformEmpiricalSubjects(tc_aal, PLACEBO_cond, NumSubjects)  # PLACEBO
    processed = G_optim.processEmpiricalSubjects(tc_transf, distanceSettings, "Data_Produced/SC90/fNeuro_emp"+suffix+".mat")
    # FCemp = FCemp_cotsampling['FC']; cotsampling = FCemp_cotsampling['swFCD'].flatten()

    # tc_transf_LSD = transformEmpiricalSubjects(tc_aal, LSD_cond, NumSubjects)  # LSD
    # FCemp_cotsampling_LSD = G_optim.processEmpiricalSubjects(tc_transf_LSD, distanceSettings, "Data_Produced/SC90/fNeuro_emp_LCD.mat")  # LCD
    # FCemp_LSD = FCemp_cotsampling_LSD['FC']; cotsampling_LSD = FCemp_cotsampling_LSD['swFCD'].flatten()

    # %%%%%%%%%%%%%%% Set General Model Parameters
    # dtt   = 1e-3   # Sampling rate of simulated neuronal activity (seconds)
    # dt    = 0.1
    # DMF.J     = np.ones(N,1)
    # Tmaxneuronal = (Tmax+10)*2000;
    J_fileNames = "Data_Produced/SC90/J_Balance_we{}.mat"  # "Data_Produced/SC90/J_test_we{}.mat"
    baseName = "Data_Produced/SC90/fitting_S_E{}.mat"

    S_EStart = 0.
    S_EStep = 0.002  # 0.002
    S_EEnd = 0.4 + S_EStep
    S_Es = np.arange(S_EStart, S_EEnd, S_EStep)
    numS_Es = len(S_Es)

    fitting = {}
    for ds in distanceSettings:
        fitting[ds] = np.zeros((numS_Es))

    # Model Simulations
    # ----------------------------
    for pos, S_E in enumerate(S_Es):  # iteration over values for G (we in this code)
        serotonin2A.wgaine = S_E
        # recompileSignatures()

        balancedParms = BalanceFIC.Balance_J9(neuronalModel.we, C, False,
                                              J_fileNames.format(np.round(neuronalModel.we, decimals=2)))  # Computes (and sets) the optimized J for Feedback Inhibition Control [DecoEtAl2014]
        balancedParms['J'] = balancedParms['J'].flatten()
        balancedParms['we'] = balancedParms['we'].flatten()

        simMeasures = G_optim.distanceForOne_G(neuronalModel.we, C, balancedParms, N, NumSubjects,
                                               distanceSettings,
                                               baseName.format(np.round(neuronalModel.we, decimals=3)))
        # FC_simul = FCsimul_cotsamplingsim['FC']
        # cotsampling_sim = FCsimul_cotsamplingsim['swFCD'].flatten()

        for ds in distanceSettings:
            fitting[ds][pos] = distanceSettings[ds][0].distance(simMeasures[ds], processed[ds])
            print(f" {ds}: {fitting[ds][pos]};", end='', flush=True)
        print("\n")

        print("{}/{}: FCDfitt = {}; FCfitt = {}\n".format(S_E, S_EEnd, fitting['FCD'][pos], fitting['FC'][pos]))  # FCDfitt_PLA[pos], fitting_PLA[pos]))

    print("\n\n#####################################################################################################")
    print(f"# Results (in ({S_Es[0]}, {S_Es[-1]})):")
    for ds in distanceSettings:
        optimValDist = distanceSettings[ds][0].findMinMax(fitting[ds])
        print(f"# Optimal {ds} = {optimValDist[0]} @ {np.round(S_Es[optimValDist[1]], decimals=3)}")
    print("#####################################################################################################\n\n")

    # filePath = 'Data_Produced/DecoEtAl2018_fneuro'+suffix+'.mat'
    # sio.savemat(filePath, #{'JI': JI})
    #             {'we': S_Es,
    #              # 'fitting_LSD': fitting['fitting_LSD,
    #              'fitting_PLA': fitting_PLA,
    #              # 'FCDfitt_LSD': FCDfitt_LSD,
    #              'FCDfitt_PLA': FCDfitt_PLA
    #             })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');

    print("DONE!!!")

if __name__ == '__main__':
    #load fMRI data
    print("Loading Data_Raw/LSDnew.mat")
    LSDnew = sio.loadmat('Data_Raw/LSDnew.mat')  #load LSDnew.mat tc_aal
    tc_aal = LSDnew['tc_aal']
    (N, Tmax) = tc_aal[1,1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time
    print(f'tc_aal is {tc_aal.shape} and each entry has N={N} regions and Tmax={Tmax}')
    NumSubjects = 15  # Number of Subjects in empirical fMRI dataset
    print("Simulating {} subjects!".format(NumSubjects))
    # Conditions = [4, 1]  # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...

    tc_transf_PLA = transformEmpiricalSubjects(tc_aal, PLACEBO_cond, NumSubjects)  # PLACEBO
    fitting_ModelParms(tc_transf_PLA, '_PLA')

    tc_transf_LSD = transformEmpiricalSubjects(tc_aal, LSD_cond, NumSubjects)  # LSD
    fitting_ModelParms(tc_transf_LSD, '_LSD')
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
