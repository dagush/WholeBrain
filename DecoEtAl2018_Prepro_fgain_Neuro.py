# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Taken from the code (fgain_Neuro.m) from:
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
from numba import jit

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import functions.Models.DynamicMeanField as neuronalModel
# import functions.Models.serotonin2A as serotonin2A
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

import functions.Optimizers.Optim1D as G_optim
G_optim.simulateBOLD = simulateBOLD
G_optim.integrator = integrator
G_optim.neuronalModel = neuronalModel

# set BOLD filter settings
import functions.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .01                         # lowpass frequency of filter
filters.fhi = .1                          # highpass

PLACEBO_cond = 4; LSD_cond = 1   # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


@jit(nopython=True)
def initRandom():
    np.random.seed(3)  # originally set to 13


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


# def recompileSignatures():
#     # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
#     # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
#     print("\n\nRecompiling signatures!!!")
#     serotonin2A.recompileSignatures()
#     integrator.recompileSignatures()


# ==========================================================================
# ==========================================================================
# ==========================================================================
# IMPORTANT: This function was created to reproduce Deco et al.'s 2018 code for Figure 3A.
# Actually, this only performs the fitting which gives the value of we (we in the original
# code, G in the paper) to use for further computations (e.g., plotting Figure 3A)
def prepro_G_Optim():
    # Load Structural Connectivity Matrix
    print("Loading Data_Raw/all_SC_FC_TC_76_90_116.mat")
    sc90 = sio.loadmat('Data_Raw/all_SC_FC_TC_76_90_116.mat')['sc90']
    C = sc90/np.max(sc90[:])*0.2  # Normalization...

    # # Load Regional Drug Receptor Map
    # print('Loading Data_Raw/mean5HT2A_bindingaal.mat')
    # mean5HT2A_aalsymm = sio.loadmat('Data_Raw/mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
    # serotonin2A.Receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()
    # recompileSignatures()

    #load fMRI data
    print("Loading Data_Raw/LSDnew.mat")
    LSDnew = sio.loadmat('Data_Raw/LSDnew.mat')  #load LSDnew.mat tc_aal
    tc_aal = LSDnew['tc_aal']
    (N, Tmax) = tc_aal[1,1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time
    print('tc_aal is {} and each entry has N={} regions and Tmax={}'.format(tc_aal.shape, N, Tmax))

    NumSubjects = 15  # Number of Subjects in empirical fMRI dataset, 20 in the original code...
    print("Simulating {} subjects!".format(NumSubjects))

    tc_transf_PLA = transformEmpiricalSubjects(tc_aal, PLACEBO_cond, NumSubjects)  # PLACEBO
    # FCemp_cotsampling_PLA = G_optim.processEmpiricalSubjects(tc_transf_PLA, distanceSettings, "Data_Produced/SC90/fNeuro_emp_PLA.mat")
    # FCemp_PLA = FCemp_cotsampling_PLA['FC']; cotsampling_PLA = FCemp_cotsampling_PLA['swFCD'].flatten()

    # tc_transf_LSD = transformEmpiricalSubjects(tc_aal, LSD_cond, NumSubjects)  # LSD
    # FCemp_cotsampling_LSD = G_optim.processEmpiricalSubjects(tc_transf_LSD, distanceSettings, "Data_Produced/SC90/fNeuro_emp_LCD.mat")  # LCD
    # FCemp_LSD = FCemp_cotsampling_LSD['FC']; cotsampling_LSD = FCemp_cotsampling_LSD['swFCD'].flatten()

    # ==============================================================
    # ==============================================================

    # %%%%%%%%%%%%%%% Set General Model Parameters
    # dtt   = 1e-3   # Sampling rate of simulated neuronal activity (seconds)
    # dt    = 0.1
    # DMF.J     = np.ones(N,1)
    # Tmaxneuronal = (Tmax+10)*2000;
    J_fileNames = "Data_Produced/SC90/J_Balance_we{}.mat"  # "Data_Produced/SC90/J_test_we{}.mat"
    baseName = "Data_Produced/SC90/fitting_we{}.mat"

    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True)}

    wStart = 0.
    step = 0.1  # 0.025
    wEnd = 2.5 + step
    WEs = np.arange(wStart, wEnd, step)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)
    # numWEs = len(WEs)

    # FCDfitt_PLA = np.zeros((numWEs))
    # FCDfitt_LSD = np.zeros((numWEs))
    # fitting_PLA = np.zeros((numWEs))
    # fitting_LSD = np.zeros((numWEs))

    # Model Simulations
    # ------------------------------------------
    BalanceFIC.verbose = True
    balancedParms = BalanceFIC.Balance_AllJ9(C, WEs, baseName=J_fileNames)

    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute G_Optim")
    print("###################################################################\n")
    outFilePath = 'Data_Produced/SC90'
    fitting = G_optim.distanceForAll_Parms(C, tc_transf_PLA, balancedParms, NumSimSubjects=NumSubjects,
                                           distanceSettings=distanceSettings,
                                           Parms=WEs,
                                           parmLabel='we',
                                           outFilePath=outFilePath, fileNameSuffix='_PLA')

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    print("Optimal:\n", optimal)

    filePath = 'Data_Produced/DecoEtAl2018_fneuro.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'we': WEs,
                 # 'fitting_LSD': fitting_LSD,
                 'fitting_PLA': fitting['FC'],  # fitting_PLA,
                 # 'FCDfitt_LSD': FCDfitt_LSD,
                 'FCDfitt_PLA': fitting['swFCD'],  # FCDfitt_PLA
                })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');
    print(f"DONE!!! (file: {filePath})")

if __name__ == '__main__':
    prepro_G_Optim()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
