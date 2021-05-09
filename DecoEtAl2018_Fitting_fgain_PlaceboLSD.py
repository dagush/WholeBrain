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
# import numpy as np
# import scipy.io as sio
# from pathlib import Path
# from numba import jit

# --------------------------------------------------------------------------
#  Begin local setup...
# --------------------------------------------------------------------------
from DecoEtAl2018_Setup import *
optim1D.neuronalModel = serotonin2A  # Finish setup definition

# # Setup for Serotonin 2A-based DMF simulation!!!
# import functions.Models.DynamicMeanField as neuronalModel
# import functions.Models.serotonin2A as serotonin2A
# neuronalModel.He = serotonin2A.phie
# neuronalModel.Hi = serotonin2A.phii
# # ----------------------------------------------
# import functions.Integrator_EulerMaruyama as integrator
# integrator.neuronalModel = neuronalModel
# integrator.verbose = False
# import functions.BOLDHemModel_Stephan2007 as Stephan2007
# import functions.simulate_SimAndBOLD as simulateBOLD
# simulateBOLD.integrator = integrator
# simulateBOLD.BOLDModel = Stephan2007
# import functions.simulateFCD as simulateFCD
# simulateFCD.integrator = integrator
# simulateFCD.BOLDModel = Stephan2007
#
# # import functions.Observables.FC as FC
# import functions.Observables.swFCD as swFCD
#
# import functions.BalanceFIC as BalanceFIC
# BalanceFIC.integrator = integrator
#
# import functions.Optimizers.Optim1D as optim1D
# optim1D.simulateBOLD = simulateBOLD
# optim1D.integrator = integrator
# optim1D.neuronalModel = serotonin2A
#
# # set BOLD filter settings
# import functions.BOLDFilters as filters
# filters.k = 2                             # 2nd order butterworth filter
# filters.flp = .01                         # lowpass frequency of filter
# filters.fhi = .1                          # highpass
#
# PLACEBO_cond = 4; LSD_cond = 1   # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...
# --------------------------------------------------------------------------
#  End local setup...
# --------------------------------------------------------------------------


# @jit(nopython=True)
# def initRandom():
#     np.random.seed(3)  # originally set to 13
#
#
# def recompileSignatures():
#     # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
#     # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
#     print("\n\nRecompiling signatures!!!")
#     serotonin2A.recompileSignatures()
#     integrator.recompileSignatures()
#
#
# def LR_version_symm(TC):
#     # returns a symmetrical LR version of AAL 90x90 matrix
#     odd = np.arange(0,90,2)
#     even = np.arange(1,90,2)[::-1]  # sort 'descend'
#     symLR = np.zeros((90,TC.shape[1]))
#     symLR[0:45,:] = TC[odd,:]
#     symLR[45:90,:] = TC[even,:]
#     return symLR
#
#
# def transformEmpiricalSubjects(tc_aal, cond, NumSubjects):
#     transformed = {}
#     for s in range(NumSubjects):
#         # transformed[s] = np.zeros(tc_aal[0,cond].shape)
#         transformed[s] = LR_version_symm(tc_aal[s,cond])
#     return transformed


# ==========================================================================
# ==========================================================================
# ==========================================================================
# IMPORTANT: This function was created to reproduce Deco et al.'s 2018 code for Figure 3B.
# Actually, this only performs the fitting which gives the value of S_E (wge in the original
# code) to use for further computations (e.g., plotting Figure 3B)
def fitting_ModelParms(C, tc_transf, suffix):
    # %%%%%%%%%%%%%%% Set General Model Parameters
    neuronalModel.we = 2.1  # Global Coupling parameter, found in the DecoEtAl2018_Prepro_* file...
    J_fileName = "Data_Produced/SC90/J_Balance_we2.1.mat"  # "Data_Produced/SC90/J_test_we{}.mat"
    balancedG = BalanceFIC.Balance_J9(neuronalModel.we, C, False, J_fileName)
    balancedG['J'] = balancedG['J'].flatten()
    balancedG['we'] = balancedG['we'].flatten()
    neuronalModel.setParms(balancedG)

    serotonin2A.wgaine = 0.
    serotonin2A.wgaini = 0.

    distanceSettings = {'swFCD': (swFCD, True)}  #'phFCD': (phFCD, True)}, 'FC': (FC, False)

    S_EStart = 0.
    S_EStep = 0.008  # 0.04  # 0.002
    S_EEnd = 0.4 + S_EStep
    S_Es = np.arange(S_EStart, S_EEnd, S_EStep)

    serotoninParms = {S_E: {'S_I': 0., 'S_E':S_E} for S_E in S_Es}  # here we leave the inhibitory component as 0.

    basePath = "Data_Produced/SC90"  # "/fitting_S_E{}.mat"

    fitting = optim1D.distanceForAll_Parms(C, tc_transf, serotoninParms, NumSimSubjects=NumSubjects,
                                           distanceSettings=distanceSettings,
                                           Parms=S_Es,
                                           parmLabel='S_E',
                                           outFilePath=basePath,
                                           fileNameSuffix=suffix)

    filePath = 'Data_Produced/DecoEtAl2018_fitting'+suffix+'.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'S_E': S_Es,
                 # 'fitting_LSD': fitting['fitting_LSD,
                 'fitting_FCD': fitting['swFCD'],
                 # 'FCDfitt_LSD': FCDfitt_LSD,
                 # 'FCDfitt'+suffix: FCDfitt_PLA
                })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');

    print("DONE!!!")


if __name__ == '__main__':
    # initRandom()
    #
    # # Load Structural Connectivity Matrix
    # print("Loading Data_Raw/all_SC_FC_TC_76_90_116.mat")
    # sc90 = sio.loadmat('Data_Raw/all_SC_FC_TC_76_90_116.mat')['sc90']
    # C = sc90/np.max(sc90[:])*0.2  # Normalization...
    #
    # # Load Regional Drug Receptor Map
    # print('Loading Data_Raw/mean5HT2A_bindingaal.mat')
    # mean5HT2A_aalsymm = sio.loadmat('Data_Raw/mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
    # serotonin2A.Receptor = (mean5HT2A_aalsymm[:,0]/np.max(mean5HT2A_aalsymm[:,0])).flatten()
    # recompileSignatures()
    #
    # #load fMRI data
    # print("Loading Data_Raw/LSDnew.mat")
    # LSDnew = sio.loadmat('Data_Raw/LSDnew.mat')  #load LSDnew.mat tc_aal
    # tc_aal = LSDnew['tc_aal']
    # (N, Tmax) = tc_aal[1,1].shape  # [N, Tmax]=size(tc_aal{1,1}) # N = number of areas; Tmax = total time
    # print(f'tc_aal is {tc_aal.shape} and each entry has N={N} regions and Tmax={Tmax}')
    #
    # NumSubjects = 15  # Number of Subjects in empirical fMRI dataset, originally 20...
    # print(f"Simulating {NumSubjects} subjects!")
    #
    # tc_transf_PLA = transformEmpiricalSubjects(tc_aal, PLACEBO_cond, NumSubjects)  # PLACEBO
    # tc_transf_LSD = transformEmpiricalSubjects(tc_aal, LSD_cond, NumSubjects)  # LSD

    # ======================== first, Placebo condition
    fitting_ModelParms(C, tc_transf_PLA, '_PLA')

    # ======================== Second, LSD condition
    fitting_ModelParms(C, tc_transf_LSD, '_LSD')
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
