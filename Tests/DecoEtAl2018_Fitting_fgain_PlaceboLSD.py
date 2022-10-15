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
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C., Logothetis,N.K. & Kringelbach,M.L.
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       (2018) Current Biology
#       https://www.cell.com/current-biology/pdfExtended/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================

# --------------------------------------------------------------------------
#  Begin local setup...
# --------------------------------------------------------------------------
from DecoEtAl2018_Setup import *
# ParmSeep.neuronalModel = serotonin2A  # Finish setup definition
# --------------------------------------------------------------------------
#  End local setup...
# --------------------------------------------------------------------------


# ==========================================================================
# ==========================================================================
# ==========================================================================
# IMPORTANT: This function was created to reproduce Deco et al.'s 2018 code for Figure 3B.
# Actually, this only performs the fitting which gives the value of S_E (wge in the original
# code) to use for further computations (e.g., plotting Figure 3B)
def fitting_ModelParms(tc_transf, suffix):
    # %%%%%%%%%%%%%%% Set General Model Parameters
    # neuronalModel.setParm({'we':2.1})  # Global Coupling parameter, found in the DecoEtAl2018_Prepro_* file...
    J_fileName = outFilePath + "/J_Balance_we2.1.mat"  # "Data_Produced/SC90/J_test_we{}.mat"
    balancedG = BalanceFIC.Balance_J9(2.1, C, False, J_fileName)
    balancedG['J'] = balancedG['J'].flatten()
    balancedG['we'] = balancedG['we']  #.flatten()
    serotonin2A.setParms(balancedG)

    # serotonin2A.wgaine = 0.
    # serotonin2A.wgaini = 0.
    serotonin2A.setParms({'S_E':0., 'S_I':0.})

    distanceSettings = {'swFCD': (swFCD, True)}  #'phFCD': (phFCD, True)}, 'FC': (FC, False)

    S_EStart = 0.
    S_EStep = 0.008  # 0.04  # 0.002
    S_EEnd = 0.4 + S_EStep
    S_Es = np.arange(S_EStart, S_EEnd, S_EStep)

    serotoninParms = [{'S_I': 0., 'S_E': S_E} for S_E in S_Es]  # here we leave the inhibitory component as 0.

    basePath = outFilePath

    fitting = optim1D.distanceForAll_Parms(tc_transf, S_Es, serotoninParms, NumSimSubjects=NumSubjects,
                                           distanceSettings=distanceSettings,
                                           parmLabel='S_E',
                                           outFilePath=basePath,
                                           fileNameSuffix=suffix)

    filePath = '../Data_Produced/DecoEtAl2018_fitting'+suffix+'.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'S_E': S_Es,
                 'fitting_FCD': fitting['swFCD'],
                })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');

    print("DONE!!!")


if __name__ == '__main__':
    # ======================== first, Placebo condition
    fitting_ModelParms(tc_transf_PLA, '_PLA')

    # ======================== Second, LSD condition
    fitting_ModelParms(tc_transf_LSD, '_LSD')
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
