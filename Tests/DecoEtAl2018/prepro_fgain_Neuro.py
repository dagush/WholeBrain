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
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C., Logothetis,N.K. & Kringelbach,M.L.
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       (2018) Current Biology
#       https://www.cell.com/current-biology/fulltext/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================


# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
from setup import *
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# ==========================================================================
# ==========================================================================
# ==========================================================================
# IMPORTANT: This function was created to reproduce Deco et al.'s 2018 code for Figure 3A.
# Actually, this only performs the fitting which gives the value of we (we in the original
# code, G in the paper) to use for further computations (e.g., plotting Figure 3A).
# For the plotting, see the respective file (fig3A.py)
def prepro_G_Optim():
    # %%%%%%%%%%%%%%% Set General Model Parameters
    J_fileNames = outFilePath + "J_Balance_we{}.mat"

    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True)}

    wStart = 0.
    step = 0.1  # 0.025
    wEnd = 2.5 + step
    WEs = np.arange(wStart, wEnd, step)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)

    # Model Simulations
    # ------------------------------------------
    BalanceFIC.verbose = True
    balancedParms = BalanceFIC.Balance_AllJ9(C, WEs, baseName=J_fileNames)
    modelParms = [balancedParms[i] for i in balancedParms]

    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute G_Optim")
    print("###################################################################\n")
    fitting = optim1D.distanceForAll_Parms(tc_transf_PLA, WEs, modelParms, NumSimSubjects=NumSubjects,
                                           distanceSettings=distanceSettings,
                                           parmLabel='we',
                                           outFilePath=outFilePath, fileNameSuffix='')

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    print("Optimal:\n", optimal)

    filePath = outFilePath + 'DecoEtAl2018_fneuro.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'we': WEs,
                 'fitting_PLA': fitting['FC'],  # fitting_PLA,
                 'FCDfitt_PLA': fitting['swFCD'],  # FCDfitt_PLA
                })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');
    print(f"DONE!!! (file: {filePath})")

if __name__ == '__main__':
    prepro_G_Optim()

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
