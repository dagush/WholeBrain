# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Execute before anything else!
#
#  Taken from the code from:
#  [NaskarEtAl_2018] Amit Naskar, Anirudh Vattikonda, Gustavo Deco,
#      Dipanjan Roy, Arpan Banerjee; Multiscale dynamic mean field (MDMF)
#      model relates resting-state brain dynamics with local cortical
#      excitatory–inhibitory neurotransmitter homeostasis.
#      Network Neuroscience 2021; 5 (3): 757–782. doi: https://doi.org/10.1162/netn_a_00197
#
#  Translated to Python by Gustavo Patow
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
# For the plotting, see the respective file (plotPrepro.py)
def prepro_G_Optim():
    # %%%%%%%%%%%%%%% Set General Model Parameters
    distanceSettings = {'FC': (FC, False),  # FC, do NOT apply filters
                        # 'swFCD': (swFCD, True)  # swFCD, apply filters
                        }

    wStart = 0.
    step = 0.01  # 0.01
    wEnd = 2. + step
    Gs = np.arange(wStart, wEnd, step)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)

    # Model Simulations
    # ------------------------------------------
    # BalanceFIC.verbose = True
    # balancedParms = BalanceFIC.Balance_AllJ9(C, WEs, baseName=J_fileNames)
    # modelParms = [balancedParms[i] for i in balancedParms]

    # Now, optimize all G values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute G_Optim")
    print("###################################################################\n")
    modelParms = [{'G': we} for we in Gs]
    fitting = optim1D.distanceForAll_Parms(fc_all, Gs, modelParms, NumSimSubjects=NumSubjects,
                                           observablesToUse=distanceSettings,
                                           doPreprocessing=False,
                                           parmLabel='G',
                                           outFilePath=outFilePath, fileNameSuffix='')

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    print("Optimal:\n", optimal)

    filePath = outFilePath + 'NaskarEtAl2021_fneuro.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'G': Gs,
                 'fitting': fitting['FC'],  # fitting_PLA,
                 # 'FCDfitt_PLA': fitting['swFCD'],  # FCDfitt_PLA
                })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');
    print(f"DONE!!! (file: {filePath})")

if __name__ == '__main__':
    prepro_G_Optim()

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
