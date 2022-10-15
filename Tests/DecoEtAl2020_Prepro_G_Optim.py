# ==========================================================================
# ==========================================================================
#  Computes, as a pre-process, the optimization for G
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN (slurm.sbatch_genes_balanced_G_optimization.m)
#
#  Taken from the code (slurm.sbatch_genes_balanced_G_optimization.m) from:
#  [DecoEtAl_2021] Gustavo Deco, Kevin Aquino, Aurina Arnatkeviciute, Stuart Oldham, Kristina Sabaroedin,
#  Nigel Rogasch, Morten L. Kringelbach, and Alex Fornito, "Dynamical consequences of regional heterogeneity
#  in the brain’s transcriptional landscape", 2021, biorXiv
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================

# --------------------------------------------------------------------------
#  Begin local setup...
# --------------------------------------------------------------------------
from DecoEtAl2020_Setup import *

import WholeBrain.Optimizers.ParmSeep as parmSweep
parmSweep.simulateBOLD = simulateBOLD
parmSweep.integrator = integrator
# --------------------------------------------------------------------------
#  End local setup...
# --------------------------------------------------------------------------


def prepro():
    # Make the neuronal model to work as the DMF model
    neuronalModel.alpha = 0.
    neuronalModel.beta = 0.

    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'GBC': (GBC, False)}  #   'phFCD': (phFCD, True)
    swFCD.windowSize = 80
    swFCD.windowStep = 18

    J_fileNames = baseOutPath+"/J_Balance_we{}.mat"
    # baseGOptimNames = baseOutPath+"/fitting_we{}.mat"

    step = 0.001
    # WEs = np.arange(0, 3.+step, step)  # Range used in the original code
    WEs = np.arange(0, 3.+step, 0.05)  # reduced range for DEBUG only!!!

    # Model Simulations
    # ------------------------------------------
    BalanceFIC.verbose = True
    balancedParms = BalanceFIC.Balance_AllJ9(C, WEs, baseName=J_fileNames)
    modelParms = [balancedParms[i] for i in balancedParms]

    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute G_Optim")
    print("###################################################################\n")
    fitting = parmSweep.distanceForAll_Parms(tc_transf, WEs, modelParms, NumSimSubjects=NumTrials,
                                             distanceSettings=distanceSettings,
                                             parmLabel='we',
                                             outFilePath=baseOutPath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    # ------------------------------------------
    # ------------------------------------------

    filePath = baseOutPath+'/DecoEtAl2020_fneuro.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'we': WEs,
                 'swFCDfitt': fitting['swFCD'],  # swFCDfitt,
                 'FCfitt': fitting['FC'],  # FCfitt,
                 'GBCfitt': fitting['GBC'],  # GBCfitt
                })
    print(f"DONE!!! (file: {filePath})")


if __name__ == '__main__':
    prepro()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
