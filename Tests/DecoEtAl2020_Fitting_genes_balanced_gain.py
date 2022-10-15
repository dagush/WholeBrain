# ==========================================================================
# ==========================================================================
#  Computes, as a pre-process, the optimization for the bias (B) and scalling (Z)
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN (slurm.sbatch_genes_balanced_gain.m)
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

import WholeBrain.Optimizers.ParmSeep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator
# --------------------------------------------------------------------------
#  End local setup...
# --------------------------------------------------------------------------


def Fitting():
    # %%%%%%%%%%%%%%% Set General Model Parameters
    we = 2.1  # Global Coupling parameter, found in the DecoEtAl2018_Prepro_* file...
    J_fileName = baseOutPath+"/J_Balance_we2.1.mat"  # "Data_Produced/SC90/J_test_we{}.mat"
    balancedG = BalanceFIC.Balance_J9(we, C, False, J_fileName)
    balancedG['J'] = balancedG['J'].flatten()
    balancedG['we'] = balancedG['we']
    neuronalModel.setParms(balancedG)

    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'GBC': (GBC, False)}  #   'phFCD': (phFCD, True)
    swFCD.windowSize = 80
    swFCD.windowStep = 18

    # J_fileNames = baseOutPath+"/J_Balance_we{}.mat"

    # step = 0.05
    # Alphas = np.arange(-0.6, 0+step, step)  # Range used in the original code for B
    # Betas = np.arange(0, 2+step, step)    # Range used in the original code for Z
    Alphas = np.arange(-0.6, 0+0.1, 0.1)  # reduced range for DEBUG only!!!
    Betas = np.arange(0, 2+0.2, 0.2)  # reduced range for DEBUG only!!!

    grid = np.meshgrid(Alphas, Betas)
    grid = np.round(grid[0],3), np.round(grid[1],3)

    gridParms = [{'alpha': a, 'beta': b} for a,b in np.nditer(grid)]

    # Model Simulations
    # ------------------------------------------
    # Now, optimize all alpha (B), beta (Z) values: determine optimal (B,Z) to work with
    print("\n\n###################################################################")
    print("# Fitting (B,Z)")
    print("###################################################################\n")
    fitting = optim1D.distanceForAll_Parms(tc_transf, grid, gridParms, NumSimSubjects=NumTrials,
                                           distanceSettings=distanceSettings,
                                           parmLabel='BZ',
                                           outFilePath=baseOutPath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    # ------------------------------------------
    # ------------------------------------------

    filePath = baseOutPath+'/DecoEtAl2020_fittingBZ.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'Alphas': Alphas,
                 'Betas': Betas,
                 'swFCDfitt': fitting['swFCD'],  # swFCDfitt,
                 'FCfitt': fitting['FC'],  # FCfitt,
                 'GBCfitt': fitting['GBC'],  # GBCfitt
                })
    print(f"DONE!!! (file: {filePath})")


if __name__ == '__main__':
    Fitting()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
