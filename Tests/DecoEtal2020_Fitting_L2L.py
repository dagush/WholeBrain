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
from collections import namedtuple
from l2l.utils.experiment import Experiment
from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters
import WholeBrain.Optimizers.L2LOptimizee as WBOptimizee

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
#  Begin local setup...
# --------------------------------------------------------------------------
from DecoEtAl2020_Setup import *

import WholeBrain.Optimizers.ParmSeep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator

from WholeBrain.Optimizers.preprocessSignal import processEmpiricalSubjects  # processBOLDSignals
# --------------------------------------------------------------------------
#  End local setup...
# --------------------------------------------------------------------------
baseOutPath = 'Data_Produced/DecoEtAl2020'
J_fileNames = baseOutPath+"/J_Balance_we{}.mat"


def setupFunc(x):
    we = x['we']


def Fitting():
    baseOutPath = 'Data_Produced/DecoEtAl2020'

    # %%%%%%%%%%%%%%% Set General Model Parameters
    we = 2.1  # Global Coupling parameter, found in the DecoEtAl2018_Prepro_* file...
    J_fileName = baseOutPath+"/J_Balance_we2.1.mat"  # "Data_Produced/SC90/J_test_we{}.mat"
    balancedG = BalanceFIC.Balance_J9(we, C, False, J_fileName)
    balancedG['J'] = balancedG['J'].flatten()
    balancedG['we'] = balancedG['we']
    neuronalModel.setParms(balancedG)

    # distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'GBC': (GBC, False)}  #   'phFCD': (phFCD, True)
    distanceSettings = {'swFCD': (swFCD, True)}
    swFCD.windowSize = 80
    swFCD.windowStep = 18

    # J_fileNames = baseOutPath+"/J_Balance_we{}.mat"

    # step = 0.05
    # Alphas = np.arange(-0.6, 0+step, step)  # Range used in the original code for B
    # Betas = np.arange(0, 2+step, step)    # Range used in the original code for Z
    Alphas = np.arange(-0.6, 0+0.1, 0.1)  # reduced range for DEBUG only!!!
    Betas = np.arange(0, 2+0.2, 0.2)  # reduced range for DEBUG only!!!

    # grid = np.meshgrid(Alphas, Betas)
    # grid = np.round(grid[0],3), np.round(grid[1],3)
    # gridParms = [{'alpha': a, 'beta': b} for a,b in np.nditer(grid)]

    # Model Simulations
    # ------------------------------------------
    # Now, optimize all alpha (B), beta (Z) values: determine optimal (B,Z) to work with
    print("\n\n###################################################################")
    print("# Fitting (B,Z)")
    print("###################################################################\n")
    experiment = Experiment(root_dir_path='../Data_Produced/L2L')
    name = 'L2L-DecoEtAl2020-Prepro'
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True, multiprocessing=False)

    # Setup the WhileBrain optimizee
    WBOptimizee.neuronalModel = neuronalModel
    WBOptimizee.integrator = integrator
    WBOptimizee.simulateBOLD = simulateBOLD

    selectedObservable = 'swFCD'
    distanceSettings = {selectedObservable: (swFCD, True)}  # We need to overwrite this, as L2L only works with ONE observable at a time.
    WBOptimizee.measure = distanceSettings[selectedObservable][0]  # Measure to use to compute the error
    WBOptimizee.applyFilters = distanceSettings[selectedObservable][1]  # Whether to apply filters to the resulting signal or not
    outEmpFileName = baseOutPath + '/fNeuro_emp_L2L.mat'
    WBOptimizee.processedEmp = processEmpiricalSubjects(tc_transf,
                                                        distanceSettings,
                                                        outEmpFileName)[selectedObservable]  # reference values (e.g., empirical) to compare to.
    WBOptimizee.N = N  # Number of regions in the parcellation
    WBOptimizee.trials = NumTrials  # Number of trials to try
    optimizee_parameters = namedtuple('OptimizeeParameters', [])

    filePattern = baseOutPath + '/fitting_{}_L2L.mat'
    optimizee = WBOptimizee.WholeBrainOptimizee(traj, {'alpha': (-0.6, 0), 'beta': (0., 2.)}, outFilenamePattern=filePattern)  #setupFunc=setupFunc,

    # =================== Test for debug only
    # traj.individual = sdict(optimizee.create_individual())
    # testing_error = optimizee.simulate(traj)
    # print("Testing error is %s", testing_error)
    # =================== end Test

    # Setup the GridSearchOptimizer
    optimizer_parameters = GridSearchParameters(param_grid={
        'alpha': (-0.6, 0., 6),
        'beta': (0., 2., 10)
    })
    optimizer = GridSearchOptimizer(traj,
                                    optimizee_create_individual=optimizee.create_individual,
                                    optimizee_fitness_weights=(-1.,),  # minimize!
                                    parameters=optimizer_parameters)

    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)
    experiment.end_experiment(optimizer)
    print(f"best: alpha={experiment.optimizer.best_individual['alpha']} & beta={experiment.optimizer.best_individual['beta']}")

    # fitting = optim1D.distanceForAll_Parms(tc_transf, grid, gridParms, NumSimSubjects=NumTrials,
    #                                        distanceSettings=distanceSettings,
    #                                        parmLabel='BZ',
    #                                        outFilePath=baseOutPath)
    #
    # optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    # ------------------------------------------
    # ------------------------------------------

    filePath = baseOutPath+'/DecoEtAl2020_fittingBZ.mat'
    # sio.savemat(filePath, #{'JI': JI})
    #             {'Alphas': Alphas,
    #              'Betas': Betas,
    #              'swFCDfitt': fitting['swFCD'],  # swFCDfitt,
    #              'FCfitt': fitting['FC'],  # FCfitt,
    #              'GBCfitt': fitting['GBC'],  # GBCfitt
    #             })
    print(f"DONE!!! (file: {filePath})")


if __name__ == '__main__':
    Fitting()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
