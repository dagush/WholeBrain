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
# import logging
# from sdict import sdict
from collections import namedtuple
from l2l.utils.experiment import Experiment
from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters
import WholeBrain.Optimizers.L2LOptimizee as WBOptimizee

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
#  Begin local setup...
# --------------------------------------------------------------------------
from DecoEtAl2020_Setup import *

import WholeBrain.Optimizers.ParmSeep as parmSweep
parmSweep.simulateBOLD = simulateBOLD
parmSweep.integrator = integrator

from WholeBrain.Optimizers.preprocessSignal import processEmpiricalSubjects  # processBOLDSignals
# --------------------------------------------------------------------------
#  End local setup...
# --------------------------------------------------------------------------

J_fileNames = baseOutPath+"/J_Balance_we{}.mat"

def plotTrajectory1D(x, values):
    plt.rcParams.update({'font.size': 15})
    plotFCD, = plt.plot(x, values)
    plotFCD.set_label("swFCD")
    plt.title("Whole-brain fitting")
    plt.ylabel("Functional Fitting")
    plt.xlabel("Global Coupling (G = we)")
    plt.legend()
    plt.show()

def setupFunc(x):
    we = x['we']
    J_file = J_fileNames.format(np.round(we, decimals=3))
    balanced = BalanceFIC.Balance_J9(we, N, J_file)['J'].flatten()
    neuronalModel.setParms({'J': balanced})


def prepro():
    # Make the neuronal model to work as the DMF model
    # neuronalModel.alpha = 0.
    # neuronalModel.beta = 0.

    # distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'GBC': (GBC, False)}  #   'phFCD': (phFCD, True)
    distanceSettings = {'swFCD': (swFCD, True)}
    swFCD.windowSize = 80
    swFCD.windowStep = 18

    # baseGOptimNames = baseOutPath+"/fitting_we{}.mat"

    # step = 0.001
    # WEs = np.arange(0, 3.+step, step)  # Range used in the original code
    # WEs = np.arange(0, 3.+step, 0.05)  # reduced range for DEBUG only!!!

    # Model Simulations
    # ------------------------------------------
    BalanceFIC.verbose = True
    # balancedParms = BalanceFIC.Balance_AllJ9(C, WEs, baseName=J_fileNames)
    # modelParms = [balancedParms[i] for i in balancedParms]

    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute optimization with L2L")
    print("###################################################################\n")
    experiment = Experiment(root_dir_path='../Data_Produced/L2L')
    name = 'L2L-DecoEtAl2020-Prepro'
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True, multiprocessing=False)

    # Setup the WhileBrain optimizee
    WBOptimizee.neuronalModel = neuronalModel
    WBOptimizee.integrator = integrator
    WBOptimizee.simulateBOLD = simulateBOLD
    WBOptimizee.measure = distanceSettings['swFCD'][0]  # Measure to use to compute the error
    WBOptimizee.applyFilters = distanceSettings['swFCD'][1]  # Whether to apply filters to the resulting signal or not
    outEmpFileName = baseOutPath + '/fNeuro_emp_L2L.mat'
    WBOptimizee.processedEmp = processEmpiricalSubjects(tc_transf,
                                                        distanceSettings,
                                                        outEmpFileName)['swFCD']  # reference values (e.g., empirical) to compare to.
    WBOptimizee.N = N  # Number of regions in the parcellation
    WBOptimizee.trials = NumTrials  # Number of trials to try
    optimizee_parameters = namedtuple('OptimizeeParameters', [])

    filePattern = baseOutPath + '/fitting_{}_L2L.mat'
    optimizee = WBOptimizee.WholeBrainOptimizee(traj, {'we': (0., 3.)}, setupFunc=setupFunc, outFilenamePattern=filePattern)

    # =================== Test for debug only
    # traj.individual = sdict(optimizee.create_individual())
    # testing_error = optimizee.simulate(traj)
    # print("Testing error is %s", testing_error)
    # =================== end Test

    # Setup the GridSearchOptimizer
    n_grid_divs_per_axis = 60  # 0.05
    optimizer_parameters = GridSearchParameters(param_grid={
        'we': (0., 3., n_grid_divs_per_axis)
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
    print(f"best: {experiment.optimizer.best_individual['we']}")
    # fitting = parmSweep.distanceForAll_Parms(tc_transf, WEs, modelParms, NumSimSubjects=NumTrials,
    #                                          distanceSettings=distanceSettings,
    #                                          parmLabel='we',
    #                                          outFilePath=baseOutPath)

    # optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    # ------------------------------------------
    # ------------------------------------------

    filePath = baseOutPath+'/DecoEtAl2020_fneuro-L2L.mat'
    # sio.savemat(filePath, #{'JI': JI})
    #             {'we': WEs,
    #              'swFCDfitt': fitting['swFCD'],  # swFCDfitt,
    #              'FCfitt': fitting['FC'],  # FCfitt,
    #              'GBCfitt': fitting['GBC'],  # GBCfitt
    #             })
    # print(f"DONE!!! (file: {filePath})")
    plotTrajectory1D(optimizer.param_list['we'], [v for (i,v) in traj.current_results])
    print("DONE!!!")


if __name__ == '__main__':
    prepro()
# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
