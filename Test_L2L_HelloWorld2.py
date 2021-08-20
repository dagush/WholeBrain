# =======================================================================
# =======================================================================
#  Test for L2L, copyied and pasted from l2l-fun-ga.py
# =======================================================================
# =======================================================================

# import os
# import yaml
import numpy as np

from l2l.optimizees.functions import tools as function_tools
from l2l.optimizees.functions.benchmarked_functions import BenchmarkedFunctions
from l2l.optimizees.functions.optimizee import FunctionGeneratorOptimizee
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters

from l2l.utils.experiment import Experiment

def main():
    experiment = Experiment(root_dir_path='Data_Produced/L2L')
    # name = 'L2L-FUN-GA'
    name = 'L2L-FUN-GS'
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True, multiprocessing=False)

    # ---------------------------------------------------------------------------------------------------------
    # Benchmark function
    """
    Ackley function has a large hole in at the centre surrounded by small hill like regions. Algorithms can get
    trapped in one of its many local minima.
    reference: https://www.sfu.ca/~ssurjano/ackley.html
    :param dims: dimensionality of the function
    Note: uses the recommended variable values, which are: a = 20, b = 0.2 and c = 2π.
    """
    function_id = 4  # Select Ackley2d
    bench_functs = BenchmarkedFunctions()
    (benchmark_name, benchmark_function), benchmark_parameters = \
        bench_functs.get_function_by_index(function_id, noise=True)
    # ---------------------------------------------------------------------------------------------------------

    optimizee_seed = 100
    random_state = np.random.RandomState(seed=optimizee_seed)
    # function_tools.plot(benchmark_function, random_state)

    ## Innerloop simulator
    optimizee = FunctionGeneratorOptimizee(traj, benchmark_function, seed=optimizee_seed)

    ## Outerloop optimizer initialization
    # parameters = GeneticAlgorithmParameters(seed=0, pop_size=50, cx_prob=0.5,
    #                                         mut_prob=0.3, n_iteration=100,
    #                                         ind_prob=0.02,
    #                                         tourn_size=15, mate_par=0.5,
    #                                         mut_par=1
    #                                         )
    #
    # optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
    #                                       optimizee_fitness_weights=(-0.1,),
    #                                       parameters=parameters)

    # Setup the GridSearchOptimizer
    n_grid_divs_per_axis = 30
    parameters = GridSearchParameters(param_grid={
        'coords': (optimizee.bound[0], optimizee.bound[1], n_grid_divs_per_axis)
    })
    optimizer = GridSearchOptimizer(traj,
                                    optimizee_create_individual=optimizee.create_individual,
                                    optimizee_fitness_weights=(-0.1,),  # minimize!
                                    parameters=parameters)

    ## Optimization!!!
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizee_parameters=parameters)
    experiment.end_experiment(optimizer)
    print(f"best: {experiment.optimizer.best_individual['coords']}")


if __name__ == '__main__':
    main()
