# ==========================================================================
# ==========================================================================
#  Wrapper class to connect with the L2L optimization library
# ==========================================================================
# ==========================================================================
import numpy as np
import logging
from sdict import sdict
from l2l.optimizees.optimizee import Optimizee
import random

from functions.Utils.decorators import loadOrCompute


# ========================================================================
# This is the actual function to optimize... It is usually never called
# directly, but through the wrapper below...
# ========================================================================
# First, some necessary modules that are needed for the simulations
integrator = None  # The integration scheme
neuronalModel = None  # The neuronal model to integrate
simulateBOLD = None  # Whether we are going to use BOLD (SimAndBOLD) or not (SimOnly)
# ========================================================================
# Now, some running parameters
measure = None  # Measure to use to compute the error
applyFilters = None  # Whether to apply filters to the resulting signal or not
processedEmp = None  # reference values (e.g., empirical) to compare to.
N = None  # Number of regions in the parcellation
# SC = None  # Structural Connectivity Matrix
trials = None  # Number of trials to try
# ========================================================================
# ========================================================================


logger = logging.getLogger("l2l-WholeBrain")

def defaultSetupFunc(parms):
    neuronalModel.setParms(parms)


def parm2filename(parm):
    return '_'.join([key+str(np.round(value, 3)) for key, value in parm.items()])


def translateParms(parm):
    return {k.replace('individual.',''): v for k, v in parm.items()}


class WholeBrainOptimizee(Optimizee):

    def __init__(self, trajectory, varsAndBounds, setupFunc=defaultSetupFunc, outFilenamePattern=''):
        super(WholeBrainOptimizee, self).__init__(trajectory)
        self.varsAndBounds = varsAndBounds
        self.setupFunc = setupFunc
        # create_individual can be called because __init__ is complete except for traj initializtion
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            trajectory.individual.f_add_parameter(key, val)
        self.filenamePattern = outFilenamePattern

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        indiv = {}
        for k in self.varsAndBounds.keys():
            bounds = self.varsAndBounds[k]
            indiv[k] = random.uniform(bounds[0],
                                      bounds[1])
        return indiv

    # def bounding_func(self, individual):
    #     return individual

    # Performs one simulation and returns the results
    @loadOrCompute
    def simulate_(self):
        print("   Going to eval:", self.x, flush=True)
        self.setupFunc(self.x)  # Use either the defaultSetupFunc or the one provided by the user...
        integrator.recompileSignatures()
        measureValues = measure.init(trials, N)
        for i in range(trials):
            bds = simulateBOLD.simulateSingleSubject().T
            procSignal = measure.from_fMRI(bds, applyFilters=applyFilters)
            measureValues = measure.accumulate(measureValues, i, procSignal)

        # ====== Measure the results...
        # measure = distanceSetting[0]  # FC, swFCD, phFCD, ...
        # applyFilters = distanceSetting[1]  # whether we apply filters or not...
        # procSignal = measure.from_fMRI(bds, applyFilters=applyFilters)
        # ====== Now, return the residuals...
        # r = processedBOLDemp - procSignal  # this will give a vector of the length of a phCFD upper triangular matrix. Then they will be compared using the l^2 metric
        # r = measure.distance(processedBOLDemp, procSignal)  # this is a float with the KS distance between the two phFCD vectors...
        measureValues = measure.postprocess(measureValues)
        r = measure.distance(measureValues, processedEmp)
        result = self.x.copy()
        result[measure.name] = r
        return result  # For the @loadOrCompute wrapper to work, all functions should return dicts

    def simulate(self, trajectory):
        self.id = trajectory.individual.ind_idx
        self.x = translateParms(trajectory.individual.params)
        # Start simulation
        filename = self.filenamePattern.format(parm2filename(self.x))
        fitness = self.simulate_(filename)[measure.name]  # For the @loadOrCompute wrapper to work, all functions should return dicts
        print("  Value:", fitness, "@", self.x, flush=True)
        # Return the last correlation coefficient as fitness of the model
        return fitness

    def end(self):
        logger.info("End of all experiments. Cleaning up...")
        # There's nothing to clean up though


# ==========================================================================
# ==========================================================================
# ==========================================================================
if __name__ == "__main__":
    def main():
        from l2l.utils.experiment import Experiment
        experiment = Experiment(root_dir_path='../../Data_Produced/L2L')
        name = 'L2L-TEST-WholeBrain'
        traj, _ = experiment.prepare_experiment(name=name, log_stdout=True, multiprocessing=False)

        optimizee = WholeBrainOptimizee(traj, {'we':(0,10)})
        traj.individual = sdict(optimizee.create_individual())

        # testing_error = optimizee.simulate(traj)
        # print("Testing error is %s", testing_error)

    main()

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
