# ================================================================================================================
# Function to preprocess empirical signals for an optimization stage
# ================================================================================================================
from WholeBrain.Utils.decorators import loadOrCompute
import time


verbose = True
def processBOLDSignals(BOLDsignals, distanceSettings):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # distanceSettings is a dictionary of {distanceMeasureName: distanceMeasurePythonModule}
    NumSubjects = len(BOLDsignals)
    N = BOLDsignals[next(iter(BOLDsignals))].shape[0]  # get the first key to retrieve the value of N = number of areas

    # First, let's create a data structure for the distance measurement operations...
    measureValues = {}
    for ds in distanceSettings:  # Initialize data structs for each distance measure
        measureValues[ds] = distanceSettings[ds][0].init(NumSubjects, N)

    # Loop over subjects
    for pos, s in enumerate(BOLDsignals):
        if verbose: print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos+1, NumSubjects, s, BOLDsignals[s].shape[0], BOLDsignals[s].shape[1]), end='', flush=True)
        signal = BOLDsignals[s]  # LR_version_symm(tc[s])
        start_time = time.perf_counter()

        for ds in distanceSettings:  # Now, let's compute each measure and store the results
            measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
            applyFilters = distanceSettings[ds][1]  # whether we apply filters or not...
            procSignal = measure.from_fMRI(signal, applyFilters=applyFilters)
            measureValues[ds] = measure.accumulate(measureValues[ds], pos, procSignal)

        if verbose: print(" -> computed in {} seconds".format(time.perf_counter() - start_time))

    for ds in distanceSettings:  # finish computing each distance measure
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        measureValues[ds] = measure.postprocess(measureValues[ds])

    return measureValues


# ============== a practical way to save recomputing necessary (but lengthy) results ==========
@loadOrCompute
def processEmpiricalSubjects(BOLDsignals, distanceSettings):
    return processBOLDSignals(BOLDsignals, distanceSettings)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
