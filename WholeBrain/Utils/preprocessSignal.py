# ================================================================================================================
# Function to preprocess empirical signals for an optimization stage
# ================================================================================================================
import WholeBrain.Utils.decorators as decorators
import time


verbose = True
def processBOLDSignals(BOLDsignals, observablesToUse):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    NumSubjects = len(BOLDsignals)
    N = BOLDsignals[next(iter(BOLDsignals))].shape[0]  # get the first key to retrieve the value of N = number of areas

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    for ds in observablesToUse:  # Initialize data structs for each observable
        measureValues[ds] = observablesToUse[ds][0].init(NumSubjects, N)

    # Loop over subjects
    for pos, s in enumerate(BOLDsignals):
        if verbose: print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos+1, NumSubjects, s, BOLDsignals[s].shape[0], BOLDsignals[s].shape[1]), end='', flush=True)
        signal = BOLDsignals[s]  # LR_version_symm(tc[s])
        start_time = time.perf_counter()

        for ds in observablesToUse:  # Now, let's compute each measure and store the results
            measure = observablesToUse[ds][0]  # FC, swFCD, phFCD, ...
            applyFilters = observablesToUse[ds][1]  # whether we apply filters or not...
            procSignal = measure.from_fMRI(signal, applyFilters=applyFilters)
            measureValues[ds] = measure.accumulate(measureValues[ds], pos, procSignal)

        if verbose: print(" -> computed in {} seconds".format(time.perf_counter() - start_time))

    for ds in observablesToUse:  # finish computing each observable
        measure = observablesToUse[ds][0]  # FC, swFCD, phFCD, ...
        measureValues[ds] = measure.postprocess(measureValues[ds])

    return measureValues


# ============== a practical way to save recomputing necessary (but lengthy) results ==========
@decorators.loadOrCompute
def processEmpiricalSubjects(BOLDsignals, observablesToUse):
    return processBOLDSignals(BOLDsignals, observablesToUse)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
