# --------------------------------------------------------------------------------------
# Multi-measurements: a practical way to perform multiple measurements at once without
# having to worry about the details of each one
# --------------------------------------------------------------------------------------
import numpy as np
from scipy import signal, stats
from WholeBrain.Utils import demean

print("Going to use multi-measurement metrics...")

name = 'multi'


distanceSettings = None
# preferredDistance = None

ERROR_VALUE = -10000


def distance(measure1, measure2):
    distances = {}
    for ds in distanceSettings:  # Now, let's compute each measure and store the results
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        if not (np.isnan(measure1[ds]).any() or np.isnan(measure2[ds]).any()):  # No problems, go ahead!!!
            distances[ds] = measure.distance(measure1[ds], measure2[ds])
        else:
            distances[ds] = ERROR_VALUE
    return distances


def from_fMRI(BOLDSignal, applyFilters=True, removeStrongArtefacts=True):
    procSignal = {}
    for ds in distanceSettings:  # Now, let's compute each measure and store the results
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        applyFilters = distanceSettings[ds][1]  # whether we apply filters or not...
        procSignal[ds] = measure.from_fMRI(BOLDSignal, applyFilters=applyFilters, removeStrongArtefacts=removeStrongArtefacts)
        # measureValues[ds] = measure.accumulate(measureValues[ds], pos, procSignal)
    return procSignal


# ==================================================================
# Simple generalization WholeBrain to abstract distance measures
# ==================================================================
def init(S, N):
    measureValues = {}
    for ds in distanceSettings:  # Initialize data structs for each distance measure
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        measureValues[ds] = measure.init(S, N)
    return measureValues


def accumulate(measureValues, nsub, procSignal):
    for ds in distanceSettings:  # Now, let's compute each measure and store the results
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        measureValues[ds] = measure.accumulate(measureValues[ds], nsub, procSignal[ds])
    return measureValues


def postprocess(measureValues):
    for ds in distanceSettings:  # finish computing each distance measure
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        measureValues[ds] = measure.postprocess(measureValues[ds])
    return measureValues


def findMinMax(arrayValues):
    minMaxValues = {}
    for ds in distanceSettings:  # finish computing each distance measure
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        minMaxValues[ds] = measure.findMinMax(arrayValues[ds])
    return minMaxValues

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
