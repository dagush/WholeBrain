# --------------------------------------------------------------------------------------
# Functions for AD subject processing, to study the effect of Amyloid-Beta and/or
# Tau over neuronal activity.
#
# Described at:
# Whole-brain modeling of the differential influences of Amyloid-Beta and Tau in Alzheimer`s Disease, by
# Gustavo Patow, Leon Stefanovski, Petra Ritter, Gustavo Deco, Xenia Kobeleva and for the
# Alzheimer’s Disease Neuroimaging Initiative. Accepted at Alzheimer's Research & Therapy, 2023
#
# --------------------------------------------------------------------------------------
import numpy as np
from WholeBrain.Utils.decorators import vectorCache  # loadOrCompute, loadSingleCache, loadMultipleCache
neuronalModel = None
integrator = None
simulateBOLD = None

# =====================================================================
# =====================================================================
#              Definitions for Single Subject Pipeline
# =====================================================================
# =====================================================================
parmLabels = [r'$b^E_{A\beta}$', r'$s^E_{A\beta}$',
              r'$b^E_\tau$', r'$s^E_\tau$',
              r'$b^I_{A\beta}$', r'$s^I_{A\beta}$',
              # no Tau inhibitory
              ]

parmBounds = [(-1.0, 1.0), (0.0, 4.0),  # Abeta increases activity of excitatory neurons > 0
              (-1.0, 1.0), (-4.0, 0.0),  # Tau silences < 0
              (-1.0, 1.0), (-4.0, 0.0)  # Abeta reduces inhibition < 0
              # Tau has no effect on inhibitory neurons ~ 0
              ]


# # ========================================================================
# # Set default parameters (all 0), for the original 8-variable model...
# # Later on we discovered that a 6-variable model was enough, see
# # comments above for the bounds.
# # ========================================================================
def setDefaultSimParms():
    # This is not very logical code-wise, but I'll try to optimize this in pairs, and use the results of the
    # previous optimization to initialize the values for the next one.... Let's hope it works!!!!
    parms = np.empty(8)

    # Excitatory: 1 + a + b * Abeta
    parms[0] = 0; parms[1] = 0
    # Excitatory: 1 + c + d * Tau
    parms[2] = 0; parms[3] = 0

    # Inhibitory: 1 + e + f * Abeta
    parms[4] = 0; parms[5] = 0  # x0=( 0.4,-0.5), f= 0.24208442, f(0,0)=0.39287662337662338
    # Inhibitory: 1 + g + h * Tau
    parms[6] = 0; parms[7] = 0  # Those are not really used, see the bounds defined above...

    return parms


# # ====================================================================================
# # The full 6D function
# # ====================================================================================
a6 = 0; b6 = 1; c6 = 2; d6 = 3; e6 = 4; f6 = 5
def func6D(optParms):
    parms = setDefaultSimParms()
    # Excitatory parms
    parms[a6] = optParms[0]  # optParms[0] + optParms[1] * Abeta
    parms[b6] = optParms[1]
    parms[c6] = optParms[2]  # optParms[2] + optParms[3] * Tau
    parms[d6] = optParms[3]
    # Inhibitory parms
    parms[e6] = optParms[4]  # optParms[4] + optParms[5] * Abeta
    parms[f6] = optParms[5]
    # Tau has no effect on inhibitory neurons, so no [6] and [7] parms used...
    return func(parms)


# ========================================================================
# This is the actual function to optimize... It is usually never called
# directly, but through the wrappers above...
# ========================================================================
measure = None
N = None
applyFilters = None
# angles_emp = None
processedEmp = None
trials = 10
# cachePath = None
parmToEval = None

# @vectorCache(filePath=cachePath)
def func(x):
    # print("   Going to eval:", x, flush=True, end='')
    neuronalModel.setParms({parmToEval:x})
    integrator.recompileSignatures()
    measureValues = measure.init(trials, N)
    for i in range(trials):
        bds = simulateBOLD.simulateSingleSubject().T
        procSignal = measure.from_fMRI(bds, applyFilters=applyFilters)
        measureValues = measure.accumulate(measureValues, i, procSignal)

    measureValues = measure.postprocess(measureValues)
    r = measure.distance(measureValues, processedEmp)
    # print("  Value:", r, flush=True)
    return r

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
