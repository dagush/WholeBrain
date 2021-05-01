# --------------------------------------------------------------------------------------
# Functions for AD subject processing
#
# --------------------------------------------------------------------------------------
import numpy as np
from functions.Utils.decorators import vectorCache  # loadOrCompute, loadSingleCache, loadMultipleCache
neuronalModel = None
integrator = None
simulateBOLD = None

# =====================================================================
# =====================================================================
#              Definitions for Single Subject Pipeline
# =====================================================================
# =====================================================================
parmLabels = [r'A$\beta^e_b$', r'A$\beta^e_s$',
              r'$\tau^e_b$', r'$\tau^e_s$',
              r'A$\beta^i_b$', r'A$\beta^i_s$',
              # no Tau inhibitory
              ]

bounds = [(-1.0, 1.0), (0.0, 4.0),  # Abeta increases activity of excitatory neurons > 0
          (-1.0, 1.0), (-4.0, 0.0), # Tau silences < 0
          (-1.0, 1.0), (-4.0, 0.0)  # Abeta reduces inhibition < 0
          # Tau has no effect on inhibitory neurons ~ 0
          ]

# ========================================================================
# AD specific optim
# ========================================================================
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
    parms[6] = 0; parms[7] = 0  # Those are not really used, see the bounds defined above

    return parms


# ========================================================================
# A nice set of proxy functions to have only the parameters we need...
# ========================================================================
aParmRange = None
bParmRange = None
aVar = 4;  bVar = 5  # Used for 2D brute-force optimization
def func2D(optParms):  # 2D wrapper for brute force optimization
    parms = setDefaultSimParms()
    parms[aVar] = optParms[0]  # optParms[0] + optParms[1] * tau
    parms[bVar] = optParms[1]
    return func(parms)


# a4 = 0; b4 = 1; c4 = 2; d4 = 3
# def func4D(optParms):
#     parms = setDefaultSimParms()
#     parms[a4] = optParms[0]  # optParms[0] + optParms[1] * Abeta
#     parms[b4] = optParms[1]
#     parms[c4] = optParms[2]  # optParms[2] + optParms[3] * Tau
#     parms[d4] = optParms[3]
#     return func(parms)


ctau = 2; dtau = 3
def func_full_Tau(optParms):
    parms = setDefaultSimParms()
    # Excitatory parms
    parms[ctau] = optParms[0]  # optParms[2] + optParms[3] * Tau
    parms[dtau] = optParms[1]
    # Tau has no effect on inhibitory neurons, so no [6] and [7] parms used...
    return func(parms)


aABeta = 0; bABeta = 1; eABeta = 4; fABeta = 5
def func_full_ABeta(optParms):
    parms = setDefaultSimParms()
    # Excitatory parms
    parms[aABeta] = optParms[0]  # optParms[0] + optParms[1] * Abeta
    parms[bABeta] = optParms[1]
    # Inhibitory parms
    parms[eABeta] = optParms[2]  # optParms[2] + optParms[3] * Abeta
    parms[fABeta] = optParms[3]
    return func(parms)


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
# The actual function to optimize...
# ========================================================================
measure = None
N = None
applyFilters = None
SC = None
angles_emp = None
trials = 10
# cachePath = None

# @vectorCache(filePath=cachePath)
def func(x):
    print("   Going to eval:", x, flush=True, end='')
    neuronalModel.set_AD_Burden(x)
    integrator.recompileSignatures()
    measureValues = measure.init(trials, N)
    for i in range(trials):
        bds = simulateBOLD.simulateSingleSubject(SC, warmup=False).T
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
    r = measure.distance(measureValues, angles_emp)
    print("  Value:", r, flush=True)
    return r
