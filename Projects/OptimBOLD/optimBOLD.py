# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# BOLD model parameter optimization. Based on:
#
# * K.J. Friston, L. Harrison, and W. Penny,
#   Dynamic causal modelling, NeuroImage 19 (2003) 1273–1302
# * Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
#   Comparing hemodynamic models with DCM, NeuroImage 38 (2007) 387–401
#
# Later revisited in
# * Klaas Enno Stephan, Lars Kasper, Lee M. Harrison, Jean Daunizeau, Hanneke E.M. den Ouden, Michael Breakspear, and Karl J. Friston
#   Nonlinear Dynamic Causal Models for fMRI, Neuroimage. 2008 Aug 15; 42(2): 649–662.
#
# Also, check:
# * K.J. Friston, Katrin H. Preller, Chris Mathys, Hayriye Cagnan, Jakob Heinzle, Adeel Razi, Peter Zeidman
#   Dynamic causal modelling revisited, NeuroImage 199 (2019) 730–744
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
import simulateFCD
from WholeBrain import BOLDHemModel_Stephan2008 as BOLDModel
from WholeBrain.Utils import errorMetrics
from Observables import BOLDFilters

optimizationVars = {
    'kappa': BOLDModel.kappa,
    'gamma': BOLDModel.gamma,
    'tau': BOLDModel.tau,
    'alpha': BOLDModel.alpha,
    'epsilon': BOLDModel.epsilon,
}
varNames = ['epsilon', 'alpha', 'tau', 'gamma', 'kappa']
numVars = len(varNames)


kappa0 = 0.65  # 0.8;    # Rate of vasodilatory signal decay, time unit (s) [Friston2003], eta = 0.64 in [Friston2019]
gamma0 = 0.41  # 0.4;    # Rate of flow-dependent elimination, time unit (s)  [Friston2003], chi = 0.32 in [Friston2019]
tau0 = 0.98  # 1;      # mean transit time (s) in [Friston2003], 1/tau = 2 in [Friston2019]
alpha0 = 0.32 #0.32; % 0.2;    % Grubb's exponent (a stiffness exponent) [Friston2003] and [Friston2019]
epsilon0 = 0.34 # Intravascular:extravascular ratio, value from [Obata et al. 2004]
initialValues = [epsilon0, alpha0, tau0, gamma0, kappa0]
initialLowerBounds = [0, 0.01, 0.01, 0.01, 0]  # [0, 0.03, 0.07, 0.09, 0]

numMethod = 'trf'  # 'lm' or 'trf'
Verbose = False
evalCounter = 0


def pairVarsAndValues(popt):
    return dict(zip(varNames, popt))


def fitBOLDBrainArea(neuronal_act, BOLDSignal, area, lowerBounds = initialLowerBounds):
    def errorFunc(neuro_act, epsilon, alpha, tau, gamma, kappa):
        if Verbose:
            global evalCounter
            evalCounter += 1
            print("Test:", evalCounter)
        BOLDModel.epsilon = epsilon
        BOLDModel.alpha = alpha
        BOLDModel.tau = tau
        BOLDModel.gamma = gamma
        BOLDModel.kappa = kappa
        bds = simulateFCD.computeSubjectBOLD(neuro_act, areasToSimulate=[area])
        bdsT = bds.T
        sim_filt = BOLDFilters.filterBrainArea(bdsT, 0)
        sim_filt /= np.std(sim_filt)
        return sim_filt

    # init...
    from scipy.optimize import curve_fit
    emp_filt = BOLDFilters.filterBrainArea(BOLDSignal, area)
    emp_filt /= np.std(emp_filt)

    # Now, fit it !!!
    # Note: Be careful with max_nfev, evaluations include numerical derivatives, so the actual number will be around 4.6 (or more) * this number before stopping!!!
    if numMethod == 'trf':
        popt, pcov = curve_fit(errorFunc, neuronal_act, emp_filt, method='trf', bounds=(lowerBounds, 3.0), p0 = initialValues, max_nfev = 100)
    else: # use 'lm'
        popt, pcov = curve_fit(errorFunc, neuronal_act, emp_filt, method='lm', p0 = initialValues)
    final_values = errorFunc(neuronal_act, popt[0], popt[1], popt[2], popt[3], popt[4])
    finalError = errorMetrics.l2(emp_filt, final_values)
    return popt, pcov, finalError


maxExhausted = 10  # Maximum number of trials before giving up...


def fitBOLDBrainAreaCatchingErrors2(neuro_act, BOLDSignal, area):  # random-based variant!
    def generateLowerBound():
        lowerBounds[1] = (1-0.01) * np.random.rand() + 0.01
        lowerBounds[2] = (1.3-0.03) * np.random.rand() + 0.03
        lowerBounds[3] = (1.3-0.03) * np.random.rand() + 0.03
    str_error = "Not executed yet."
    exhausted = 0
    lowerBounds = initialLowerBounds
    while str_error and not exhausted >= maxExhausted:
        try:
            popt, pcov, finalError = fitBOLDBrainArea(neuro_act, BOLDSignal, area,  lowerBounds)
            str_error = None
        except (ValueError, RuntimeError) as e:
            generateLowerBound()
            exhausted += 1
            print("ValueError|RuntimeError Exception! Changing lower bounds... ({}): now".format(exhausted), lowerBounds)
    if exhausted > 0:
        print("Had to recompute bounds {} times".format(exhausted+1), "final bounds:", lowerBounds)
        if exhausted == maxExhausted:
            print("And couldn't find the right value!!! \nNeeds re-computing...")
            popt, pcov, finalError = initialValues, np.ones(len(initialValues))*np.inf, None
    return popt, pcov, finalError


def fitBOLDBrainAreaCatchingErrors(neuro_act, BOLDSignal, area):  # Deterministic (but imprecise and unreliable) version
    str_error = "Not executed yet."
    exhausted = 0
    lowerBounds = initialLowerBounds
    while str_error and not exhausted >= maxExhausted:
        try:
            # replace line below with your logic , i.e. time out, max attempts
            popt, pcov, finalError = fitBOLDBrainArea(neuro_act, BOLDSignal, area,  lowerBounds)
            str_error=None
        except ValueError:
            lowerBounds[1] += 0.01; lowerBounds[2] += 0.02; lowerBounds[3] += 0.02
            exhausted += 1
            print("ValueError Exception! Increasing lower bounds... ({}): now".format(exhausted), lowerBounds)
        except RuntimeError:
            if lowerBounds[2] == 0.01:
                lowerBounds[1] = 0.05; lowerBounds[2] = 0.05; lowerBounds[3] = 0.05
            else:
                lowerBounds[1] -= 0.01; lowerBounds[2] -= 0.01; lowerBounds[3] -= 0.01
            exhausted += 1
            print("RuntimeError Exception! Changing lower bounds... ({}): now".format(exhausted), lowerBounds)
    if exhausted > 0:
        print("Had to recompute bounds {} times".format(exhausted+1), "final bounds:", lowerBounds)
        if exhausted == maxExhausted:
            print("And couldn't find the right value!!! \nNeeds re-computing...")
            popt, pcov, finalError = initialValues, np.ones(len(initialValues))*np.inf, np.inf
    return popt, pcov, finalError

