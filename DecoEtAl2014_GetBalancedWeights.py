# ================================================================================================================
#
# This prog. optimizes the strengh of the feedback inhibition of the FIC model 
# for varying global couplings (G)
# Saves the steady states and the feedback inhibition (J).
#
# see:
# [DecoEtAl2014] Gustavo Deco, Adrián Ponce-Alvarez, Patric Hagmann, Gian Luca Romani, Dante Mantini and Maurizio
#           Corbetta, "How Local Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics" (2014), Journal of
#           Neuroscience 4 June 2014, 34 (23) 7886-7898; DOI: https://doi.org/10.1523/JNEUROSCI.5068-13.2014
#
# Original code by Adrian Ponce-Alvarez. Refactoring by Gustavo Patow
# ================================================================================================================

import numpy as np
import scipy.io as sio
import functions.Models.DynamicMeanField as DMF
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = DMF
import functions.Balance_J9 as Balance_J9
Balance_J9.integrator = integrator


np.random.seed(42)  # Fix the seed for debug purposes...


# ======================================================================
# ======================================================================
# ======================================================================
def computeAllJs(C):
    # all tested global couplings (G in the paper):
    wStart = 0
    wEnd = 6 + 0.001  # 2
    wStep = 0.05
    wes = np.arange(wStart + wStep,
                    wEnd,
                    wStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of wes depends on the conectome.
    numW = wes.size  # length(wes);

    # ==========================
    # Some monitoring info: initialization
    N = C.shape[0]
    JI=np.zeros((N,numW))
    Se_init = np.zeros((N,numW))
    Si_init = np.zeros((N,numW))
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        J = Balance_J9.JOptim(C, we)
        Se_init[:, kk] = integrator.simVars[0].reshape(N)  # Store steady states S^E (after many iterations/simulations) -> sn
        Si_init[:, kk] = integrator.simVars[1].reshape(N)   # Store steady states S^I -> sg
        JI[:,kk]=J

    sio.savemat('Data_Produced/BenjiBalancedWeights-py.mat', #{'JI': JI})
                {'wes': wes,
                 'JI': JI,
                 'Se_init': Se_init,
                 'Si_init': Si_init})  # save Benji_Balanced_weights wes JI Se_init Si_init

    return JI

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
integrator.verbose = False

# Simple verification test, to check the info from the paper...
I_e = -0.026+DMF.be/DMF.ae
print("phie",DMF.phie(I_e))
# result: phie 3.06308542427

# Load connectome:
# --------------------------------
CFile = sio.loadmat('Data_Raw/Human_66.mat')  # load Human_66.mat C
C = CFile['C']

JI = computeAllJs(C)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
