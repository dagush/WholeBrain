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
from pathlib import Path
# import functions.Models.DynamicMeanField as DMF
integrator = None  # import functions.Integrator_EulerMaruyama as integrator
# integrator.neuronalModel = DMF


np.random.seed(42)  # Fix the seed for debug purposes...


# ======================================================================
# ======================================================================
# ======================================================================
# filePath = 'Data_Produced/BenjiBalancedWeights-py.mat'
subjectPath = 'Data_Produced/BenjiBalancedWeights_{}_kk.mat'
def computeSingleJ(C, we):
    import functions.BalanceFIC as BalanceFIC
    BalanceFIC.integrator = integrator

    N = C.shape[0]
    filePath = subjectPath.format(we)
    # ==== J is calculated this only once, then saved
    if not Path(filePath).is_file():
        print("Computing "+ filePath +" !!!")

        integrator.neuronalModel.we = we
        J, nodesSolved = BalanceFIC.JOptim(C)

        # Se_init = integrator.simVars[0].reshape(N)  # Store steady states S^E (after many iterations/simulations) -> sn
        # Si_init = integrator.simVars[1].reshape(N)   # Store steady states S^I -> sg
        sio.savemat(filePath, #{'JI': JI})
                    {'we': we,
                     'J': J,
                     # 'Se_init': Se_init,
                     # 'Si_init': Si_init
                    })  # save Benji_Balanced_weights wes JI Se_init Si_init
    else:
        print("Loading "+ filePath +" !!!")
        # ==== J can be calculated only once and then load J_Balance J
        JIfile = sio.loadmat(filePath)
        Jpos = np.where(JIfile['we']==we)[0][0]
        J = JIfile['J'][Jpos]
        nodesSolved = len(J)
    return J, nodesSolved


def computeAllJs(C, wStart=0, wEnd=6+0.001, wStep=0.05):
    # all tested global couplings (G in the paper):
    wes = np.arange(wStart + wStep,
                    wEnd,
                    wStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of wes depends on the conectome.
    numW = wes.size  # length(wes);

    # ==========================
    # Some monitoring info: initialization
    N = C.shape[0]
    failedGs = []
    JI=np.zeros((N,numW))
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        J, nodesSolved = computeSingleJ(C, we)
        JI[:,kk] = J
        if nodesSolved < N:
            failedGs.append((we, nodesSolved))

    # sio.savemat(filePath, #{'JI': JI})
    #             {'wes': wes,
    #              'JI': JI,
    #              'Se_init': Se_init,
    #              'Si_init': Si_init})  # save Benji_Balanced_weights wes JI Se_init Si_init

    print('Failed G computations:', failedGs)
    return JI

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
if __name__ == '__main__':
    integrator.verbose = False

    # Simple verification test, to check the info from the paper...
    # I_e = -0.026+DMF.be/DMF.ae
    # print("phie",DMF.phie(I_e))
    # result: phie 3.06308542427

    # Load connectome:
    # --------------------------------
    CFile = sio.loadmat('Data_Raw/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']
    subjectPath = 'Data_Produced/Human_66/Benji_Human66_{}.mat'
    # import functions.Utils.plotSC as plotSC
    # plotSC.plotSC_and_Histogram("Human_66", np.log(C+1))

    JI = computeAllJs(C)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
