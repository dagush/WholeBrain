# ================================================================================================================
# This prog. plots the max frec for varying global couplings (G)
#
# see:
#  [NaskarEtAl_2018] Amit Naskar, Anirudh Vattikonda, Gustavo Deco,
#      Dipanjan Roy, Arpan Banerjee; Multiscale dynamic mean field (MDMF)
#      model relates resting-state brain dynamics with local cortical
#      excitatory–inhibitory neurotransmitter homeostasis.
#      Network Neuroscience 2021; 5 (3): 757–782. doi: https://doi.org/10.1162/netn_a_00197
#
# By Gustavo Patow
# ================================================================================================================
import numpy as np
import scipy.io as sio
# import os, csv
# from pathlib import Path
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
from setup import *
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# # ============== chose a model
# import Models.DynamicMeanField as DMF
# import Models.Couplings as Couplings
# # ============== chose and setup an integrator
# import Integrators.EulerMaruyama as scheme
# scheme.neuronalModel = DMF
# import Integrators.Integrator as integrator
# integrator.integrationScheme = scheme
# integrator.neuronalModel = DMF
# integrator.verbose = False
# # ============== chose a FIC mechanism
# import Utils.FIC.BalanceFIC as BalanceFIC
# BalanceFIC.integrator = integrator
# import Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
# BalanceFIC.balancingMechanism = Deco2014Mechanism  # default behaviour for this project

# np.random.seed(42)  # Fix the seed for debug purposes...


def plotMaxFrecForAllG(SC):
    # Integration parms...
    dt = 0.1
    tmax = 9 * 60 * 1000.
    Tmaxneuronal = int((tmax+dt))
    # ---------------------------- DEBUG
    wStart = 0.
    wEnd = 1.5
    wStep = 0.1
    # ---------------------------- END DEBUG
    # all tested global couplings (G in the paper):
    Gs = np.arange(wStart, wEnd, wStep)  # warning: the range of wes depends on the conectome.
    # N = SC.shape[0]

    print("==========================================")
    print("=    simulating freq evaluation (Naskar) =")
    print("==========================================")
    maxRateNoFIC = np.zeros(len(Gs))
    # DMF.setParms({'J': np.ones(N)})  # E-E = Excitatory-Excitatory, no FIC...
    for kk, G in enumerate(Gs):  # iterate over the weight range (G in the paper, we here)
        print("Processing: {}".format(G), end='')
        Naskar.setParms({'G': G})
        # integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:,1,:]  # [1] is the output from the excitatory pool, in Hz.
        maxRateNoFIC[kk] = np.max(np.mean(v,0))
        print(" => {}".format(maxRateNoFIC[kk]))
    ee, = plt.plot(Gs, maxRateNoFIC)
    ee.set_label("Naskar")

    # print("======================================")
    # print("=    simulating FIC                  =")
    # print("======================================")
    # # DMF.lambda = 0.  # make sure no long-range feedforward inhibition (FFI) is computed
    # maxRateFIC = np.zeros(len(wes))
    # # if precompute:
    # #     BalanceFIC.Balance_AllJ9(C, wes,
    # #                              baseName=fileName)
    # for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
    #     print("\nProcessing: {}  ".format(we), end='')
    #     DMF.setParms({'we': we})
    #     # balancedJ = BalanceFIC.Balance_J9(we, C, fileName.format(np.round(we, decimals=2)))['J'].flatten()
    #     # integrator.neuronalModel.setParms({'J': balancedJ})
    #     # integrator.recompileSignatures()
    #     v = integrator.simulate(dt, Tmaxneuronal)[:,1,:]  # [1] is the output from the excitatory pool, in Hz.
    #     maxRateFIC[kk] = np.max(np.mean(v,0))
    #     print("maxRateFIC => {}".format(maxRateFIC[kk]))
    # fic, = plt.plot(wes, maxRateFIC)
    # fic.set_label("FIC")

    # for line, color in zip([1.47, 4.45], ['r','b']):
    #     plt.axvline(x=line, label='line at x = {}'.format(line), c=color)
    plt.title("Large-scale network (DMF)")
    plt.ylabel("Maximum rate (Hz)")
    plt.xlabel("Global Coupling G")
    plt.legend()
    plt.show()


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 15})

    # ============= For debug only! =============
    import WholeBrain.Utils.decorators as decorators
    decorators.forceCompute = True
    # ===========================================

    # # Simple verification test, to check the info from the paper...
    # print(f"Simple test for verification: phie={DMF.phie(-0.026+DMF.be/DMF.ae)}")
    # print("Should print result: phie 3.06308542427")

    # print("Running single node...")
    # N = 1
    # DMF.we = 0.
    # C = np.zeros((N,N))  # redundant, I know...
    # DMF.J = np.ones(N)
    # runAndPlotSim(C, "Single node simulation")

    # Load connectome:
    # --------------------------------
    # inFilePath = '../../Data_Raw'
    # outFilePath = '../../Data_Produced'
    # CFile = sio.loadmat(inFilePath + '/Human_66.mat')  # load Human_66.mat C
    # C = CFile['C']
    # fileName = outFilePath + '/Human_66/Benji_Human66_{}.mat'  # integrationMode+'Benji_Human66_{}.mat'

    # ================================================================
    # This plots the graphs at Fig 2c of [D*2014]
    plotMaxFrecForAllG(sc68)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
