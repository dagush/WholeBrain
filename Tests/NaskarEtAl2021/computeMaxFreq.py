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
# import numpy as np
# import scipy.io as sio
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


def plotMaxFrecForAllG(SC):
    # Integration parms...
    dt = 0.1
    tmax = 8 * 60 * 1000.
    twarmup = 1 * 60 * 1000.
    Tmaxneuronal = int((tmax+dt))
    # ---------------------------- DEBUG
    wStart = 0.
    wEnd = 1.50001
    wStep = 0.1
    # ---------------------------- END DEBUG
    # all tested global couplings (G in the paper):
    Gs = np.arange(wStart, wEnd, wStep)  # warning: the range of wes depends on the conectome.

    print("==========================================")
    print("=    simulating freq evaluation (Naskar) =")
    print("==========================================")
    maxRateNoFIC = np.zeros(len(Gs))
    for kk, G in enumerate(Gs):  # iterate over the weight range (G in the paper, we here)
        print("Processing: {}".format(G), end='')
        Naskar.setParms({'G': G})
        v = integrator.warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=twarmup)[:,1,:]  # [1] is the output from the excitatory pool, in Hz.
        # maxRateNoFIC[kk] = np.max(np.mean(v, axis=0))  # the original code from [DecoEtAl_2014]
        maxRateNoFIC[kk] = np.mean(np.max(v, axis=0))  # this is what is implemented in the code [NaskarEtAl_2018].
        print(" => {}".format(maxRateNoFIC[kk]))
    ee, = plt.plot(Gs, maxRateNoFIC)
    ee.set_label("Naskar")

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
