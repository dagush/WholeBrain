a# ================================================================================================================
#
# This prog. plots the max frec for varying global couplings (G)
#
# see:
# [D*2014]  Deco et al. (2014) J Neurosci.
#           http://www.jneurosci.org/content/34/23/7886.long
#
# By Gustavo Patow
# ================================================================================================================
import numpy as np
import scipy.io as sio
# import os, csv
# from pathlib import Path
import matplotlib.pyplot as plt
import WholeBrain.Models.DynamicMeanField as DMF
# ============== chose an integrator
import WholeBrain.Integrator_EulerMaruyama as integrator
# integrationMode = ''
# import WholeBrain.Integrator_Euler as integrator
# integrationMode = 'Deterministic/'
# ============== done !!!
integrator.neuronalModel = DMF
import WholeBrain.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator

np.random.seed(42)  # Fix the seed for debug purposes...


def plotMaxFrecForAllWe(C, wStart=0, wEnd=6+0.001, wStep=0.05,
                        extraTitle='', precompute=True, fileName=None):
    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax+dt))
    # all tested global couplings (G in the paper):
    wes = np.arange(wStart + wStep, wEnd, wStep)  # warning: the range of wes depends on the conectome.
    # wes = np.arange(2.0, 2.11, 0.1)  # only for debug purposes...
    # numW = wes.size  # length(wes);
    N = C.shape[0]

    DMF.setParms({'SC': C})

    print("======================================")
    print("=    simulating E-E (no FIC)         =")
    print("======================================")
    maxRateNoFIC = np.zeros(len(wes))
    DMF.setParms({'J': np.ones(N)})  # E-E = Excitatory-Excitatory, no FIC...
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("Processing: {}".format(we), end='')
        DMF.setParms({'we': we})
        integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:,1,:]  # [1] is the output from the excitatory pool, in Hz.
        maxRateNoFIC[kk] = np.max(np.mean(v,0))
        print(" => {}".format(maxRateNoFIC[kk]))
    ee, = plt.plot(wes, maxRateNoFIC)
    ee.set_label("E-E")

    print("======================================")
    print("=    simulating FIC                  =")
    print("======================================")
    # DMF.lambda = 0.  # make sure no long-range feedforward inhibition (FFI) is computed
    maxRateFIC = np.zeros(len(wes))
    if precompute:
        BalanceFIC.Balance_AllJ9(C, wes,  # wStart, wEnd, wStep,
                                 baseName=fileName)
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("\nProcessing: {}  ".format(we), end='')
        DMF.setParms({'we': we})
        balancedJ = BalanceFIC.Balance_J9(we, C, fileName.format(np.round(we, decimals=2)))['J'].flatten()
        integrator.neuronalModel.setParms({'J': balancedJ})
        integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:,1,:]  # [1] is the output from the excitatory pool, in Hz.
        maxRateFIC[kk] = np.max(np.mean(v,0))
        print("maxRateFIC => {}".format(maxRateFIC[kk]))
    fic, = plt.plot(wes, maxRateFIC)
    fic.set_label("FIC")

    for line, color in zip([1.47, 4.45], ['r','b']):
        plt.axvline(x=line, label='line at x = {}'.format(line), c=color)
    plt.title("Large-scale network (DMF)" + extraTitle)
    plt.ylabel("Maximum rate (Hz)")
    plt.xlabel("Global Coupling (G = we)")
    plt.legend()
    plt.show()


# def debug_one_we(C):
#     DMF.SC = C
#     dt = 0.1
#     tmax = 10000.
#     Tmaxneuronal = int((tmax+dt))
#     we = 2.1
#
#     DMF.we = we
#     # wePos, = np.where(np.isclose(wes, we))  #origwes may be differeent than the wes we are using now... be careful!!
#     DMF.J = np.ones(C.shape[0])  # E-E = Excitatory-Excitatory, no FIC...
#     print("Processing: {}".format(we), end='')
#     integrator.recompileSignatures()
#     v = integrator.simulate(dt, Tmaxneuronal)[:,1,:]  # [1] is the output from the excitatory pool, in Hz.
#     maxRateFIC = np.max(np.mean(v,0))
#     print("=> {}".format(maxRateFIC))


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
if __name__ == '__main__':
    integrator.verbose = False

    # Simple verification test, to check the info from the paper...
    # print("Simple test for verification: phie={}".format(DMF.phie(-0.026+DMF.be/DMF.ae)))
    # result: phie 3.06308542427

    plt.rcParams.update({'font.size': 15})

    # print("Running single node...")
    # N = 1
    # DMF.we = 0.
    # C = np.zeros((N,N))  # redundant, I know...
    # DMF.J = np.ones(N)
    # runAndPlotSim(C, "Single node simulation")

    # Load connectome:
    # --------------------------------
    inFilePath = '../Data_Raw'
    outFilePath = '../Data_Produced'
    CFile = sio.loadmat(inFilePath + '/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']
    fileName = outFilePath + '/Human_66/Benji_Human66_{}.mat'  # integrationMode+'Benji_Human66_{}.mat'

    # ================================================================
    # This plots the graphs at Fig 2, d of [D*2014]
    # DMF.J = np.ones(N)
    #
    # print("Running connectivity matrix with we = 0...")
    # DMF.we = 0.
    # runAndPlotSim(C, "Full connectivity matrix simulation, no long range interactions (No FIC)")
    #
    # print("Running connectivity matrix with we = 1.2...")
    # DMF.we = 1.2
    # runAndPlotSim(C, "Full connectivity matrix simulation (No FIC)")
    #
    # print("Running connectivity matrix with we = 2.1...")
    # DMF.we = 2.1
    # runAndPlotSim(C, "Full connectivity matrix simulation (No FIC)")
    #
    # print("Running connectivity matrix with FIC control, with we = 2.1...")
    # DMF.we = 2.1
    # J = Balance_J9.JOptim(C, DMF.we)
    # DMF.J = J
    # runAndPlotSim(C, "Full connectivity matrix simulation with FIC control")

    plotMaxFrecForAllWe(C, fileName=fileName)
    # debug_one_we(C)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
