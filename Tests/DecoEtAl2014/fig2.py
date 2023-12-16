# ================================================================================================================
#
# This prog. plots the max frec for varying global couplings (G)
#
# see:
# [D*2014] How Local Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics
#          Gustavo Deco, Adrián Ponce-Alvarez, Patric Hagmann, Gian Luca Romani, Dante Mantini and Maurizio Corbetta
#          Journal of Neuroscience 4 June 2014, 34 (23) 7886-7898;
#          DOI: https://doi.org/10.1523/JNEUROSCI.5068-13.2014
#
# By Gustavo Patow
# ================================================================================================================
import numpy as np
import scipy.io as sio
# import os, csv
# from pathlib import Path
import matplotlib.pyplot as plt

# ============== chose a model
import Models.DynamicMeanField as DMF
import Models.Couplings as Couplings
# ============== chose and setup an integrator
import Integrators.EulerMaruyama as scheme
scheme.neuronalModel = DMF
import Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.neuronalModel = DMF
integrator.verbose = False
# ============== chose a FIC mechanism
import Utils.FIC.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator
import Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
BalanceFIC.balancingMechanism = Deco2014Mechanism  # default behaviour for this project

np.random.seed(42)  # Fix the seed for debug purposes...


def plotMaxFrecForAllWe(C, wStart=0, wEnd=6+0.001, wStep=0.05,
                        extraTitle='', precompute=True, fileName=None):
    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax+dt))
    # all tested global couplings (G in the paper):
    wes = np.arange(wStart, wEnd, wStep)  # warning: the range of wes depends on the conectome.
    # wes = np.arange(2.0, 2.11, 0.1)  # only for debug purposes...
    # numW = wes.size  # length(wes);
    N = C.shape[0]

    DMF.setParms({'SC': C})
    DMF.couplingOp.setParms(C)

    print("======================================")
    print("=    simulating E-E (no FIC)         =")
    print("======================================")
    maxRateNoFIC = np.zeros(len(wes))
    DMF.setParms({'J': np.ones(N)})  # E-E = Excitatory-Excitatory, no FIC...
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("Processing: {}".format(we), end='')
        DMF.setParms({'we': we})
        # integrator.recompileSignatures()
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
        BalanceFIC.Balance_AllJ9(C, wes,
                                 baseName=fileName)
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("\nProcessing: {}  ".format(we), end='')
        DMF.setParms({'we': we})
        balancedJ = BalanceFIC.Balance_J9(we, C, fileName.format(np.round(we, decimals=2)))['J'].flatten()
        integrator.neuronalModel.setParms({'J': balancedJ})
        # integrator.recompileSignatures()
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


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 15})

    # Simple verification test, to check the info from the paper...
    print(f"Simple test for verification: phie={DMF.phie(-0.026+DMF.be/DMF.ae)}")
    print("Should print result: phie 3.06308542427")

    # print("Running single node...")
    # N = 1
    # DMF.we = 0.
    # C = np.zeros((N,N))  # redundant, I know...
    # DMF.J = np.ones(N)
    # runAndPlotSim(C, "Single node simulation")

    # Load connectome:
    # --------------------------------
    inFilePath = '../../Data_Raw'
    outFilePath = '../../Data_Produced'
    CFile = sio.loadmat(inFilePath + '/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']
    fileName = outFilePath + '/Human_66/Benji_Human66_{}.mat'  # integrationMode+'Benji_Human66_{}.mat'

    # ================================================================
    # This plots the graphs at Fig 2c of [D*2014]
    plotMaxFrecForAllWe(C, fileName=fileName)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
