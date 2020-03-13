# ================================================================================================================
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
import os, csv
from pathlib import Path
import matplotlib.pyplot as plt
import functions.Models.DynamicMeanField as DMF
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = DMF
# import functions.Balance_J9 as Balance_J9
# Balance_J9.integrator = integrator
import DecoEtAl2014_GetBalancedWeights as getJs

np.random.seed(42)  # Fix the seed for debug purposes...


# filePath = 'Data_Produced/BenjiBalancedWeights-py.mat'
# def computeAllJs(C, wStart=0, wEnd=6+0.001, wStep=0.05):
#     wes = np.arange(wStart + wStep, wEnd, wStep)  # warning: the range of wes depends on the conectome.
#     # ==== J is calculated this only once, then saved
#     if not Path(filePath).is_file():
#         print("Computing "+ filePath +" !!!")
#         getJs.subjectPath = filePath
#         JI = getJs.computeAllJs(C, wStart, wEnd, wStep)
#     else:
#         print("Loading "+ filePath +" !!!")
#         # ==== J can be calculated only once and then load J_Balance J
#         JIfile = sio.loadmat(filePath)
#         JI = JIfile['JI']
#     return JI, wes


def runAndPlotSim(Conn, title):
    print("Running simulation for we={}...".format(DMF.we))

    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax+dt))

    N = Conn.shape[0]
    DMF.initBookkeeping(N, tmax)
    integrator.simulate(dt, Tmaxneuronal, Conn)
    v = DMF.returnBookkeeping()[1]  # [1] is the output from the excitatory pool, in Hz.

    f = np.mean(v,0)  # takes the mean of all xn values along dimension 0...
                      # This is the "averaged level of the input of the local excitatory pool of each brain area,
                      # i.e., I_i^{(E)}" in the text (pp 7889, right column, subsection "FIC").
    print('Finished sim for we={}. Mean of means = {}'.format(DMF.we, np.mean(f)))
    plt.bar(np.arange(N)+1, f)
    plt.title(title + ' (we={})'.format(DMF.we))
    plt.xlabel('Cortical Area')
    plt.ylabel('freq')
    plt.show()


def plotMaxFrecForAllWe(C, wStart=0, wEnd=6+0.001, wStep=0.05):
    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax+dt))
    # all tested global couplings (G in the paper):
    wes = np.arange(wStart + wStep, wEnd, wStep)  # warning: the range of wes depends on the conectome.
    # numW = wes.size  # length(wes);
    N = C.shape[0]

    DMF.SC = C

    # print("======================================")
    # print("=    simulating E-E (no FIC)         =")
    # print("======================================")
    # maxRateNoFIC = np.zeros(len(wes))
    # DMF.J = np.ones(N)  # E-E = No FIC...
    # for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
    #     print("Processing: {}".format(we))
    #     DMF.we = we
    #     DMF.initBookkeeping(N, tmax)
    #     integrator.simulate(dt, Tmaxneuronal)
    #     v = DMF.returnBookkeeping()[1]  # [1] is the output from the excitatory pool, in Hz.
    #     maxRateNoFIC[kk] = np.max(np.mean(v,0))
    # ee, = plt.plot(wes, maxRateNoFIC)
    # ee.set_label("E-E")

    print("======================================")
    print("=    simulating FIC                  =")
    print("======================================")
    # DMF.lambda = 0.  # make sure no long-range feedforward inhibition (FFI) is computed
    maxRateFIC = np.zeros(len(wes))
    getJs.subjectPath = filePath
    JI = getJs.computeAllJs(C, wStart, wEnd, wStep)
    # JI, origWes = computeAllJs(C, wStart, wEnd, wStep)
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("Processing: {}".format(we))
        DMF.we = we
        wePos, = np.where(np.isclose(wes, we))  #origwes may be differeent than the wes we are using now... be careful!!
        DMF.J = JI[:,wePos].flatten()
        DMF.initBookkeeping(N, tmax)
        integrator.simulate(dt, Tmaxneuronal)
        v = DMF.returnBookkeeping()[1]  # [1] is the output from the excitatory pool, in Hz.
        maxRateFIC[kk] = np.max(np.mean(v,0))
    fic, = plt.plot(wes, maxRateFIC)
    fic.set_label("FIC")

    for line, color in zip([1.47, 4.45], ['r','b']):
        plt.axvline(x=line, label='line at x = {}'.format(line), c=color)
    plt.title("Large-scale network (DMF)")
    plt.ylabel("Maximum rate (Hz)")
    plt.xlabel("Global Coupling (G = we)")
    plt.legend()
    plt.show()


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
if __name__ == '__main__':
    integrator.verbose = False

    # Simple verification test, to check the info from the paper...
    print("Simple test: phie={}".format(DMF.phie(-0.026+DMF.be/DMF.ae)))
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
    CFile = sio.loadmat('Data_Raw/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']

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

    plotMaxFrecForAllWe(C)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
