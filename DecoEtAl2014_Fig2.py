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
import matplotlib.pyplot as plt
import functions.Models.DynamicMeanField as DMF
import functions.Integrator_EulerMaruyama as integrator
integrator.neuronalModel = DMF
import functions.Balance_J9 as Balance_J9
Balance_J9.integrator = integrator


np.random.seed(42)  # Fix the seed for debug purposes...


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


def plotMaxFrecForAllWe(C):
    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax+dt))
    # all tested global couplings (G in the paper):
    wStart = 0
    wEnd = 6 + 0.001
    wStep = 0.05
    wes = np.arange(wStart + wStep,
                    wEnd,
                    wStep)  # .05:0.05:2; #.05:0.05:4.5; # warning: the range of wes depends on the conectome.
    maxRateNoFIC = np.zeros(len(wes))
    # numW = wes.size  # length(wes);
    N = C.shape[0]

    DMF.J = np.ones(N)  # No FIC...
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("Processing: {}".format(we))
        DMF.initBookkeeping(N, tmax)
        integrator.simulate(dt, Tmaxneuronal, C)
        v = DMF.returnBookkeeping()[1]  # [1] is the output from the excitatory pool, in Hz.
        maxRateNoFIC[kk] = np.max(np.mean(v,0))

    plt.plot(wes, maxRateNoFIC)
    for line, color in zip([1.47, 4.45], ['r','b']):
        plt.axvline(x=line, label='line at x = {}'.format(line), c=color)
    plt.title("Large-scale network (DMF)")
    plt.ylabel("Maximum rate (Hz)")
    plt.xlabel("Global Coupling (G = we)")
    plt.show()


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
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
